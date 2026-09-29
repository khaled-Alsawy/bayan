from __future__ import annotations

import asyncio
import json
import re
from contextlib import asynccontextmanager
from uuid import UUID

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response, StreamingResponse
from pydantic import ValidationError

from islamic_research_ai.composition import AppContainer, build_container
from islamic_research_ai.contracts import (
    ConversationCreateRequest,
    ConversationDetailV1,
    ConversationPageV1,
    ConversationSummaryV1,
    ConversationTurnCreateRequest,
    ConversationTurnV1,
    ConversationUpdateRequest,
    ResearchRequest,
    ResearchResponseV1,
    TraceEventV1,
)
from islamic_research_ai.conversations import (
    ConversationArchivedError,
    ConversationNotFoundError,
    IdempotencyConflictError,
)
from islamic_research_ai.logging import configure_logging, log_event


@asynccontextmanager
async def lifespan(app: FastAPI):
    container = await build_container()
    app.state.container = container
    try:
        yield
    finally:
        await container.close()


app = FastAPI(title="Islamic Research AI POC", version="0.1.0", lifespan=lifespan)
_OWNER_KEY = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")


def _validation_error_detail(exc: ValidationError) -> list[dict[str, object]]:
    """Keep contract failures actionable without logging the invalid source payload itself."""

    detail: list[dict[str, object]] = []
    for error in exc.errors(include_url=False):
        invalid_input = error.get("input")
        detail.append(
            {
                "location": [str(part) for part in error["loc"]],
                "constraint": error["type"],
                "message": error["msg"],
                "context": error.get("ctx"),
                "invalid_collection_count": (
                    len(invalid_input)
                    if isinstance(invalid_input, (list, tuple, set, dict))
                    else None
                ),
            }
        )
    return detail


def _owner_key(request: Request) -> str:
    _require_private_or_bearer(request)
    value = request.headers.get("x-owner-key", "local-device")
    if not _OWNER_KEY.fullmatch(value):
        raise HTTPException(status_code=400, detail="Invalid owner boundary")
    return value


def _require_private_or_bearer(request: Request) -> None:
    """Keep the API loopback-only unless an explicit public bearer boundary exists."""
    container: AppContainer | None = getattr(request.app.state, "container", None)
    token = container.settings.public_access_token if container is not None else None
    client_host = request.client.host if request.client is not None else ""
    is_loopback = client_host in {"127.0.0.1", "::1", "localhost"}
    if token is None:
        if not is_loopback:
            raise HTTPException(status_code=403, detail="Public access is disabled")
        return
    supplied = request.headers.get("x-public-access-token")
    if supplied != token.get_secret_value():
        raise HTTPException(status_code=401, detail="Public access token required")


@app.get("/health")
async def health(request: Request) -> dict[str, object]:
    container: AppContainer | None = getattr(request.app.state, "container", None)
    if container is None:
        return {"status": "starting", "local_quota_fallback": "unknown"}
    readiness = await container.local_chat.readiness(run_schema_probe=True)
    container.local_readiness = readiness
    return {
        "status": "ok" if readiness.ready else "degraded",
        "primary_chat": "groq_direct",
        "local_quota_fallback": "ready" if readiness.ready else "unavailable",
        "tafsir_source": "local_qul",
    }


@app.post("/v1/research", response_model=ResearchResponseV1)
async def research(payload: ResearchRequest, request: Request) -> ResearchResponseV1:
    container: AppContainer = request.app.state.container
    _require_private_or_bearer(request)
    try:
        response, trace = await container.service.research(payload)
        log_event(
            "research_request_complete",
            turn_id=trace.identity.turn_id,
            candidate_count=len(trace.candidates),
            evidence_count=len(trace.evidence.items_by_id),
            status=response.status,
            # This is a bounded internal diagnostic, never returned to the client
            # and never canonical source text. It makes failed-safe evaluation
            # auditable without weakening the response contract.
            terminal_reason=trace.terminal_reason,
        )
        return response
    except ValidationError as exc:
        log_event(
            "research_request_internal_invariant_violation",
            error_type=type(exc).__name__,
            validation_model=exc.title,
            validation_errors=_validation_error_detail(exc),
        )
        raise HTTPException(status_code=503, detail="Research request failed safely") from exc
    except Exception as exc:
        log_event("research_request_failed_safe", error_type=type(exc).__name__)
        raise HTTPException(status_code=503, detail="Research request failed safely") from exc


@app.post("/v1/research/stream")
async def research_stream(payload: ResearchRequest, request: Request) -> StreamingResponse:
    """Owner-only development stream: emitted events and final response share one service run."""

    container: AppContainer = request.app.state.container
    _require_private_or_bearer(request)
    queue: asyncio.Queue[TraceEventV1 | ResearchResponseV1 | Exception] = asyncio.Queue()

    def on_event(event: TraceEventV1) -> None:
        queue.put_nowait(event)

    async def produce() -> None:
        try:
            response, _trace = await container.service.research(
                payload,
                on_trace_event=on_event if container.settings.developer_trace_enabled else None,
            )
            queue.put_nowait(response)
        except Exception as exc:  # Preserve already-emitted diagnostics before the terminal error.
            queue.put_nowait(exc)

    async def event_stream():
        task = asyncio.create_task(produce())
        while True:
            item = await queue.get()
            if isinstance(item, TraceEventV1):
                yield f"event: trace\ndata: {item.model_dump_json()}\n\n"
                continue
            if isinstance(item, ResearchResponseV1):
                yield f"event: final\ndata: {item.model_dump_json()}\n\n"
            else:
                body = json.dumps({"detail": "Research request failed safely"})
                yield f"event: error\ndata: {body}\n\n"
            break
        await task

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.post("/v1/conversations", response_model=ConversationSummaryV1, status_code=201)
async def create_conversation(
    payload: ConversationCreateRequest, request: Request
) -> ConversationSummaryV1:
    return await request.app.state.container.conversations.create(
        payload, owner_key=_owner_key(request)
    )


@app.get("/v1/conversations", response_model=ConversationPageV1)
async def list_conversations(
    request: Request,
    limit: int = 30,
    cursor: str | None = None,
    include_archived: bool = False,
) -> ConversationPageV1:
    if not 1 <= limit <= 100:
        raise HTTPException(status_code=422, detail="limit must be between 1 and 100")
    try:
        return await request.app.state.container.conversations.list(
            owner_key=_owner_key(request),
            include_archived=include_archived,
            limit=limit,
            cursor=cursor,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid pagination cursor") from exc


@app.get("/v1/conversations/{conversation_id}", response_model=ConversationDetailV1)
async def get_conversation(
    conversation_id: UUID,
    request: Request,
    limit: int = 50,
    cursor: str | None = None,
) -> Response | ConversationDetailV1:
    if not 1 <= limit <= 100:
        raise HTTPException(status_code=422, detail="limit must be between 1 and 100")
    try:
        conversation, next_cursor = await request.app.state.container.conversations.get(
            conversation_id,
            owner_key=_owner_key(request),
            turn_limit=limit,
            cursor=cursor,
        )
    except ConversationNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Conversation not found") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid pagination cursor") from exc
    headers = {"X-Next-Cursor": next_cursor} if next_cursor is not None else {}
    return Response(
        content=conversation.model_dump_json(),
        media_type="application/json",
        headers=headers,
    )


@app.patch("/v1/conversations/{conversation_id}", response_model=ConversationSummaryV1)
async def update_conversation(
    conversation_id: UUID,
    payload: ConversationUpdateRequest,
    request: Request,
) -> ConversationSummaryV1:
    try:
        return await request.app.state.container.conversations.update(
            conversation_id, payload, owner_key=_owner_key(request)
        )
    except ConversationNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Conversation not found") from exc


@app.delete("/v1/conversations/{conversation_id}", status_code=204)
async def delete_conversation(conversation_id: UUID, request: Request) -> Response:
    try:
        await request.app.state.container.conversations.delete(
            conversation_id, owner_key=_owner_key(request)
        )
    except ConversationNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Conversation not found") from exc
    return Response(status_code=204)


@app.post(
    "/v1/conversations/{conversation_id}/turns",
    response_model=ConversationTurnV1,
    status_code=201,
)
async def create_conversation_turn(
    conversation_id: UUID,
    payload: ConversationTurnCreateRequest,
    request: Request,
) -> ConversationTurnV1:
    try:
        return await request.app.state.container.conversations.submit_turn(
            conversation_id, payload, owner_key=_owner_key(request)
        )
    except ConversationNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Conversation not found") from exc
    except ConversationArchivedError as exc:
        raise HTTPException(status_code=409, detail="Conversation is archived") from exc
    except IdempotencyConflictError as exc:
        raise HTTPException(status_code=409, detail="Idempotency key conflict") from exc


@app.post("/v1/conversations/{conversation_id}/turns/stream")
async def stream_conversation_turn(
    conversation_id: UUID,
    payload: ConversationTurnCreateRequest,
    request: Request,
) -> StreamingResponse:
    """Persist the same turn that is streamed; a reconnect can reopen its saved trace."""

    container: AppContainer = request.app.state.container
    owner_key = _owner_key(request)
    queue: asyncio.Queue[TraceEventV1 | ConversationTurnV1 | Exception] = asyncio.Queue()

    async def produce() -> None:
        try:
            turn = await container.conversations.submit_turn(
                conversation_id,
                payload,
                owner_key=owner_key,
                on_trace_event=(
                    queue.put_nowait if container.settings.developer_trace_enabled else None
                ),
            )
            queue.put_nowait(turn)
        except Exception as exc:
            queue.put_nowait(exc)

    async def event_stream():
        task = asyncio.create_task(produce())
        while True:
            item = await queue.get()
            if isinstance(item, TraceEventV1):
                yield f"event: trace\ndata: {item.model_dump_json()}\n\n"
                continue
            if isinstance(item, ConversationTurnV1):
                yield f"event: final\ndata: {item.model_dump_json()}\n\n"
            else:
                body = json.dumps({"detail": "Research request failed safely"})
                yield f"event: error\ndata: {body}\n\n"
            break
        await task

    return StreamingResponse(event_stream(), media_type="text/event-stream")


def run() -> None:
    configure_logging()
    uvicorn.run("islamic_research_ai.api:app", host="127.0.0.1", port=8000)
