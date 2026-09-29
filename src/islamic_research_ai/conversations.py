from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import Callable
from uuid import UUID, uuid4

from islamic_research_ai.contracts import (
    ConversationCreateRequest,
    ConversationDetailV1,
    ConversationPageV1,
    ConversationSummaryV1,
    ConversationTurnCreateRequest,
    ConversationTurnV1,
    ConversationUpdateRequest,
    ResearchRequest,
    TraceEventV1,
)
from islamic_research_ai.repositories.postgres import PostgresRepository
from islamic_research_ai.service import ResearchService


class ConversationNotFoundError(LookupError):
    pass


class ConversationArchivedError(ValueError):
    pass


class IdempotencyConflictError(ValueError):
    pass


class ConversationService:
    """Application-owned history around the unchanged canonical research endpoint."""

    def __init__(self, repository: PostgresRepository, research: ResearchService) -> None:
        self.repository = repository
        self.research = research
        self._locks: dict[tuple[UUID, str], asyncio.Lock] = {}

    async def create(
        self, payload: ConversationCreateRequest, *, owner_key: str
    ) -> ConversationSummaryV1:
        title = payload.title or ("محادثة جديدة" if payload.locale == "ar" else "New conversation")
        return await self.repository.create_conversation(
            conversation_id=uuid4(),
            owner_key=owner_key,
            title=title,
            locale=payload.locale,
        )

    async def list(
        self,
        *,
        owner_key: str,
        include_archived: bool,
        limit: int,
        cursor: str | None,
    ) -> ConversationPageV1:
        items, next_cursor = await self.repository.list_conversations(
            owner_key=owner_key,
            include_archived=include_archived,
            limit=limit,
            cursor=cursor,
        )
        return ConversationPageV1(items=items, next_cursor=next_cursor)

    async def get(
        self,
        conversation_id: UUID,
        *,
        owner_key: str,
        turn_limit: int = 50,
        cursor: str | None = None,
    ) -> tuple[ConversationDetailV1, str | None]:
        conversation, next_cursor = await self.repository.get_conversation(
            conversation_id=conversation_id,
            owner_key=owner_key,
            turn_limit=turn_limit,
            turn_cursor=cursor,
        )
        if conversation is None:
            raise ConversationNotFoundError
        return conversation, next_cursor

    async def update(
        self,
        conversation_id: UUID,
        payload: ConversationUpdateRequest,
        *,
        owner_key: str,
    ) -> ConversationSummaryV1:
        updated = await self.repository.update_conversation(
            conversation_id=conversation_id,
            owner_key=owner_key,
            title=payload.title,
            archived=payload.archived,
        )
        if updated is None:
            raise ConversationNotFoundError
        return updated

    async def delete(self, conversation_id: UUID, *, owner_key: str) -> None:
        if not await self.repository.delete_conversation(
            conversation_id=conversation_id, owner_key=owner_key
        ):
            raise ConversationNotFoundError

    async def submit_turn(
        self,
        conversation_id: UUID,
        payload: ConversationTurnCreateRequest,
        *,
        owner_key: str,
        on_trace_event: Callable[[TraceEventV1], None] | None = None,
    ) -> ConversationTurnV1:
        request_hash = self._request_hash(payload)
        lock_key = (conversation_id, payload.idempotency_key)
        lock = self._locks.setdefault(lock_key, asyncio.Lock())
        try:
            async with lock:
                existing = await self.repository.get_turn_by_idempotency_key(
                    conversation_id=conversation_id,
                    owner_key=owner_key,
                    idempotency_key=payload.idempotency_key,
                )
                if existing is not None:
                    existing_hash, turn = existing
                    if existing_hash != request_hash:
                        raise IdempotencyConflictError
                    return turn

                conversation, _ = await self.get(conversation_id, owner_key=owner_key)
                if conversation.archived:
                    raise ConversationArchivedError
                response, trace = await self.research.research(
                    ResearchRequest(
                        message=payload.message,
                        locale=payload.locale,
                        conversation_id=conversation_id,
                    ),
                    recent_user_turns=conversation.turns[-5:],
                    on_trace_event=on_trace_event,
                )
                turn = await self.repository.persist_conversation_turn(
                    conversation_id=conversation_id,
                    owner_key=owner_key,
                    idempotency_key=payload.idempotency_key,
                    request_hash=request_hash,
                    user_message=payload.message,
                    locale=payload.locale,
                    response=response,
                    trace_events=getattr(trace, "events", []),
                )
                if conversation.turn_count == 0:
                    await self.repository.update_conversation(
                        conversation_id=conversation_id,
                        owner_key=owner_key,
                        title=self._title_from_question(payload.message),
                    )
                # Return the transient response on the first request. The stored turn returned by
                # history remains policy-redacted for Quran Foundation source blocks.
                return turn.model_copy(update={"response": response})
        finally:
            if not lock.locked():
                self._locks.pop(lock_key, None)

    @staticmethod
    def _request_hash(payload: ConversationTurnCreateRequest) -> str:
        canonical = json.dumps(
            {"message": payload.message.strip(), "locale": payload.locale},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(canonical.encode()).hexdigest()

    @staticmethod
    def _title_from_question(question: str) -> str:
        compact = " ".join(question.split())
        if len(compact) <= 60:
            return compact
        prefix = compact[:60].rsplit(" ", 1)[0].rstrip("،,:;؛.!?")
        return f"{prefix or compact[:57]}…"
