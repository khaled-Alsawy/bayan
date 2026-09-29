from __future__ import annotations

import argparse
import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from islamic_research_ai.api import app
from islamic_research_ai.composition import build_container
from islamic_research_ai.contracts import ResearchResponseV1
from islamic_research_ai.logging import configure_logging


async def run_e2e(query: str, artifact_path: Path | None = None) -> dict[str, object]:
    container = await build_container()
    try:
        captured: dict[str, Any] = {}
        original_research = container.service.research

        async def traced_research(request: Any) -> Any:
            response_and_trace = await original_research(request)
            captured["trace"] = response_and_trace[1]
            return response_and_trace

        container.service.research = traced_research  # type: ignore[method-assign]
        app.state.container = container
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            api_response = await client.post(
                "/v1/research", json={"message": query, "locale": "en"}
            )
        api_response.raise_for_status()
        response = ResearchResponseV1.model_validate(api_response.json())
        trace = captured["trace"]
        artifact: dict[str, object] = {
            "completed_at": datetime.now(UTC).isoformat(),
            "query": query,
            "api": {"method": "POST", "path": "/v1/research", "http_status": 200},
            "status": response.status,
            "internal_terminal_reason": trace.terminal_reason,
            "provider_failure": (
                trace.provider_failure.model_dump(mode="json")
                if trace.provider_failure is not None
                else None
            ),
            "chat_route": trace.chat_route,
            "chat_completion_records_v22": trace.chat_completion_records_v22,
            "local_readiness": container.local_readiness.model_dump(mode="json"),
            "conversation_id": str(response.conversation_id),
            "turn_id": str(response.turn_id),
            "correlation_id": str(response.correlation_id),
            "working_context": trace.context.model_dump(mode="json"),
            "deterministic_resolution": trace.resolution.model_dump(mode="json"),
            "scope_decision": trace.scope.model_dump(mode="json"),
            "plan": trace.plan.model_dump(mode="json") if trace.plan else None,
            "execution_plan": (
                trace.execution_plan.model_dump(mode="json") if trace.execution_plan else None
            ),
            "retrieval": {
                "candidate_count": len(trace.candidates),
                "candidates": [item.model_dump(mode="json") for item in trace.candidates],
                "tool_results": {
                    node_id: result.model_dump(mode="json")
                    for node_id, result in trace.results.items()
                },
                "evidence_set": trace.evidence.model_dump(mode="json"),
            },
            "provider_completions": {
                "planner": [item.model_dump(mode="json") for item in trace.planner_completions],
                "query_embedding": (
                    trace.query_embedding_completion.model_dump(mode="json")
                    if trace.query_embedding_completion
                    else None
                ),
                "synthesis": [item.model_dump(mode="json") for item in trace.synthesis_completions],
            },
            "response": response.model_dump(mode="json"),
            "gate_3": "OPEN_AWAITING_RESPONSES",
            "development_override": "PROJECT_OWNER_ACCEPTED_RISK",
        }
        path = artifact_path or (
            container.settings.artifact_dir / "quran-topic-search-v2.1-conformant.json"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        serialized = json.dumps(artifact, ensure_ascii=False, indent=2)
        path.write_text(serialized, encoding="utf-8")
        canonical_path = container.settings.artifact_dir / "quran-topic-search.json"
        if response.status.value == "ANSWERABLE" and canonical_path != path:
            canonical_path.write_text(serialized, encoding="utf-8")
        return artifact
    finally:
        if hasattr(app.state, "container"):
            del app.state.container
        await container.close()


def main() -> None:
    configure_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "query",
        nargs="?",
        default="What does the Quran say about patience during hardship?",
    )
    parser.add_argument("--artifact", type=Path)
    args = parser.parse_args()
    result = asyncio.run(run_e2e(args.query, args.artifact))
    print(json.dumps(result["response"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
