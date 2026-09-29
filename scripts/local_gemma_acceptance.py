from __future__ import annotations

import asyncio
import json
import os
from datetime import UTC, datetime
from typing import Any

import httpx

from islamic_research_ai.api import app
from islamic_research_ai.composition import build_container
from islamic_research_ai.contracts import ResearchResponseV1
from islamic_research_ai.settings import ROOT

SOURCE_SET = ROOT / "artifacts/e2e/quran-retrieval-sanity-development.json"
OUTPUT = ROOT / "artifacts/e2e/local-gemma-forced-development-set.json"
SINGLE_E2E_OUTPUT = ROOT / "artifacts/e2e/local-gemma-forced-quran-e2e.json"


def _locale(language_mix: str) -> str:
    return "ar" if language_mix == "Arabic" else "en"


def _completion_summary(values: list[Any]) -> list[dict[str, Any]]:
    return [
        {
            "inference_location": (
                "LOCAL" if value.effective_deployment.effective_provider_id == "local" else "CLOUD"
            ),
            "deployment_id": value.effective_deployment.deployment_id,
            "model_id": value.effective_deployment.model_id,
            "resolved_model_version_or_name": (
                value.effective_deployment.resolved_model_version_or_name
            ),
            "latency_ms": value.latency_ms,
            "input_tokens": value.input_tokens,
            "output_tokens": value.output_tokens,
            "paid_route": value.paid_route,
            "fallback_used": value.fallback_used,
        }
        for value in values
    ]


async def run() -> dict[str, Any]:
    os.environ["LOCAL_CHAT_FORCE"] = "true"
    source = json.loads(SOURCE_SET.read_text(encoding="utf-8"))
    container = await build_container()
    captured: dict[str, Any] = {}
    original_research = container.service.research

    async def traced_research(request: Any) -> Any:
        result = await original_research(request)
        captured["trace"] = result[1]
        return result

    container.service.research = traced_research  # type: ignore[method-assign]
    app.state.container = container
    evaluations: list[dict[str, Any]] = []
    try:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
            timeout=container.settings.local_chat_timeout_seconds + 30,
        ) as client:
            for item in source["evaluations"]:
                captured.clear()
                locale = _locale(item["language_mix"])
                started = datetime.now(UTC)
                response = await client.post(
                    "/v1/research",
                    json={"message": item["query"], "locale": locale},
                )
                elapsed_ms = round((datetime.now(UTC) - started).total_seconds() * 1000)
                record: dict[str, Any] = {
                    "case_id": item["id"],
                    "language_mix": item["language_mix"],
                    "locale": locale,
                    "http_status": response.status_code,
                    "elapsed_ms": elapsed_ms,
                }
                if response.status_code != 200 or "trace" not in captured:
                    record["contract_valid"] = False
                    record["safe_failure"] = "API request did not produce a canonical response"
                    evaluations.append(record)
                    continue
                canonical = ResearchResponseV1.model_validate(response.json())
                trace = captured["trace"]
                references = sorted(
                    {evidence.reference.value for evidence in trace.evidence.items_by_id.values()}
                )
                expected = item["manually_expected_or_reasonable_references"]
                overlap = sorted(set(references) & set(expected))
                record.update(
                    {
                        "contract_valid": True,
                        "status": canonical.status.value,
                        "chat_route": trace.chat_route,
                        "plan": (
                            {
                                "intent": trace.plan.intent.value,
                                "response_language": trace.plan.response_language,
                                "requested_operations": [
                                    operation.operation
                                    for operation in trace.plan.requested_operations
                                ],
                            }
                            if trace.plan is not None
                            else None
                        ),
                        "candidate_count": len(trace.candidates),
                        "hydrated_reference_ids": references,
                        "expected_reference_overlap": overlap,
                        "expected_reference_overlap_count": len(overlap),
                        "claim_count": sum(
                            block.block_type == "CLAIM" for block in canonical.blocks
                        ),
                        "source_block_count": sum(
                            block.block_type == "SOURCE" for block in canonical.blocks
                        ),
                        "citation_count": len(canonical.citations),
                        "planner_completions": _completion_summary(trace.planner_completions),
                        "synthesis_completions": _completion_summary(trace.synthesis_completions),
                    }
                )
                evaluations.append(record)
    finally:
        if hasattr(app.state, "container"):
            del app.state.container
        await container.close()

    answerable = sum(item.get("status") == "ANSWERABLE" for item in evaluations)
    canonical = sum(item.get("contract_valid") is True for item in evaluations)
    local_only = all(
        item.get("chat_route", {}).get("selected_route") == "LOCAL"
        for item in evaluations
        if item.get("contract_valid")
    )
    artifact = {
        "artifact_kind": "LOCAL_GEMMA_FORCED_QURAN_ACCEPTANCE_V2_2",
        "completed_at": datetime.now(UTC).isoformat(),
        "privacy": {
            "prompt_content_persisted": False,
            "evidence_text_persisted": False,
            "generated_claim_text_persisted": False,
            "local_paths_or_machine_ids_persisted": False,
        },
        "input_set": "quran-retrieval-sanity-development / NOT_FROZEN_FINAL_HOLDOUT",
        "case_count": len(evaluations),
        "canonical_response_count": canonical,
        "answerable_count": answerable,
        "local_only": local_only,
        "runtime_readiness": container.local_readiness.model_dump(mode="json"),
        "evaluations": evaluations,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8")
    first = evaluations[0]
    single_e2e = {
        "artifact_kind": "LOCAL_GEMMA_FORCED_QURAN_E2E_V2_2",
        "completed_at": artifact["completed_at"],
        "case_id": first["case_id"],
        "privacy": artifact["privacy"],
        "pipeline": {
            "fastapi_post_v1_research": first.get("http_status") == 200,
            "research_service": first.get("contract_valid") is True,
            "deterministic_preprocessing": first.get("plan") is not None,
            "local_planner": bool(first.get("planner_completions")),
            "canonical_research_plan": first.get("plan") is not None,
            "application_execution_plan": first.get("candidate_count") is not None,
            "non_citable_search_candidates": first.get("candidate_count", 0) > 0,
            "canonical_hydration": bool(first.get("hydrated_reference_ids")),
            "local_synthesis": bool(first.get("synthesis_completions")),
            "deterministic_renderer": first.get("source_block_count", 0) > 0,
            "canonical_response": first.get("contract_valid") is True,
        },
        "result": first,
        "runtime_readiness": artifact["runtime_readiness"],
    }
    SINGLE_E2E_OUTPUT.write_text(
        json.dumps(single_e2e, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return artifact


def main() -> None:
    result = asyncio.run(run())
    print(
        json.dumps(
            {
                "case_count": result["case_count"],
                "canonical_response_count": result["canonical_response_count"],
                "answerable_count": result["answerable_count"],
                "local_only": result["local_only"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
