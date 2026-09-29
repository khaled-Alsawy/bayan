from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import httpx

from islamic_research_ai.contracts import TraceEventV1
from islamic_research_ai.providers.groq import _availability_error, _rate_limit_reset_ms


def runner_module() -> Any:
    path = Path(__file__).resolve().parents[1] / "qa/live_quality_runner.py"
    spec = importlib.util.spec_from_file_location("live_quality_runner", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def telemetry(*, retry_after_ms: int | None, token_reset_ms: int | None = None) -> dict[str, Any]:
    return {
        "any_local_fallback": False,
        "nodes": [],
        "provider_failures": [
            {
                "provider_id": "groq",
                "category": "RATE_LIMITED",
                "retry_after_ms": retry_after_ms,
                "rate_limit_reset_tokens_ms": token_reset_ms,
            }
        ],
        "timeout_fired": False,
        "terminal_reason": None,
    }


def test_daily_retry_after_exceeding_cap_is_not_slept() -> None:
    runner = runner_module()
    decision = runner.retry_decision(
        telemetry(retry_after_ms=5_616_000),
        attempt=0,
        groq_429_backoff_seconds=6,
        max_backoff_seconds=120,
    )
    assert decision == {
        "action": "GIVE_UP",
        "reason": "GROQ_RETRY_AFTER_EXCEEDS_CAP",
        "wait_seconds": 0.0,
        "retry_after_ms_advertised": 5_616_000,
        "token_reset_ms_advertised": None,
    }


def test_token_reset_is_preferred_over_a_long_daily_retry_after() -> None:
    runner = runner_module()
    decision = runner.retry_decision(
        telemetry(retry_after_ms=5_616_000, token_reset_ms=800),
        attempt=0,
        groq_429_backoff_seconds=6,
        max_backoff_seconds=120,
    )
    assert decision["action"] == "RETRY"
    assert decision["reason"] == "GROQ_TOKEN_BUCKET_RESET"
    assert decision["wait_seconds"] == 0.8


def test_rate_limit_without_headers_uses_bounded_exponential_backoff() -> None:
    runner = runner_module()
    decision = runner.retry_decision(
        telemetry(retry_after_ms=None),
        attempt=1,
        groq_429_backoff_seconds=6,
        max_backoff_seconds=120,
    )
    assert decision["action"] == "RETRY"
    assert decision["reason"] == "GROQ_EXPONENTIAL_BACKOFF"
    assert decision["wait_seconds"] == 12


def test_groq_token_reset_header_is_parsed_without_retaining_raw_headers() -> None:
    response = httpx.Response(429, headers={"x-ratelimit-reset-tokens": "1m2.5s"})
    assert _rate_limit_reset_ms(response, "x-ratelimit-reset-tokens") == 62_500
    milliseconds = httpx.Response(429, headers={"x-ratelimit-reset-tokens": "800ms"})
    assert _rate_limit_reset_ms(milliseconds, "x-ratelimit-reset-tokens") == 800


def test_groq_rate_limit_diagnostic_carries_the_token_reset() -> None:
    response = httpx.Response(429, headers={"x-ratelimit-reset-tokens": "0.8s"})
    error = _availability_error(
        response=response,
        category="RATE_LIMITED",
        safe_detail="Groq temporarily rate limited the request.",
    )
    assert error.diagnostic.rate_limit_reset_tokens_ms == 800


def test_recovered_429_http_200_is_scored_degraded_and_not_retried() -> None:
    runner = runner_module()
    recovered_telemetry = telemetry(retry_after_ms=800)
    recovered_telemetry["any_local_fallback"] = True
    response = httpx.Response(200, json={"status": "ANSWERABLE"})
    decision = runner.retry_decision(
        recovered_telemetry,
        attempt=0,
        groq_429_backoff_seconds=6,
        max_backoff_seconds=120,
        http_status=200,
    )
    assert runner.response_is_scoreable(response, response.json()) is True
    assert (
        runner.should_retry(
            response_scoreable=True,
            decision=decision,
            attempt=0,
            groq_429_retries=3,
        )
        is False
    )
    result = {
        **runner.CASES[0],
        "http_status": 200,
        "elapsed_ms": 12,
        "attempts": [{"attempt": 1}],
        "telemetry": recovered_telemetry,
        "response": {"status": "ANSWERABLE"},
        "degraded_to_local": True,
    }
    metrics = runner.metrics_for([result])
    assert result["evaluation_bucket"] == "ANSWERABLE"
    assert "infrastructure_reason" not in result
    assert metrics["degraded_to_local_count"] == 1


def test_http_503_is_infrastructure_and_is_retried() -> None:
    runner = runner_module()
    decision = runner.retry_decision(
        runner.empty_telemetry(),
        attempt=0,
        groq_429_backoff_seconds=6,
        max_backoff_seconds=120,
        http_status=503,
    )
    assert decision is not None
    assert decision["reason"] == "HTTP_503_EXPONENTIAL_BACKOFF"
    assert (
        runner.should_retry(
            response_scoreable=False,
            decision=decision,
            attempt=0,
            groq_429_retries=3,
        )
        is True
    )
    result = {
        **runner.CASES[0],
        "http_status": 503,
        "elapsed_ms": 12,
        "attempts": [],
        "telemetry": runner.empty_telemetry(),
        "response": None,
    }
    assert runner.metrics_for([result])["infrastructure_failure_count"] == 1
    assert result["evaluation_bucket"] == "INFRASTRUCTURE_FAILURE"


def test_daily_budget_exhaustion_writes_a_partial_artifact_with_reason(
    tmp_path: Path, monkeypatch
) -> None:
    runner = runner_module()
    output = tmp_path / "live-api-quality-results.json"
    monkeypatch.setattr(runner, "OUTPUT", output)
    exhausted = telemetry(retry_after_ms=5_616_000)
    exhausted["provider_failures"][0].update(
        quota_limit_tokens=200_000,
        quota_used_tokens=199_430,
        quota_requested_tokens=793,
    )
    assert runner.daily_budget_effectively_exhausted(exhausted) is True
    runner.write_artifact(
        [],
        complete=False,
        completion_reason="DAILY_BUDGET_EFFECTIVELY_EXHAUSTED",
        inter_case_delay_seconds=60,
        groq_429_retries=3,
        groq_429_backoff_seconds=6,
        max_backoff_seconds=120,
        minimum_remaining_tokens=4000,
        token_budget_wait_seconds=60,
    )
    artifact = json.loads(output.read_text(encoding="utf-8"))
    assert artifact["complete"] is False
    assert artifact["completion_reason"] == "DAILY_BUDGET_EFFECTIVELY_EXHAUSTED"


def test_force_local_chat_is_explicitly_marked_in_the_artifact(
    tmp_path: Path, monkeypatch
) -> None:
    runner = runner_module()
    output = tmp_path / "live-api-quality-results.json"
    monkeypatch.setattr(runner, "OUTPUT", output)
    runner.write_artifact(
        [],
        complete=True,
        inter_case_delay_seconds=60,
        groq_429_retries=3,
        groq_429_backoff_seconds=6,
        max_backoff_seconds=120,
        minimum_remaining_tokens=4000,
        token_budget_wait_seconds=60,
        force_local_chat=True,
    )
    artifact = json.loads(output.read_text(encoding="utf-8"))
    assert artifact["measurement"] == {
        "force_local_chat": True,
        "chat_model_measurement": "LOCAL_MODEL_ONLY",
    }


def test_artifact_trace_events_are_persisted_and_redacted() -> None:
    runner = runner_module()

    class Trace:
        events = [
            TraceEventV1(
                seq=1,
                elapsed_ms=17,
                node="retrieval",
                phase="completed",
                summary="Retrieved candidates",
                detail={"candidate_count": 3, "source_text": "must not persist"},
                provider=None,
            )
        ]

    assert runner.artifact_safe_trace_events(Trace()) == [
        {
            "seq": 1,
            "elapsed_ms": 17,
            "node": "retrieval",
            "phase": "completed",
            "summary": "Retrieved candidates",
            "detail": {"candidate_count": 3, "source_text": "[REDACTED]"},
            "provider": None,
        }
    ]


def test_artifact_trace_events_preserve_safe_nested_numeric_telemetry() -> None:
    runner = runner_module()

    class Trace:
        events = [
            TraceEventV1(
                seq=1,
                elapsed_ms=17,
                node="retrieval",
                phase="completed",
                summary="Ranked retrieval candidates observed",
                detail={
                    "ranking_signal_ranges": {
                        "rrf_score": {"count": 12, "min": 0.03, "max": 0.11}
                    }
                },
                provider=None,
            )
        ]

    assert runner.artifact_safe_trace_events(Trace())[0]["detail"] == {
        "ranking_signal_ranges": {
            "rrf_score": {"count": 12, "min": 0.03, "max": 0.11}
        }
    }


def test_partial_artifact_is_incremental_and_uses_partial_denominator(
    tmp_path: Path, monkeypatch
) -> None:
    runner = runner_module()
    output = tmp_path / "live-api-quality-results.json"
    monkeypatch.setattr(runner, "OUTPUT", output)
    result = {
        **runner.CASES[0],
        "http_status": 200,
        "elapsed_ms": 12,
        "attempts": [],
        "telemetry": {
            "any_local_fallback": False,
            "nodes": [],
            "provider_failures": [],
            "timeout_fired": False,
            "terminal_reason": None,
        },
        "response": {"status": "ANSWERABLE"},
    }
    runner.write_artifact(
        [result],
        complete=False,
        inter_case_delay_seconds=60,
        groq_429_retries=3,
        groq_429_backoff_seconds=6,
        max_backoff_seconds=120,
        minimum_remaining_tokens=4000,
        token_budget_wait_seconds=60,
    )
    artifact = json.loads(output.read_text(encoding="utf-8"))
    assert artifact["complete"] is False
    assert artifact["attempted_case_count"] == 1
    assert artifact["metrics"]["raw_answerable_denominator"] == 1
    assert "raw_40_case_answerable_share" not in artifact["metrics"]
