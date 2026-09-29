from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from islamic_research_ai.service import TraceRecorder
from islamic_research_ai.settings import Settings


def test_trace_events_are_ordered_bounded_and_redacted() -> None:
    recorder = TraceRecorder(True)
    sentinel = "TRACE_SECRET_DO_NOT_LEAK"
    canonical_arabic = "إِنَّ مَعَ الْعُسْرِ يُسْرًا"
    recorder.emit(
        "retrieval",
        "completed",
        "Retrieved candidate identifiers",
        detail={
            "groq_api_key": sentinel,
            "Authorization": f"Bearer {sentinel}",
            "exact_text": canonical_arabic,
            "candidate_ids": ["C1", "C2"],
            "long_value": "x" * 300,
        },
    )
    recorder.emit("tafsir", "completed", "Hydrated approved Tafsir", detail={"hit_count": 1})
    serialized = json.dumps([event.model_dump(mode="json") for event in recorder.events])
    assert [event.seq for event in recorder.events] == [1, 2]
    assert [event.elapsed_ms for event in recorder.events] == sorted(
        event.elapsed_ms for event in recorder.events
    )
    assert sentinel not in serialized
    assert canonical_arabic not in serialized
    assert "[REDACTED]" in serialized
    assert "[TRUNCATED]" in serialized


def test_production_hard_disables_developer_trace() -> None:
    settings = Settings(app_environment="production", developer_trace_enabled=True)
    assert settings.developer_trace_enabled is False


def test_k1_anchor_retention_is_enabled_by_default() -> None:
    assert Settings().relevance_empty_match_retains_anchors is True


def test_unknown_topic_tafsir_alias_fails_at_settings_construction() -> None:
    with pytest.raises(ValidationError, match="known QUL Tafsir resource"):
        Settings(topic_tafsir_alias="qul-tafsir-typo-ar-v1")
