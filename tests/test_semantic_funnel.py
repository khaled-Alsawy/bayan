from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any


def funnel_module() -> Any:
    path = Path(__file__).resolve().parents[1] / "qa/semantic_funnel.py"
    spec = importlib.util.spec_from_file_location("semantic_funnel", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_funnel_tracks_anchor_survival_through_selection_and_citations() -> None:
    funnel = funnel_module()
    cases = [
        {
            "id": "sem_survives",
            "indirection": "none",
            "intent": "answer",
            "acceptable_anchors": ["2:153"],
            "unacceptable_anchors": ["4:11"],
        },
        {
            "id": "sem_dropped",
            "indirection": "high",
            "intent": "answer",
            "acceptable_anchors": ["3:7"],
            "unacceptable_anchors": [],
        },
    ]
    records = [
        {
            "id": "sem_survives",
            "events": [
                {
                    "summary": "Candidates dropped before evidence assembly",
                    "detail": {
                        "selected": [{"reference": "2:153"}],
                        "dropped": [{"reference": "2:154"}],
                    },
                },
                {
                    "summary": "Topic relevance assessment completed",
                    "detail": {"retained_references": ["2:153", "2:154"]},
                },
            ],
            "response": {
                "output_kind": "SUBSTANTIVE_SYNTHESIS",
                "citations": [
                    {"reference": {"value": "2:153"}},
                    {"reference": {"value": "4:11"}},
                ]
            },
        },
        {
            "id": "sem_dropped",
            "events": [
                {
                    "summary": "Candidates dropped before evidence assembly",
                    "detail": {
                        "selected": [{"reference": "3:10"}],
                        "dropped": [{"reference": "3:7"}],
                    },
                }
            ],
            "response": {
                "output_kind": "TEMPLATE",
                "citations": [{"reference": {"value": "3:10"}}],
            },
        },
    ]

    result = funnel.compute_funnel(cases, records)

    overall = result["overall"]
    assert overall["answer_cases"] == 2
    assert overall["anchor_presence"] == {
        "late_binding_candidate_pool": {
            "hits": 2,
            "rate": 1.0,
            "available_case_count": 2,
            "unavailable_case_count": 0,
        },
        "selected_anchors": {
            "hits": 1,
            "rate": 0.5,
            "available_case_count": 2,
            "unavailable_case_count": 0,
        },
        "hydrated_structural_window": {
            "hits": 1,
            "rate": 0.5,
            "available_case_count": 2,
            "unavailable_case_count": 0,
        },
        "retained_relevance_set": {
            "hits": 1,
            "rate": 1.0,
            "available_case_count": 1,
            "unavailable_case_count": 1,
        },
        "final_citations": {
            "hits": 1,
            "rate": 0.5,
            "available_case_count": 2,
            "unavailable_case_count": 0,
        },
    }
    assert overall["distractor_rate"] == {"hits": 1, "rate": 0.5}
    assert overall["mean_distinct_quran_verses_cited"] == 1.5
    assert result["output_kind"] == {
        "counts": {
            "SUBSTANTIVE_SYNTHESIS": 1,
            "SOURCE_ONLY": 0,
            "TEMPLATE": 1,
            "NO_CONTENT": 0,
        },
        "by_citation_hit": {
            "hit": {
                "SUBSTANTIVE_SYNTHESIS": 1,
                "SOURCE_ONLY": 0,
                "TEMPLATE": 0,
                "NO_CONTENT": 0,
            },
            "miss": {
                "SUBSTANTIVE_SYNTHESIS": 0,
                "SOURCE_ONLY": 0,
                "TEMPLATE": 1,
                "NO_CONTENT": 0,
            },
        },
        "available_case_count": 2,
        "unavailable_case_count": 0,
    }
    assert "output_kind by citation hit" in funnel.format_funnel(result)
    assert result["by_indirection"]["none"]["anchor_presence"]["final_citations"] == {
        "hits": 1,
        "rate": 1.0,
        "available_case_count": 1,
        "unavailable_case_count": 0,
    }
    assert result["by_indirection"]["high"]["anchor_presence"]["selected_anchors"] == {
        "hits": 0,
        "rate": 0.0,
        "available_case_count": 1,
        "unavailable_case_count": 0,
    }
    assert result["by_indirection"]["high"]["anchor_presence"]["retained_relevance_set"] == {
        "hits": None,
        "rate": None,
        "available_case_count": 0,
        "unavailable_case_count": 1,
    }


def test_funnel_marks_retained_relevance_stage_unavailable_for_older_artifacts() -> None:
    funnel = funnel_module()
    cases = [
        {
            "id": "sem_older_trace",
            "indirection": "high",
            "intent": "answer",
            "acceptable_anchors": ["2:153"],
            "unacceptable_anchors": [],
        }
    ]
    records = [
        {
            "id": "sem_older_trace",
            "events": [
                {
                    "summary": "Candidates dropped before evidence assembly",
                    "detail": {"selected": [{"reference": "2:153"}], "dropped": []},
                }
            ],
            "response": {"citations": [{"reference": {"value": "2:153"}}]},
        }
    ]

    result = funnel.compute_funnel(cases, records)

    assert result["overall"]["anchor_presence"]["retained_relevance_set"] == {
        "hits": None,
        "rate": None,
        "available_case_count": 0,
        "unavailable_case_count": 1,
    }
    assert "retained_relevance_set: unavailable" in funnel.format_funnel(result)
    assert result["output_kind"] == {
        "counts": None,
        "by_citation_hit": None,
        "available_case_count": 0,
        "unavailable_case_count": 1,
    }
    assert "output_kind: unavailable" in funnel.format_funnel(result)
