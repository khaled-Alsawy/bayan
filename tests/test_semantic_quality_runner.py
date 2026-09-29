from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any


def runner_module() -> Any:
    path = Path(__file__).resolve().parents[1] / "qa/semantic_quality_runner.py"
    spec = importlib.util.spec_from_file_location("semantic_quality_runner", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_manifest_has_required_semantic_coverage() -> None:
    runner = runner_module()
    summary = runner.validate_cases(runner.load_cases())
    assert summary == {
        "case_count": 60,
        "none_count": 15,
        "low_count": 20,
        "high_count": 25,
        "answer_count": 50,
        "clarify_count": 2,
        "refuse_count": 8,
    }


def test_layer_one_scores_citations_anchors_and_status_without_a_model() -> None:
    runner = runner_module()
    cases = [
        {
            "id": "sem_example",
            "ui_language": "en",
            "prompt": "a hard situation",
            "indirection": "high",
            "expected_themes": ["patience"],
            "acceptable_anchors": ["2:153"],
            "unacceptable_anchors": ["4:11"],
            "forbidden_terms": ["patience"],
            "intent": "answer",
            "notes": "fixture",
        }
    ]
    record = {
        "id": "sem_example",
        "response": {
            "status": "ANSWERABLE",
            "citations": [{"evidence_id": "E1", "reference": {"value": "2:153"}}],
        },
        "retrieved_evidence": [{"evidence_id": "E1", "reference": {"value": "2:153"}}],
    }

    outcome = runner.score_case(cases[0], record)

    assert outcome["layer_1_pass"] is True
    assert outcome["anchor_hit"] is True
    assert outcome["distractor_hit"] is False
    assert outcome["layer_2"] == {"scored_by": None, "human_confirmed": False, "status": "UNSET"}


def test_layer_one_rejects_invented_citations_and_distractors() -> None:
    runner = runner_module()
    case = {
        "id": "sem_example",
        "indirection": "high",
        "intent": "answer",
        "acceptable_anchors": ["2:153"],
        "unacceptable_anchors": ["4:11"],
    }
    record = {
        "response": {
            "status": "ANSWERABLE",
            "citations": [{"evidence_id": "E9", "reference": {"value": "4:11"}}],
        },
        "retrieved_evidence": [{"evidence_id": "E1", "reference": {"value": "2:153"}}],
    }

    outcome = runner.score_case(case, record)

    assert outcome["citation_in_retrieved_evidence"] is False
    assert outcome["anchor_hit"] is False
    assert outcome["distractor_hit"] is True
    assert outcome["layer_1_pass"] is False
