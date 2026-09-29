from __future__ import annotations

import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

from islamic_research_ai.contracts import RenderedOutputKind, derive_rendered_output_kind
from qa.build_audit_sample import (
    DEFAULT_ARTIFACT,
    _claims,
    _is_template_only_response,
    _pair_order,
    _record_id,
    _response,
    build_audit,
    compare_verdicts,
    contextual_topic_templates,
    read_verdicts,
    select_sample,
)


def test_fixed_contextual_topic_templates_are_not_substantive_claims() -> None:
    templates = contextual_topic_templates()
    arabic_template = (
        "الجواب المباشر: المقطع القرآني المسترجع يُعرض بسياقه المتصل، لا كآية منفصلة. "
        "ولا يضيف هذا الجواب أذكارًا أو عددًا للتلاوة؛ فمصدر حديث معتمدًا لم يُضف إلى "
        "هذا النطاق بعد."
    )
    assert "الجواب المباشر: المقطع القرآني المسترجع يُعرض بسياقه المتصل" in templates
    template_response = {"blocks": [{"block_type": "CLAIM", "text": arabic_template}]}
    assert _is_template_only_response(template_response)
    assert _claims(template_response) == []

    mixed_response = {
        "blocks": [
            {"block_type": "CLAIM", "text": arabic_template},
            {"block_type": "CLAIM", "text": "A genuinely substantive rendered claim."},
        ]
    }
    assert not _is_template_only_response(mixed_response)
    assert _claims(mixed_response) == ["A genuinely substantive rendered claim."]


def test_v19_rendered_output_kind_distribution_is_pinned_to_answer_cases() -> None:
    """The fixed display template must never be counted as substantive output."""

    import qa.build_audit_sample as audit

    cases = audit.load_cases()
    classifications = Counter(
        derive_rendered_output_kind(_response(record).get("blocks"))
        for record in audit.read_records(DEFAULT_ARTIFACT)
        if (case_id := _record_id(record)) in cases and cases[case_id].get("intent") == "answer"
    )

    assert classifications == {
        RenderedOutputKind.TEMPLATE: 22,
        RenderedOutputKind.SUBSTANTIVE_SYNTHESIS: 20,
        RenderedOutputKind.NO_CONTENT: 8,
    }


def _case(case_id: str) -> dict[str, object]:
    return {
        "id": case_id,
        "intent": "answer",
        "prompt": "A reader question",
        "indirection": "high",
        "acceptable_anchors": ["2:2"],
        "unacceptable_anchors": [],
    }


def _record(case_id: str, claim: str) -> dict[str, object]:
    citation = {
        "evidence_id": "E1",
        "source_kind": "QURAN_TRANSLATION",
        "reference": {"value": "1:1"},
    }
    return {
        "id": case_id,
        "events": [
            {
                "summary": "Synthesis grounding validation failed",
                "node": "synthesis",
                "phase": "failed",
            }
        ],
        "retrieved_evidence": [citation],
        "response": {
            "status": "PARTIAL",
            "citations": [citation],
            "blocks": [
                {"block_type": "CLAIM", "text": claim},
                {
                    "block_type": "SOURCE",
                    "exact_text": "Exact canonical display text — لا يُغيَّر",
                    "citation": citation,
                },
            ],
        },
    }


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.write_text("\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n")


def test_selection_is_deterministic_and_pair_order_is_counterbalanced() -> None:
    cases = {f"case-{index}": _case(f"case-{index}") for index in range(13)}
    records = [_record(case_id, f"Claim {case_id}") for case_id in cases]
    first = select_sample(records, cases)
    second = select_sample(records, cases)
    assert first[0] == second[0]
    assert first[1] == second[1]
    assert len(first[0]) == 12
    orders = {_pair_order(case_id) for case_id in first[0]}
    assert orders == {("baseline", "challenger"), ("challenger", "baseline")}


def test_review_is_blinded_and_verdict_readback_is_a_spot_audit(tmp_path: Path) -> None:
    cases = {f"case-{index}": _case(f"case-{index}") for index in range(13)}
    cases_path = tmp_path / "cases.yaml"
    import yaml

    cases_path.write_text(yaml.safe_dump(list(cases.values()), allow_unicode=True))
    baseline = tmp_path / "baseline.jsonl"
    challenger = tmp_path / "challenger.jsonl"
    _write_jsonl(baseline, [_record(case_id, "First rendered claim") for case_id in cases])
    _write_jsonl(challenger, [_record(case_id, "Second rendered claim") for case_id in cases])

    review_path, key_path, key = build_audit(
        baseline,
        output_dir=tmp_path,
        name="blinded",
        cases_path=cases_path,
        challenger=challenger,
        for_model=True,
    )
    review = review_path.read_text(encoding="utf-8")
    model_prompt = Path(key["model_prompt_path"]).read_text(encoding="utf-8")
    assert "Exact canonical display text — لا يُغيَّر" in review
    assert "Translation — 1:1" in review
    assert "case-0" not in review
    assert "fallback_rendered" not in review
    assert "baseline" not in review.casefold()
    assert "acceptable_anchors" not in review
    assert "anchor_hit" not in review
    assert "evidence_span" not in review
    assert "model verdict" not in review.casefold()
    assert key["items"][0]["case_id"].startswith("case-")
    assert model_prompt.count("## Case ") == 12
    assert "case-0" not in model_prompt
    assert "fallback_rendered" not in model_prompt
    assert "baseline" not in model_prompt.casefold()
    assert "Provenance is not " in model_prompt

    completed = review.replace(
        "Verdict [supported / unsupported / unclear]: \n",
        "Verdict [supported / unsupported / unclear]: supported\n",
        1,
    )
    review_path.write_text(completed, encoding="utf-8")
    verdicts = read_verdicts(review_path, key_path)
    assert verdicts["claim_verdict_counts"]["overall"] == {"supported": 1}
    assert "spot audit, not an accuracy percentage" in verdicts["caveat"]

    completed = review.replace(
        "Verdict [supported / unsupported / unclear]: \n",
        "Verdict [supported / unsupported / unclear]: supported\n",
    )
    review_path.write_text(completed, encoding="utf-8")
    model_responses = []
    for item in key["items"]:
        for response in item["responses"]:
            model_responses.append(
                {
                    "case_number": item["case_number"],
                    "response_label": response["label"],
                    "claim_verdicts": [
                        {
                            "claim_number": claim_number,
                            "verdict": "unsupported" if claim_number == 1 else "supported",
                            "evidence_span": "Exact canonical display text",
                            "reasoning": "A fixture-only independent judgement.",
                        }
                        for claim_number in range(1, response["claim_count"] + 1)
                    ],
                    "addresses_question": "yes",
                }
            )
    model_json = tmp_path / "model.json"
    model_json.write_text(json.dumps({"responses": model_responses}), encoding="utf-8")
    comparison = compare_verdicts(review_path, model_json, key_path)
    assert comparison["overall_agreement"]["compared_claims"] == 24
    assert comparison["disagreement_direction"] == {"model_less_generous": 24}
    assert "Human verdicts are the audit result" in comparison["judgement_role"]
    assert "spot audit, not an accuracy percentage" in comparison["caveat"]

    cli = subprocess.run(
        [
            sys.executable,
            "qa/build_audit_sample.py",
            "--compare-verdicts",
            str(review_path),
            str(model_json),
            "--key",
            str(key_path),
        ],
        check=True,
        capture_output=True,
        text=True,
        cwd=Path(__file__).parents[1],
    )
    assert json.loads(cli.stdout)["overall_agreement"]["compared_claims"] == 24
