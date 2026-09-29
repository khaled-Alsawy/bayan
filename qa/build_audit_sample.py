"""Build a deterministic, blinded human support-audit sample from collected responses."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import yaml

from islamic_research_ai.contextual_topic_template import (
    deterministic_contextual_topic_template_markers,
    is_deterministic_contextual_topic_template_prose,
)

try:
    from semantic_quality_runner import score_case
except ModuleNotFoundError:
    from qa.semantic_quality_runner import score_case

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACT = ROOT / "artifacts/qa/semantic/semantic_responses_v19.jsonl"
DEFAULT_CASES = ROOT / "qa/cases/semantic_cases.yaml"
DEFAULT_OUTPUT_DIR = ROOT / "artifacts/qa/audits"
SAMPLE_SIZE = 12
TEMPLATE_ONLY_LIMIT = 4
SEED = "support-audit-sample-v1"
TAFSIR_REVIEW_CHARACTER_LIMIT = 4_000
FALLBACK_SUMMARIES = {
    "Synthesis grounding validation failed",
    "Composition provider failed; retained grounded claims",
}
VERDICT_VALUES = ("supported", "unsupported", "unclear")
SPOT_AUDIT_CAVEAT = "This is a spot audit, not an accuracy percentage for the product."


def contextual_topic_templates() -> frozenset[str]:
    """Expose the production-derived fallback markers for audit callers."""

    return deterministic_contextual_topic_template_markers()


def read_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(record, dict):
            records.append(record)
    return records


def load_cases(path: Path = DEFAULT_CASES) -> dict[str, dict[str, Any]]:
    parsed = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(parsed, list):
        raise ValueError("semantic case file must contain a YAML list")
    return {case["id"]: case for case in parsed if isinstance(case, dict) and "id" in case}


def _record_id(record: dict[str, Any]) -> str | None:
    value = record.get("id", record.get("case_id"))
    return value if isinstance(value, str) else None


def _response(record: dict[str, Any]) -> dict[str, Any]:
    value = record.get("response")
    return value if isinstance(value, dict) else {}


def _events(record: dict[str, Any]) -> list[dict[str, Any]]:
    events = record.get("events")
    return (
        [event for event in events if isinstance(event, dict)] if isinstance(events, list) else []
    )


def _has_fallback(record: dict[str, Any]) -> bool:
    return any(event.get("summary") in FALLBACK_SUMMARIES for event in _events(record))


def _has_gate_failure(record: dict[str, Any]) -> bool:
    return any(
        event.get("node") == "gate" and event.get("phase") == "failed" for event in _events(record)
    )


def _stable_order(values: Iterable[str], label: str) -> list[str]:
    def order_key(value: str) -> str:
        return hashlib.sha256(f"{SEED}\0{label}\0{value}".encode()).hexdigest()

    return sorted(values, key=order_key)


def _outcome(case: dict[str, Any], record: dict[str, Any]) -> dict[str, Any]:
    scored = score_case(case, record)
    return {
        "status": _response(record).get("status"),
        "citation_hit": bool(scored["anchor_hit"]),
        "citation_miss": not bool(scored["anchor_hit"]),
        "indirection": case.get("indirection"),
        "scoring_outcome": scored,
    }


def _remainder_strata() -> list[tuple[str, bool, str]]:
    return [
        (status, hit, band)
        for status in ("ANSWERABLE", "PARTIAL")
        for hit in (True, False)
        for band in ("none", "low", "high")
    ]


def _claim_block_texts(response: dict[str, Any]) -> list[str]:
    """Return the exact rendered CLAIM block text, excluding source quotations."""

    claims: list[str] = []
    for block in response.get("blocks", []):
        if not isinstance(block, dict):
            continue
        if isinstance(block.get("text"), str) and block.get("block_type") == "CLAIM":
            claims.append(block["text"].strip())
    return [text for text in claims if text]


def _is_template_only_response(response: dict[str, Any]) -> bool:
    claims = _claim_block_texts(response)
    return bool(claims) and all(
        is_deterministic_contextual_topic_template_prose(text) for text in claims
    )


def _has_substantive_prose(response: dict[str, Any]) -> bool:
    return bool(_claims(response))


def select_sample(
    records: Sequence[dict[str, Any]], cases: dict[str, dict[str, Any]]
) -> tuple[list[str], list[dict[str, Any]], dict[str, dict[str, Any]]]:
    """Select a fixed mix of at most four template-only and substantive answer cases."""

    eligible = {
        case_id: record
        for record in records
        if (case_id := _record_id(record)) in cases and cases[case_id].get("intent") == "answer"
    }
    outcomes = {case_id: _outcome(cases[case_id], record) for case_id, record in eligible.items()}
    audit_strata: list[dict[str, Any]] = []
    template_only_ids = {
        case_id
        for case_id, record in eligible.items()
        if _is_template_only_response(_response(record))
    }
    substantive_ids = {
        case_id
        for case_id, record in eligible.items()
        if case_id not in template_only_ids and _has_substantive_prose(_response(record))
    }
    selected_templates: list[str] = []
    selected_substantive: list[str] = []

    def fill_priority(
        label: str, values: Iterable[str], selected: list[str], capacity: int
    ) -> None:
        candidates = [
            case_id for case_id in _stable_order(values, label) if case_id not in selected
        ]
        chosen = candidates[: max(0, capacity - len(selected))]
        selected.extend(chosen)
        audit_strata.append(
            {
                "stratum": label,
                "available": len(candidates),
                "selected_case_ids": chosen,
                "state": "empty"
                if not candidates
                else ("filled" if chosen else "capacity_reached"),
            }
        )

    # Template-only outputs are observational examples, not a substitute for substantive work.
    fill_priority(
        "template_only_observational",
        template_only_ids,
        selected_templates,
        TEMPLATE_ONLY_LIMIT,
    )
    substantive_capacity = SAMPLE_SIZE - len(selected_templates)
    fill_priority(
        "substantive_fallback_rendered",
        (
            case_id
            for case_id in substantive_ids
            if _has_fallback(eligible[case_id])
        ),
        selected_substantive,
        substantive_capacity,
    )
    fill_priority(
        "substantive_evidence_or_gate_failure",
        (
            case_id
            for case_id in substantive_ids
            if _has_gate_failure(eligible[case_id])
        ),
        selected_substantive,
        substantive_capacity,
    )

    # One pass through every declared status × hit/miss × band stratum before repeating.
    remainder: dict[tuple[str, bool, str], list[str]] = {}
    for status, hit, band in _remainder_strata():
        label = f"remainder:status={status}:citation_hit={hit}:band={band}"
        remainder[(status, hit, band)] = _stable_order(
            (
                case_id
                for case_id, outcome in outcomes.items()
                if case_id in substantive_ids
                and case_id not in selected_substantive
                and outcome["status"] == status
                and outcome["citation_hit"] is hit
                and outcome["indirection"] == band
            ),
            label,
        )
        audit_strata.append(
            {
                "stratum": label,
                "available": len(remainder[(status, hit, band)]),
                "selected_case_ids": [],
                "state": "empty" if not remainder[(status, hit, band)] else "unfilled",
            }
        )
    remainder_offset = len(audit_strata) - len(remainder)
    while len(selected_substantive) < substantive_capacity and any(remainder.values()):
        for index, values in enumerate(remainder.values(), start=remainder_offset):
            if len(selected_substantive) == substantive_capacity:
                break
            if values:
                case_id = values.pop(0)
                selected_substantive.append(case_id)
                audit_strata[index]["selected_case_ids"].append(case_id)
                audit_strata[index]["state"] = "filled"
    return [*selected_substantive, *selected_templates], audit_strata, outcomes


def _claims(response: dict[str, Any]) -> list[str]:
    """Extract only rendered claim/point text, longest first, with exact duplicate removal."""

    claim_blocks = _claim_block_texts(response)
    has_claim_blocks = bool(claim_blocks)
    narrative_points: list[str] = []
    for block in response.get("blocks", []):
        if not isinstance(block, dict):
            continue
        if block.get("block_type") == "NARRATIVE":
            for segment in block.get("segments", []):
                if (
                    isinstance(segment, dict)
                    and segment.get("kind") in {"POINT", "SUMMARY"}
                    and isinstance(segment.get("text"), str)
                ):
                    narrative_points.append(segment["text"].strip())
    # The fixed contextual-topic fallback is explanatory UI copy, not a substantive claim.
    # Explicit rendered CLAIM blocks are otherwise the least ambiguous unit. Narrative points
    # are used only when the rendered response did not expose a substantive claim block.
    template_markers = (
        "الجواب المباشر: المقطع القرآني المسترجع يُعرض بسياقه المتصل",
        "Direct answer: the retrieved Quran passage is shown in its connected context",
    )
    claim_blocks = [
        text for text in claim_blocks if not any(marker in text for marker in template_markers)
    ]
    narrative_points = [
        text for text in narrative_points if not any(marker in text for marker in template_markers)
    ]
    candidates = claim_blocks if has_claim_blocks else narrative_points
    distinct = list(dict.fromkeys(value for value in candidates if value))
    return sorted(distinct, key=lambda value: (-len(value), value))[:2]


def _source_label(citation: dict[str, Any]) -> str:
    source_kind = citation.get("source_kind")
    if source_kind == "TAFSIR":
        return "Tafsir"
    if source_kind == "QURAN_TRANSLATION":
        return "Translation"
    if source_kind == "QURAN":
        return "Quran canonical text"
    return str(source_kind or "Source")


def _reference(citation: dict[str, Any]) -> str:
    value = citation.get("reference")
    if isinstance(value, dict) and isinstance(value.get("value"), str):
        return value["value"]
    return str(value or citation.get("display_label") or "Unknown reference")


def _trim_tafsir(text: str) -> str:
    if len(text) <= TAFSIR_REVIEW_CHARACTER_LIMIT:
        return text
    cut = max(
        text.rfind("\n", 0, TAFSIR_REVIEW_CHARACTER_LIMIT),
        text.rfind(" ", 0, TAFSIR_REVIEW_CHARACTER_LIMIT),
    )
    if cut <= 0:
        cut = TAFSIR_REVIEW_CHARACTER_LIMIT
    return text[:cut] + "\n\n[TAFSIR TEXT TRIMMED FOR REVIEW]\n"


def _sources(response: dict[str, Any]) -> list[dict[str, str]]:
    sources: list[dict[str, str]] = []
    for block in response.get("blocks", []):
        if not isinstance(block, dict) or block.get("block_type") != "SOURCE":
            continue
        citation = block.get("citation")
        text = block.get("exact_text")
        if not isinstance(citation, dict) or not isinstance(text, str):
            continue
        kind = _source_label(citation)
        sources.append(
            {
                "kind": kind,
                "reference": _reference(citation),
                "text": _trim_tafsir(text) if kind == "Tafsir" else text,
            }
        )
    return sources


def _response_review_lines(
    record: dict[str, Any], response_label: str, index: int
) -> tuple[list[str], int]:
    response = _response(record)
    claims = _claims(response)
    lines = [f"#### Response {response_label}", f"Status: {response.get('status', 'UNKNOWN')}", ""]
    if _is_template_only_response(response):
        lines.extend(
            [
                "This response contains only the fixed contextual display template, not a "
                "substantive claim. Do not assign per-claim support verdicts; assess only "
                "whether the response addresses the question.",
                "",
            ]
        )
    elif claims:
        for claim_number, claim in enumerate(claims, start=1):
            lines.extend(
                [
                    f"Claim {claim_number}",
                    claim,
                    "Verdict [supported / unsupported / unclear]: ",
                    "",
                ]
            )
    else:
        lines.extend(["No substantive claim was rendered.", ""])
    lines.extend(["Displayed sources", ""])
    for source in _sources(response):
        lines.extend([f"{source['kind']} — {source['reference']}", source["text"], ""])
    lines.extend(
        [
            "Addresses the question [yes / no / unclear]: ",
            "Optional notes: ",
            "",
        ]
    )
    return lines, len(claims)


def _response_model_lines(record: dict[str, Any], response_label: str) -> tuple[list[str], int]:
    """Render the same blinded material for a grader without any human verdict fields."""

    response = _response(record)
    claims = _claims(response)
    label = f" {response_label}" if response_label else ""
    lines = [f"### Response{label}", f"Status: {response.get('status', 'UNKNOWN')}", "Claims:"]
    if _is_template_only_response(response):
        lines.append(
            "This response contains only the fixed contextual display template, not a "
            "substantive claim. Return an empty claim_verdicts list and judge only whether it "
            "addresses the question."
        )
    elif claims:
        lines.extend(
            f"{claim_number}. {claim}" for claim_number, claim in enumerate(claims, start=1)
        )
    else:
        lines.append("No substantive claim was rendered.")
    lines.extend(["", "Displayed sources:"])
    for source in _sources(response):
        lines.extend([f"{source['kind']} — {source['reference']}", source["text"], ""])
    return lines, len(claims)


def _model_prompt_header() -> list[str]:
    return [
        "# Independent rendered-response support grading task",
        "",
        "Grade each response independently from the displayed sources. Provenance is not "
        "entailment: a verse being authentically displayed says nothing by itself about whether "
        "it supports the attached sentence. Use `unclear` when the displayed evidence does not "
        "settle the question; do not guess.",
        "",
        "For every listed claim, return exactly one of `supported`, `unsupported`, or `unclear`, "
        "an exact quoted span from the displayed evidence on which the verdict relies, and one "
        "sentence of reasoning. Also judge whether each response addresses its question. Do not "
        "infer any hidden evaluation labels.",
        "",
        "Return exactly one JSON object, with no Markdown fence, matching this schema:",
        "```json",
        "{",
        '  "responses": [',
        "    {",
        '      "case_number": 1,',
        '      "response_label": "",',
        '      "claim_verdicts": [',
        "        {",
        '          "claim_number": 1,',
        '          "verdict": "supported",',
        '          "evidence_span": "exact quoted span",',
        '          "reasoning": "one sentence"',
        "        }",
        "      ],",
        '      "addresses_question": "yes"',
        "    }",
        "  ]",
        "}",
        "```",
        "",
        SPOT_AUDIT_CAVEAT,
        "",
    ]


def _pair_order(case_id: str) -> tuple[str, str]:
    digest = hashlib.sha256(f"{SEED}\0pair-order\0{case_id}".encode()).digest()
    return ("baseline", "challenger") if digest[0] % 2 == 0 else ("challenger", "baseline")


def _composition(items: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    return {
        "stratum": dict(Counter(item["stratum"] for item in items)),
        "status": dict(Counter(item["status"] for item in items)),
        "indirection_band": dict(Counter(item["indirection"] for item in items)),
        "citation": dict(Counter("hit" if item["citation_hit"] else "miss" for item in items)),
    }


def build_audit(
    artifact: Path,
    *,
    output_dir: Path,
    name: str,
    cases_path: Path = DEFAULT_CASES,
    challenger: Path | None = None,
    for_model: bool = False,
) -> tuple[Path, Path, dict[str, Any]]:
    """Build blinded Markdown and a separate private key from one or two artifacts."""

    cases = load_cases(cases_path)
    baseline_records = {
        _record_id(record): record for record in read_records(artifact) if _record_id(record)
    }
    selected, selection_strata, outcomes = select_sample(list(baseline_records.values()), cases)
    challenger_records: dict[str, dict[str, Any]] | None = None
    if challenger is not None:
        challenger_records = {
            _record_id(record): record for record in read_records(challenger) if _record_id(record)
        }
        missing = [case_id for case_id in selected if case_id not in challenger_records]
        if missing:
            raise ValueError(f"challenger artifact is missing selected case ids: {missing}")

    output_dir.mkdir(parents=True, exist_ok=True)
    review_path = output_dir / f"audit_{name}.md"
    key_path = output_dir / f"audit_{name}_key.json"
    model_prompt_path = output_dir / f"audit_{name}_model_prompt.md"
    review = [
        "# Rendered-response support audit",
        "",
        f"Selection rule: {SEED}; a capped observational set of fixed-template responses and "
        "substantive responses prioritized by fallback, evidence/gate failure, then status × "
        "citation hit/miss × indirection-band remainder. The private key contains selection "
        "outcomes and mappings.",
        SPOT_AUDIT_CAVEAT,
        "",
    ]
    model_prompt = _model_prompt_header() if for_model else []
    key_items: list[dict[str, Any]] = []
    no_claims = 0
    template_only_responses = 0
    for number, case_id in enumerate(selected, start=1):
        case = cases[case_id]
        baseline = baseline_records[case_id]
        outcome = outcomes[case_id]
        matching_stratum = next(
            entry["stratum"] for entry in selection_strata if case_id in entry["selected_case_ids"]
        )
        review.extend(
            [
                f"## Case {number}",
                "",
                f"Question: {case['prompt']}",
                f"Indirection band: {case['indirection']}",
                "",
            ]
        )
        if for_model:
            model_prompt.extend(
                [
                    f"## Case {number}",
                    "",
                    f"Question: {case['prompt']}",
                    f"Indirection band: {case['indirection']}",
                    "",
                ]
            )
        if challenger_records is None:
            response_lines, claim_count = _response_review_lines(baseline, "", number)
            review.extend(response_lines)
            response_mapping = {"": "artifact"}
            no_claims += int(claim_count == 0)
            template_only = _is_template_only_response(_response(baseline))
            template_only_responses += int(template_only)
            response_metadata = [
                {
                    "label": "",
                    "artifact": "artifact",
                    "claim_count": claim_count,
                    "template_only": template_only,
                }
            ]
            if for_model:
                model_lines, _ = _response_model_lines(baseline, "")
                model_prompt.extend([*model_lines, ""])
        else:
            order = _pair_order(case_id)
            response_mapping = {"A": order[0], "B": order[1]}
            response_metadata = []
            for label, side in response_mapping.items():
                record = baseline if side == "baseline" else challenger_records[case_id]
                response_lines, claim_count = _response_review_lines(record, label, number)
                review.extend(response_lines)
                response_metadata.append(
                    {
                        "label": label,
                        "artifact": side,
                        "claim_count": claim_count,
                        "template_only": _is_template_only_response(_response(record)),
                    }
                )
                no_claims += int(claim_count == 0)
                template_only_responses += int(
                    _is_template_only_response(_response(record))
                )
                if for_model:
                    model_lines, _ = _response_model_lines(record, label)
                    model_prompt.extend([*model_lines, ""])
        key_items.append(
            {
                "case_number": number,
                "case_id": case_id,
                "stratum": matching_stratum,
                "status": outcome["status"],
                "citation_hit": outcome["citation_hit"],
                "indirection": outcome["indirection"],
                "scoring_outcome": outcome["scoring_outcome"],
                "response_mapping": response_mapping,
                "responses": response_metadata,
            }
        )
    review_path.write_text("\n".join(review).rstrip() + "\n", encoding="utf-8")
    key = {
        "seed": SEED,
        "selection_rule": (
            "up to four template_only_observational cases; substantive cases fill the remaining "
            "capacity by substantive_fallback_rendered, substantive_evidence_or_gate_failure, "
            "then stratified remainder"
        ),
        "sample_size": len(selected),
        "selection_strata": selection_strata,
        "composition": _composition(key_items),
        "no_substantive_claim_response_count": no_claims,
        "template_only_response_count": template_only_responses,
        "answer_case_population": {
            "template_only": sum(
                _is_template_only_response(_response(record))
                for record in baseline_records.values()
                if (case_id := _record_id(record)) in cases
                and cases[case_id].get("intent") == "answer"
            ),
            "substantive": sum(
                _has_substantive_prose(_response(record))
                and not _is_template_only_response(_response(record))
                for record in baseline_records.values()
                if (case_id := _record_id(record)) in cases
                and cases[case_id].get("intent") == "answer"
            ),
            "no_prose": sum(
                not _is_template_only_response(_response(record))
                and not _has_substantive_prose(_response(record))
                for record in baseline_records.values()
                if (case_id := _record_id(record)) in cases
                and cases[case_id].get("intent") == "answer"
            ),
        },
        "paired": challenger is not None,
        "items": key_items,
    }
    if for_model:
        model_prompt_path.write_text("\n".join(model_prompt).rstrip() + "\n", encoding="utf-8")
        key["model_prompt_path"] = str(model_prompt_path)
    key_path.write_text(json.dumps(key, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return review_path, key_path, key


_VERDICT = re.compile(r"^Verdict \[supported / unsupported / unclear\]:\s*(\w*)\s*$", re.MULTILINE)


def _expected_claims(key: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "case_number": item["case_number"],
            "response_label": response["label"],
            "claim_number": claim_number,
            "status": item["status"],
            "citation_hit": item["citation_hit"],
            "stratum": item["stratum"],
        }
        for item in key["items"]
        for response in item["responses"]
        for claim_number in range(1, response["claim_count"] + 1)
    ]


def _human_claim_verdicts(review_path: Path, key: dict[str, Any]) -> list[dict[str, Any]]:
    verdicts = [
        value.casefold() for value in _VERDICT.findall(review_path.read_text(encoding="utf-8"))
    ]
    expected = _expected_claims(key)
    if len(verdicts) != len(expected):
        raise ValueError(f"found {len(verdicts)} claim verdict fields; key expects {len(expected)}")
    return [
        {**metadata, "verdict": verdict}
        for verdict, metadata in zip(verdicts, expected, strict=True)
    ]


def read_verdicts(review_path: Path, key_path: Path | None = None) -> dict[str, Any]:
    """Read completed claim verdicts and join them to blinded key metadata."""

    if key_path is None:
        key_path = review_path.with_name(f"{review_path.stem}_key.json")
    key = json.loads(key_path.read_text(encoding="utf-8"))
    verdicts = _human_claim_verdicts(review_path, key)
    counts: dict[str, Counter[str]] = {
        "overall": Counter(),
        "by_status": Counter(),
        "by_citation": Counter(),
        "by_stratum": Counter(),
    }
    unfilled = 0
    for item in verdicts:
        verdict = item["verdict"]
        if verdict not in VERDICT_VALUES:
            unfilled += 1
            continue
        counts["overall"][verdict] += 1
        counts["by_status"][f"{item['status']}:{verdict}"] += 1
        citation = "hit" if item["citation_hit"] else "miss"
        counts["by_citation"][f"{citation}:{verdict}"] += 1
        counts["by_stratum"][f"{item['stratum']}:{verdict}"] += 1
    return {
        "caveat": "These counts are a spot audit, not an accuracy percentage for the product.",
        "claim_verdict_counts": {name: dict(values) for name, values in counts.items()},
        "unfilled_claim_verdicts": unfilled,
    }


def _load_model_verdicts(
    path: Path, expected: list[dict[str, Any]]
) -> dict[tuple[int, str, int], str]:
    """Validate the grader's declared JSON protocol without using it to alter the audit."""

    raw = path.read_text(encoding="utf-8").strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.IGNORECASE)
    parsed = json.loads(raw)
    responses = parsed.get("responses") if isinstance(parsed, dict) else None
    if not isinstance(responses, list):
        raise ValueError("model JSON must contain a responses list")
    observed: dict[tuple[int, str, int], str] = {}
    for response in responses:
        if not isinstance(response, dict):
            raise ValueError("model response entries must be objects")
        case_number = response.get("case_number")
        label = response.get("response_label")
        addresses = response.get("addresses_question")
        claims = response.get("claim_verdicts")
        if (
            not isinstance(case_number, int)
            or not isinstance(label, str)
            or addresses not in {"yes", "no", "unclear"}
            or not isinstance(claims, list)
        ):
            raise ValueError("model response entry does not match the declared schema")
        for claim in claims:
            if not isinstance(claim, dict):
                raise ValueError("model claim verdict entries must be objects")
            claim_number = claim.get("claim_number")
            verdict = claim.get("verdict")
            if (
                not isinstance(claim_number, int)
                or verdict not in VERDICT_VALUES
                or not isinstance(claim.get("evidence_span"), str)
                or not isinstance(claim.get("reasoning"), str)
            ):
                raise ValueError("model claim verdict does not match the declared schema")
            identity = (case_number, label, claim_number)
            if identity in observed:
                raise ValueError(f"duplicate model verdict for {identity}")
            observed[identity] = verdict
    required = {
        (item["case_number"], item["response_label"], item["claim_number"]) for item in expected
    }
    if set(observed) != required:
        raise ValueError("model verdicts do not cover exactly the claims in the blinded key")
    return observed


def compare_verdicts(
    human_path: Path, model_path: Path, key_path: Path | None = None
) -> dict[str, Any]:
    """Measure independent agreement; human verdicts remain the audit result."""

    if key_path is None:
        key_path = human_path.with_name(f"{human_path.stem}_key.json")
    key = json.loads(key_path.read_text(encoding="utf-8"))
    human = _human_claim_verdicts(human_path, key)
    model = _load_model_verdicts(model_path, _expected_claims(key))
    per_claim: list[dict[str, Any]] = []
    compared: list[tuple[str, str]] = []
    confusion: Counter[str] = Counter()
    direction: Counter[str] = Counter()
    order = {"unsupported": 0, "unclear": 1, "supported": 2}
    for item in human:
        identity = (item["case_number"], item["response_label"], item["claim_number"])
        human_verdict = item["verdict"]
        model_verdict = model[identity]
        valid = human_verdict in VERDICT_VALUES
        agrees = valid and human_verdict == model_verdict
        per_claim.append(
            {
                "case_number": item["case_number"],
                "response_label": item["response_label"],
                "claim_number": item["claim_number"],
                "human_verdict": human_verdict if valid else "unfilled",
                "model_verdict": model_verdict,
                "agrees": agrees,
            }
        )
        if not valid:
            continue
        compared.append((human_verdict, model_verdict))
        confusion[f"human:{human_verdict}|model:{model_verdict}"] += 1
        if human_verdict != model_verdict:
            if order[model_verdict] > order[human_verdict]:
                direction["model_more_generous"] += 1
            else:
                direction["model_less_generous"] += 1
    agreement = sum(left == right for left, right in compared)
    non_unclear = [(left, right) for left, right in compared if "unclear" not in {left, right}]
    non_unclear_agreement = sum(left == right for left, right in non_unclear)
    if compared:
        human_counts = Counter(left for left, _ in compared)
        model_counts = Counter(right for _, right in compared)
        total = len(compared)
        expected_agreement = (
            sum(human_counts[value] * model_counts[value] for value in VERDICT_VALUES) / total**2
        )
        observed_agreement = agreement / total
        kappa = (
            (observed_agreement - expected_agreement) / (1 - expected_agreement)
            if expected_agreement != 1
            else None
        )
    else:
        kappa = None
    return {
        "caveat": SPOT_AUDIT_CAVEAT,
        "judgement_role": (
            "Human verdicts are the audit result. Model verdicts are only a reference measurement "
            "of whether this grader can be trusted on this task; they are never blended into a "
            "score."
        ),
        "per_claim": per_claim,
        "overall_agreement": {"agreeing_claims": agreement, "compared_claims": len(compared)},
        "agreement_excluding_unclear": {
            "agreeing_claims": non_unclear_agreement,
            "compared_claims": len(non_unclear),
        },
        "confusion_breakdown": dict(confusion),
        "disagreement_direction": dict(direction),
        "chance_corrected_agreement_cohens_kappa": kappa,
        "unfilled_human_claim_verdicts": len(human) - len(compared),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--challenger", type=Path)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--name", default="semantic_v18")
    parser.add_argument("--verdicts", type=Path)
    parser.add_argument(
        "--compare-verdicts", type=Path, nargs=2, metavar=("HUMAN_FILE", "MODEL_JSON")
    )
    parser.add_argument("--key", type=Path)
    parser.add_argument("--for-model", action="store_true")
    arguments = parser.parse_args()
    if arguments.compare_verdicts is not None:
        human_file, model_json = arguments.compare_verdicts
        print(
            json.dumps(
                compare_verdicts(human_file, model_json, arguments.key),
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    if arguments.verdicts is not None:
        print(
            json.dumps(
                read_verdicts(arguments.verdicts, arguments.key), ensure_ascii=False, indent=2
            )
        )
        return 0
    if (arguments.baseline is None) != (arguments.challenger is None):
        parser.error("--baseline and --challenger must be supplied together")
    artifact = arguments.baseline or arguments.artifact
    review, key, summary = build_audit(
        artifact,
        output_dir=arguments.output_dir,
        name=arguments.name,
        cases_path=arguments.cases,
        challenger=arguments.challenger,
        for_model=arguments.for_model,
    )
    print(f"review: {review}")
    print(f"key: {key}")
    print(json.dumps(summary["composition"], ensure_ascii=False, indent=2))
    print(f"responses with no substantive claim: {summary['no_substantive_claim_response_count']}")
    print(
        "answer-case population: "
        f"{summary['answer_case_population']['template_only']} template-only, "
        f"{summary['answer_case_population']['substantive']} substantive, "
        f"{summary['answer_case_population']['no_prose']} no prose"
    )
    if arguments.for_model:
        print(f"model prompt: {summary['model_prompt_path']}")
    print(SPOT_AUDIT_CAVEAT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
