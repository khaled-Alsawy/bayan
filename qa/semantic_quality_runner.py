"""Deterministic Layer 1 scoring for the semantic-retrieval evaluation cases.

This module deliberately does not grade prose. It measures only citations, anchors, and terminal
status; Layer 2 remains a human-review field until an independent reviewer is introduced.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CASES = ROOT / "qa/cases/semantic_cases.yaml"
_ANCHOR = re.compile(r"^(?P<chapter>\d{1,3}):(?P<verse>\d{1,3})$")
_VERSE_COUNTS = (
    7,
    286,
    200,
    176,
    120,
    165,
    206,
    75,
    129,
    109,
    123,
    111,
    43,
    52,
    99,
    128,
    111,
    110,
    98,
    135,
    112,
    78,
    118,
    64,
    77,
    227,
    93,
    88,
    69,
    60,
    34,
    30,
    73,
    54,
    45,
    83,
    182,
    88,
    75,
    85,
    54,
    53,
    89,
    59,
    37,
    35,
    38,
    29,
    18,
    45,
    60,
    49,
    62,
    55,
    78,
    96,
    29,
    22,
    24,
    13,
    14,
    11,
    11,
    18,
    12,
    12,
    30,
    52,
    52,
    44,
    28,
    28,
    20,
    56,
    40,
    31,
    50,
    40,
    46,
    42,
    29,
    19,
    36,
    25,
    22,
    17,
    19,
    26,
    30,
    20,
    15,
    21,
    11,
    8,
    8,
    19,
    5,
    8,
    8,
    11,
    11,
    8,
    3,
    9,
    5,
    4,
    7,
    3,
    6,
    3,
    5,
    4,
    5,
    6,
)
_EXPECTED_STATUSES = {
    "answer": "ANSWERABLE",
    "clarify": "CLARIFICATION_REQUIRED",
    "refuse": "UNSUPPORTED_SCOPE",
}
_REQUIRED_FIELDS = {
    "id",
    "ui_language",
    "prompt",
    "indirection",
    "expected_themes",
    "acceptable_anchors",
    "unacceptable_anchors",
    "forbidden_terms",
    "intent",
    "notes",
}


def is_real_anchor(value: str) -> bool:
    """Check a Qur'an chapter:verse reference against canonical surah verse counts."""

    match = _ANCHOR.fullmatch(value)
    if match is None:
        return False
    chapter = int(match.group("chapter"))
    verse = int(match.group("verse"))
    return 1 <= chapter <= len(_VERSE_COUNTS) and 1 <= verse <= _VERSE_COUNTS[chapter - 1]


def _normalise_text(value: str) -> str:
    return " ".join(value.casefold().split())


def load_cases(path: Path = DEFAULT_CASES) -> list[dict[str, Any]]:
    parsed = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(parsed, list):
        raise ValueError(f"{path} must contain a YAML list")
    return parsed


def validate_cases(cases: list[dict[str, Any]]) -> dict[str, int]:
    """Reject an under-specified or lexically compromised semantic case manifest."""

    errors: list[str] = []
    ids: set[str] = set()
    levels: Counter[str] = Counter()
    intents: Counter[str] = Counter()
    for index, case in enumerate(cases, start=1):
        if not isinstance(case, dict):
            errors.append(f"case {index} is not a mapping")
            continue
        missing = _REQUIRED_FIELDS.difference(case)
        if missing:
            errors.append(f"case {index} missing {sorted(missing)}")
            continue
        case_id = case["id"]
        if not isinstance(case_id, str) or not case_id.startswith("sem_"):
            errors.append(f"case {index} has invalid id {case_id!r}")
        elif case_id in ids:
            errors.append(f"duplicate case id {case_id}")
        else:
            ids.add(case_id)
        if case["ui_language"] not in {"ar", "en"}:
            errors.append(f"{case_id}: ui_language must be ar or en")
        if case["indirection"] not in {"none", "low", "high"}:
            errors.append(f"{case_id}: invalid indirection")
        else:
            levels[case["indirection"]] += 1
        if case["intent"] not in _EXPECTED_STATUSES:
            errors.append(f"{case_id}: invalid intent")
        else:
            intents[case["intent"]] += 1
        if not isinstance(case["prompt"], str) or len(case["prompt"].strip()) < 2:
            errors.append(f"{case_id}: prompt is empty")
        for field in (
            "expected_themes",
            "acceptable_anchors",
            "unacceptable_anchors",
            "forbidden_terms",
        ):
            if not isinstance(case[field], list) or not all(
                isinstance(value, str) for value in case[field]
            ):
                errors.append(f"{case_id}: {field} must be a list of strings")
        for anchor in [*case["acceptable_anchors"], *case["unacceptable_anchors"]]:
            if not is_real_anchor(anchor):
                errors.append(f"{case_id}: invalid Qur'an anchor {anchor!r}")
        if set(case["acceptable_anchors"]).intersection(case["unacceptable_anchors"]):
            errors.append(f"{case_id}: acceptable and unacceptable anchors overlap")
        if case["intent"] == "answer" and not case["acceptable_anchors"]:
            errors.append(f"{case_id}: answer cases require acceptable anchors")
        if case["intent"] != "answer" and case["acceptable_anchors"]:
            errors.append(f"{case_id}: non-answer cases cannot have acceptable anchors")
        if case["indirection"] != "none":
            prompt = _normalise_text(case["prompt"])
            leaked = [term for term in case["forbidden_terms"] if _normalise_text(term) in prompt]
            if leaked:
                errors.append(f"{case_id}: lexical giveaway(s) in prompt: {leaked}")
    if len(cases) < 60:
        errors.append(f"expected at least 60 cases, found {len(cases)}")
    for level, minimum in {"none": 15, "low": 20, "high": 25}.items():
        if levels[level] < minimum:
            errors.append(f"expected at least {minimum} {level} cases, found {levels[level]}")
    non_answer = intents["clarify"] + intents["refuse"]
    if non_answer < 8:
        errors.append(f"expected about 10 non-answer controls (minimum 8), found {non_answer}")
    if errors:
        raise ValueError("Invalid semantic case manifest:\n- " + "\n- ".join(errors))
    return {
        "case_count": len(cases),
        **{f"{level}_count": levels[level] for level in ("none", "low", "high")},
        **{f"{intent}_count": intents[intent] for intent in _EXPECTED_STATUSES},
    }


def _citation_anchor(citation: dict[str, Any]) -> str | None:
    reference = citation.get("reference")
    if isinstance(reference, dict) and isinstance(reference.get("value"), str):
        return reference["value"]
    if isinstance(citation.get("reference"), str):
        return citation["reference"]
    return None


def _response_citations(record: dict[str, Any]) -> list[dict[str, Any]]:
    response = record.get("response", record)
    citations = response.get("citations", []) if isinstance(response, dict) else []
    return (
        [citation for citation in citations if isinstance(citation, dict)]
        if isinstance(citations, list)
        else []
    )


def _retrieved_evidence(record: dict[str, Any]) -> tuple[set[str], set[str], bool]:
    raw = record.get("retrieved_evidence")
    if raw is None:
        return set(), set(), False
    entries = raw.values() if isinstance(raw, dict) else raw
    if (
        not isinstance(entries, list)
        and not isinstance(entries, tuple)
        and not isinstance(entries, set)
    ):
        return set(), set(), False
    evidence_ids: set[str] = set()
    anchors: set[str] = set()
    for entry in entries:
        if isinstance(entry, str):
            if is_real_anchor(entry):
                anchors.add(entry)
            else:
                evidence_ids.add(entry)
        elif isinstance(entry, dict):
            evidence_id = entry.get("evidence_id")
            if isinstance(evidence_id, str):
                evidence_ids.add(evidence_id)
            anchor = _citation_anchor(entry)
            if anchor is not None:
                anchors.add(anchor)
    return evidence_ids, anchors, True


def score_case(case: dict[str, Any], record: dict[str, Any]) -> dict[str, Any]:
    """Score Layer 1 only; no prose/rubric judgement is performed here."""

    response = record.get("response", record)
    status = response.get("status") if isinstance(response, dict) else None
    citations = _response_citations(record)
    anchors = {
        anchor for citation in citations if (anchor := _citation_anchor(citation)) is not None
    }
    retrieved_ids, retrieved_anchors, evidence_present = _retrieved_evidence(record)
    citation_resolves = all(is_real_anchor(anchor) for anchor in anchors)
    citation_in_evidence = evidence_present and all(
        (isinstance(citation.get("evidence_id"), str) and citation["evidence_id"] in retrieved_ids)
        or ((_citation_anchor(citation) or "") in retrieved_anchors)
        for citation in citations
    )
    expected_status = _EXPECTED_STATUSES[case["intent"]]
    if case["intent"] == "answer":
        anchor_hit = bool(anchors.intersection(case["acceptable_anchors"]))
        distractor_hit = bool(anchors.intersection(case["unacceptable_anchors"]))
    else:
        anchor_hit = None
        distractor_hit = bool(anchors.intersection(case["unacceptable_anchors"]))
    status_matches_intent = status == expected_status
    layer_1_pass = (
        citation_resolves
        and citation_in_evidence
        and status_matches_intent
        and (anchor_hit is not False)
        and not distractor_hit
    )
    return {
        "id": case["id"],
        "indirection": case["indirection"],
        "intent": case["intent"],
        "status": status,
        "expected_status": expected_status,
        "returned_anchors": sorted(anchors),
        "citation_resolves": citation_resolves,
        "citation_in_retrieved_evidence": citation_in_evidence,
        "retrieved_evidence_present": evidence_present,
        "anchor_hit": anchor_hit,
        "distractor_hit": distractor_hit,
        "status_matches_intent": status_matches_intent,
        "layer_1_pass": layer_1_pass,
        "layer_2": {"scored_by": None, "human_confirmed": False, "status": "UNSET"},
    }


def _read_records(path: Path) -> list[dict[str, Any]]:
    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        return []
    parsed = (
        json.loads(raw)
        if raw.startswith("[")
        else [json.loads(line) for line in raw.splitlines() if line.strip()]
    )
    if not isinstance(parsed, list) or not all(isinstance(item, dict) for item in parsed):
        raise ValueError("response input must be a JSON array or JSONL object stream")
    return parsed


def score_records(cases: list[dict[str, Any]], records: list[dict[str, Any]]) -> dict[str, Any]:
    cases_by_id = {case["id"]: case for case in cases}
    records_by_id = {record.get("id", record.get("case_id")): record for record in records}
    unknown = sorted(str(case_id) for case_id in records_by_id if case_id not in cases_by_id)
    outcomes = [
        score_case(case, records_by_id[case_id])
        for case_id, case in cases_by_id.items()
        if case_id in records_by_id
    ]
    missing = sorted(case_id for case_id in cases_by_id if case_id not in records_by_id)
    answer_outcomes = [outcome for outcome in outcomes if outcome["intent"] == "answer"]
    by_indirection: dict[str, dict[str, int | float]] = {}
    for level in ("none", "low", "high"):
        subset = [outcome for outcome in answer_outcomes if outcome["indirection"] == level]
        hits = sum(outcome["anchor_hit"] is True for outcome in subset)
        by_indirection[level] = {
            "answer_cases": len(subset),
            "anchor_hits": hits,
            "anchor_hit_rate": hits / len(subset) if subset else 0.0,
        }
    distractor_hits = sum(outcome["distractor_hit"] for outcome in outcomes)
    matrix: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for outcome in outcomes:
        matrix[outcome["intent"]][str(outcome["status"])] += 1
    return {
        "scoring_layer": "Layer 1 deterministic",
        "case_count": len(cases),
        "scored_case_count": len(outcomes),
        "missing_case_ids": missing,
        "unknown_record_ids": unknown,
        "anchor_hit": {
            "answer_cases": len(answer_outcomes),
            "hits": sum(outcome["anchor_hit"] is True for outcome in answer_outcomes),
            "rate": sum(outcome["anchor_hit"] is True for outcome in answer_outcomes)
            / len(answer_outcomes)
            if answer_outcomes
            else 0.0,
            "by_indirection": by_indirection,
            "none_minus_high_gap": by_indirection["none"]["anchor_hit_rate"]
            - by_indirection["high"]["anchor_hit_rate"],
        },
        "distractor_hit": {
            "cases": len(outcomes),
            "hits": distractor_hits,
            "rate": distractor_hits / len(outcomes) if outcomes else 0.0,
        },
        "status_vs_intent": {intent: dict(row) for intent, row in matrix.items()},
        "outcomes": outcomes,
        "layer_2": (
            "UNSET — indicative model scoring is intentionally not a quality measurement; "
            "human review is required."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score semantic retrieval cases with deterministic Layer 1 checks"
    )
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument(
        "--responses", type=Path, help="JSON/JSONL records with response and retrieved_evidence"
    )
    parser.add_argument("--output", type=Path, help="Write a JSON score artifact")
    parser.add_argument(
        "--validate-cases", action="store_true", help="Validate only the YAML manifest"
    )
    args = parser.parse_args()
    cases = load_cases(args.cases)
    summary = validate_cases(cases)
    if args.validate_cases:
        print(json.dumps({"manifest": summary, "layer_2": "UNSET"}, ensure_ascii=False, indent=2))
        return
    if args.responses is None:
        parser.error("--responses is required unless --validate-cases is used")
    result = score_records(cases, _read_records(args.responses))
    encoded = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output is not None:
        args.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)


if __name__ == "__main__":
    main()
