"""Deterministically measure semantic retrieval-to-citation funnel stages from trace artifacts."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import yaml

try:
    from semantic_quality_runner import is_real_anchor
except ModuleNotFoundError:
    from qa.semantic_quality_runner import is_real_anchor

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CASES = ROOT / "qa/cases/semantic_cases.yaml"
_REFERENCE = re.compile(r"^(?P<chapter>[1-9]\d{0,2}):(?P<verse>[1-9]\d{0,2})$")
_SELECTION_SUMMARY = "Candidates dropped before evidence assembly"
_RELEVANCE_SUMMARY = "Topic relevance assessment completed"
_STAGES = (
    "late_binding_candidate_pool",
    "selected_anchors",
    "hydrated_structural_window",
    "retained_relevance_set",
    "final_citations",
)
_OUTPUT_KINDS = (
    "SUBSTANTIVE_SYNTHESIS",
    "SOURCE_ONLY",
    "TEMPLATE",
    "NO_CONTENT",
)


def load_cases(path: Path = DEFAULT_CASES) -> list[dict[str, Any]]:
    parsed = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(parsed, list):
        raise ValueError(f"{path} must contain a YAML list")
    return parsed


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


def _reference(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    if not is_real_anchor(value):
        return None
    return value


def _citation_references(record: dict[str, Any]) -> set[str]:
    response = record.get("response", record)
    citations = response.get("citations") if isinstance(response, dict) else None
    if not isinstance(citations, list):
        return set()
    references: set[str] = set()
    for citation in citations:
        if not isinstance(citation, dict):
            continue
        reference = citation.get("reference")
        value = reference.get("value") if isinstance(reference, dict) else reference
        if (parsed := _reference(value)) is not None:
            references.add(parsed)
    return references


def _event_references(detail: dict[str, Any], field: str) -> set[str]:
    candidates = detail.get(field)
    if not isinstance(candidates, list):
        return set()
    return {
        parsed
        for candidate in candidates
        if isinstance(candidate, dict)
        if (parsed := _reference(candidate.get("reference"))) is not None
    }


def _selection_references(record: dict[str, Any]) -> tuple[set[str], set[str]] | None:
    events = record.get("events")
    if not isinstance(events, list):
        return None
    selected: set[str] = set()
    dropped: set[str] = set()
    found = False
    for event in events:
        if not isinstance(event, dict) or event.get("summary") != _SELECTION_SUMMARY:
            continue
        detail = event.get("detail")
        if not isinstance(detail, dict):
            continue
        found = True
        selected.update(_event_references(detail, "selected"))
        dropped.update(_event_references(detail, "dropped"))
    return (selected, dropped) if found else None


def _retained_references(record: dict[str, Any]) -> set[str] | None:
    events = record.get("events")
    if not isinstance(events, list):
        return None
    for event in events:
        if not isinstance(event, dict) or event.get("summary") != _RELEVANCE_SUMMARY:
            continue
        detail = event.get("detail")
        if not isinstance(detail, dict) or "retained_references" not in detail:
            continue
        values = detail["retained_references"]
        if not isinstance(values, list):
            return None
        return {parsed for value in values if (parsed := _reference(value)) is not None}
    return None


def _in_structural_window(anchor: str, selected: set[str]) -> bool:
    anchor_match = _REFERENCE.fullmatch(anchor)
    if anchor_match is None:
        return False
    chapter = int(anchor_match.group("chapter"))
    verse = int(anchor_match.group("verse"))
    for reference in selected:
        selected_match = _REFERENCE.fullmatch(reference)
        if selected_match is None:
            continue
        if (
            int(selected_match.group("chapter")) == chapter
            and abs(int(selected_match.group("verse")) - verse) <= 2
        ):
            return True
    return False


def _rate(hits: int, cases: int) -> float:
    return hits / cases if cases else 0.0


def _output_kind_summary(observations: list[tuple[str, bool]], cases: int) -> dict[str, object]:
    """Summarize rendered output classifications, retaining old-artifact absence explicitly."""

    available_cases = len(observations)
    if not available_cases:
        return {
            "counts": None,
            "by_citation_hit": None,
            "available_case_count": 0,
            "unavailable_case_count": cases,
        }
    counts = {kind: 0 for kind in _OUTPUT_KINDS}
    by_citation_hit = {
        "hit": {kind: 0 for kind in _OUTPUT_KINDS},
        "miss": {kind: 0 for kind in _OUTPUT_KINDS},
    }
    for output_kind, citation_hit in observations:
        counts[output_kind] += 1
        by_citation_hit["hit" if citation_hit else "miss"][output_kind] += 1
    return {
        "counts": counts,
        "by_citation_hit": by_citation_hit,
        "available_case_count": available_cases,
        "unavailable_case_count": cases - available_cases,
    }


def _summarize(observations: list[dict[str, object]]) -> dict[str, object]:
    cases = len(observations)
    stages: dict[str, dict[str, object]] = {}
    for stage in _STAGES:
        available = [observation for observation in observations if observation[stage] is not None]
        hits = sum(observation[stage] is True for observation in available)
        available_cases = len(available)
        stages[stage] = {
            "hits": hits if available_cases else None,
            "rate": _rate(hits, available_cases) if available_cases else None,
            "available_case_count": available_cases,
            "unavailable_case_count": cases - available_cases,
        }
    distractor_hits = sum(observation["distractor"] is True for observation in observations)
    distinct_citations = [len(observation["citations"]) for observation in observations]
    return {
        "answer_cases": cases,
        "anchor_presence": stages,
        "distractor_rate": {
            "hits": distractor_hits,
            "rate": _rate(distractor_hits, cases),
        },
        "mean_distinct_quran_verses_cited": sum(distinct_citations) / cases if cases else 0.0,
    }


def compute_funnel(cases: list[dict[str, Any]], records: list[dict[str, Any]]) -> dict[str, object]:
    """Compute only stages supported by a case's retained selection trace event."""

    records_by_id = {
        record_id: record
        for record in records
        if isinstance(record_id := record.get("id", record.get("case_id")), str)
    }
    observations: list[dict[str, object]] = []
    unavailable_trace_case_ids: list[str] = []
    output_kind_observations: list[tuple[str, bool]] = []
    answer_cases = [
        case
        for case in cases
        if case.get("intent") == "answer" and isinstance(case.get("id"), str)
    ]
    for case in answer_cases:
        case_id = case["id"]
        record = records_by_id.get(case_id)
        acceptable = {
            anchor
            for anchor in case.get("acceptable_anchors", [])
            if _reference(anchor) is not None
        }
        if record is not None:
            response = record.get("response", record)
            output_kind = response.get("output_kind") if isinstance(response, dict) else None
            if output_kind in _OUTPUT_KINDS:
                output_kind_observations.append(
                    (output_kind, bool(acceptable.intersection(_citation_references(record))))
                )
        selection = _selection_references(record) if record is not None else None
        if selection is None:
            unavailable_trace_case_ids.append(case_id)
            continue
        selected, dropped = selection
        unacceptable = {
            anchor
            for anchor in case.get("unacceptable_anchors", [])
            if _reference(anchor) is not None
        }
        citations = _citation_references(record)
        retained = _retained_references(record)
        observations.append(
            {
                "indirection": case.get("indirection"),
                "late_binding_candidate_pool": bool(acceptable.intersection(selected | dropped)),
                "selected_anchors": bool(acceptable.intersection(selected)),
                "hydrated_structural_window": any(
                    _in_structural_window(anchor, selected) for anchor in acceptable
                ),
                "retained_relevance_set": (
                    bool(acceptable.intersection(retained)) if retained is not None else None
                ),
                "final_citations": bool(acceptable.intersection(citations)),
                "distractor": bool(unacceptable.intersection(citations)),
                "citations": citations,
            }
        )
    by_indirection = {
        level: _summarize(
            [observation for observation in observations if observation["indirection"] == level]
        )
        for level in ("none", "low", "high")
    }
    return {
        "analysis": "deterministic semantic retrieval funnel",
        "answer_case_count": len(answer_cases),
        "trace_answer_case_count": len(observations),
        "unavailable_trace_case_ids": unavailable_trace_case_ids,
        "output_kind": _output_kind_summary(output_kind_observations, len(answer_cases)),
        "overall": _summarize(observations),
        "by_indirection": by_indirection,
    }


def format_funnel(result: dict[str, object]) -> str:
    lines = ["semantic retrieval funnel"]
    summaries = {"overall": result["overall"], **result["by_indirection"]}  # type: ignore[arg-type]
    for label, summary in summaries.items():
        lines.append(f"{label}: {summary['answer_cases']} traced answer cases")  # type: ignore[index]
        for stage in _STAGES:
            value = summary["anchor_presence"][stage]  # type: ignore[index]
            if value["available_case_count"] == 0:  # type: ignore[index]
                lines.append(f"  {stage}: unavailable")
            else:
                unavailable_suffix = (
                    f"; unavailable for {value['unavailable_case_count']}"  # type: ignore[index]
                    if value["unavailable_case_count"]  # type: ignore[index]
                    else ""
                )
                lines.append(
                    f"  {stage}: {value['hits']}/{value['available_case_count']} "  # type: ignore[index]
                    f"({value['rate']:.1%}{unavailable_suffix})"  # type: ignore[index]
                )
        distractors = summary["distractor_rate"]  # type: ignore[index]
        lines.append(
            f"  distractor_rate: {distractors['hits']}/{summary['answer_cases']} "  # type: ignore[index]
            f"({distractors['rate']:.1%})"  # type: ignore[index]
        )
        lines.append(
            "  mean_distinct_quran_verses_cited: "
            f"{summary['mean_distinct_quran_verses_cited']:.2f}"  # type: ignore[index]
        )
    unavailable = result["unavailable_trace_case_ids"]
    if unavailable:
        lines.append(f"trace unavailable: {', '.join(unavailable)}")
    output_kind = result["output_kind"]  # type: ignore[assignment]
    if output_kind["available_case_count"] == 0:  # type: ignore[index]
        lines.append("output_kind: unavailable")
    else:
        unavailable_suffix = (
            f"; unavailable for {output_kind['unavailable_case_count']}"  # type: ignore[index]
            if output_kind["unavailable_case_count"]  # type: ignore[index]
            else ""
        )
        lines.append(f"output_kind counts{unavailable_suffix}:")
        for kind, count in output_kind["counts"].items():  # type: ignore[index,union-attr]
            lines.append(f"  {kind}: {count}")
        lines.append("output_kind by citation hit:")
        for citation_hit, counts in output_kind["by_citation_hit"].items():  # type: ignore[index,union-attr]
            lines.append(f"  citation_{citation_hit}: {counts}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute a deterministic retrieval-to-citation funnel from trace records"
    )
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--responses", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compute_funnel(load_cases(args.cases), read_records(args.responses))
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(format_funnel(result))


if __name__ == "__main__":
    main()
