"""Part D deterministic status-consistency scoring for equivalent semantic cases."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CASES = ROOT / "qa/cases/semantic_cases.yaml"
DEFAULT_PAIRS = ROOT / "qa/cases/equivalent_query_pairs.yaml"


def load_yaml(path: Path) -> list[dict[str, Any]]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise ValueError(f"{path} must contain a YAML list of mappings")
    return value


def validate_pairs(cases: list[dict[str, Any]], pairs: list[dict[str, Any]]) -> None:
    case_ids = {case.get("id") for case in cases}
    if len(pairs) != 10:
        raise ValueError("Part D requires exactly 10 equivalent-query pairs")
    for pair in pairs:
        if not all(isinstance(pair.get(key), str) for key in ("id", "first", "second")):
            raise ValueError(f"Invalid equivalent pair: {pair!r}")
        if pair["first"] == pair["second"] or not {pair["first"], pair["second"]} <= case_ids:
            raise ValueError(f"Unknown or duplicate pair members: {pair!r}")


def score_pairs(
    pairs: list[dict[str, Any]], records: dict[str, dict[str, str]]
) -> dict[str, Any]:
    """Score statuses from both orders; different statuses are a failure, never an average."""

    outcomes = []
    for pair in pairs:
        forward = records.get(f"{pair['id']}:forward", {})
        reverse = records.get(f"{pair['id']}:reverse", {})
        statuses = {
            forward.get(pair["first"]),
            forward.get(pair["second"]),
            reverse.get(pair["first"]),
            reverse.get(pair["second"]),
        }
        statuses.discard(None)
        complete = all(
            member in order
            for order in (forward, reverse)
            for member in (pair["first"], pair["second"])
        )
        outcomes.append(
            {
                "id": pair["id"],
                "complete": complete,
                "statuses": sorted(statuses),
                "consistent": complete and len(statuses) == 1,
            }
        )
    failures = [outcome for outcome in outcomes if not outcome["consistent"]]
    return {
        "pair_count": len(outcomes),
        "status_delta_count": len(failures),
        "status_delta_rate": len(failures) / len(outcomes) if outcomes else 0.0,
        "outcomes": outcomes,
    }


def main() -> None:
    cases = load_yaml(DEFAULT_CASES)
    pairs = load_yaml(DEFAULT_PAIRS)
    validate_pairs(cases, pairs)
    print(json.dumps({"pair_count": len(pairs), "orders_required": ["forward", "reverse"]}))


if __name__ == "__main__":
    main()
