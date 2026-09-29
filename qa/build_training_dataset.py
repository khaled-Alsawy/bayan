"""Build a review-only preference dataset while enforcing source training restrictions."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "artifacts/qa/live-api-quality-results.json"
OUTPUT = ROOT / "artifacts/qa/training-candidates.jsonl"
BLOCKED_SOURCE_IDS = {"quran_foundation", "qul"}


def main() -> None:
    payload = json.loads(INPUT.read_text(encoding="utf-8"))
    accepted: list[dict[str, object]] = []
    for item in payload.get("results", []):
        response = item.get("response", {})
        # A terminal notice has no preference target.  Keeping it in a training export merely
        # teaches the future model to emit generic failures, and previously let empty-evidence
        # failed-safe rows pass the source-licence filter below.
        if response.get("status") != "ANSWERABLE":
            continue
        citations = response.get("citations", [])
        if not citations:
            continue
        # QF has an explicit no-training rule. QUL has unknown per-resource licences, so it is
        # conservatively excluded too until a written licence changes the recorded policy.
        if any(
            citation.get("source_revision", {}).get("source_id") in BLOCKED_SOURCE_IDS
            for citation in citations
        ):
            continue
        narrative = next(
            (
                block
                for block in response.get("blocks", [])
                if block.get("block_type") == "NARRATIVE"
            ),
            None,
        )
        accepted.append(
            {
                "question": item.get("prompt"),
                "evidence": citations,
                "claims": [
                    block
                    for block in response.get("blocks", [])
                    if block.get("block_type") == "CLAIM"
                ],
                "narrative": narrative,
                # The public response deliberately does not disclose internal rule diagnostics;
                # the durable evaluation record keeps this observable post-gate result instead.
                "gate_verdict": "ANSWERABLE",
                "rating": None,
            }
        )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in accepted),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
