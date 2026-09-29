from __future__ import annotations

import pytest

from demo.preflight import outcome, status_output_kind_and_claim_citations


@pytest.mark.parametrize(
    ("output_kind", "blocks"),
    [
        (
            "SOURCE_ONLY",
            [
                {
                    "block_type": "SOURCE",
                    "citation": {"source_kind": "QURAN"},
                }
            ],
        ),
        (
            "SUBSTANTIVE_SYNTHESIS",
            [
                {
                    "block_type": "CLAIM",
                    "citations": [{"source_kind": "TAFSIR"}],
                }
            ],
        ),
    ],
)
def test_grounded_preflight_requires_non_source_only_claim_with_quran_citation(
    output_kind: str, blocks: list[dict[str, object]]
) -> None:
    turn = {
        "response": {
            "status": "ANSWERABLE",
            "output_kind": output_kind,
            "blocks": blocks,
            # Top-level Quran citations alone used to make source-only turns pass.
            "citations": [{"source_kind": "QURAN"}],
        }
    }
    status, actual_output_kind, claim_has_quran_citation = (
        status_output_kind_and_claim_citations(turn)
    )
    actual = outcome(status, actual_output_kind, claim_has_quran_citation)
    step = {"expect": "grounded"}

    assert actual != step["expect"]
