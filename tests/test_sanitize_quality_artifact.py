from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any


def sanitizer_module() -> Any:
    path = Path(__file__).resolve().parents[1] / "qa/sanitize_quality_artifact.py"
    spec = importlib.util.spec_from_file_location("sanitize_quality_artifact", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_sanitize_jsonl_redacts_only_quran_foundation_tafsir_and_is_idempotent(
    tmp_path: Path,
) -> None:
    sanitizer = sanitizer_module()
    raw_tafsir = "Quran Foundation tafsir text that cannot be retained."
    events = [{"summary": "Candidates dropped before evidence assembly", "detail": {}}]
    records = [
        {
            "id": "sem_quran_foundation",
            "events": events,
            "response": {
                "blocks": [
                    {
                        "block_type": "SOURCE",
                        "citation": {
                            "source_kind": "TAFSIR",
                            "source_revision": {"source_id": "quran_foundation"},
                        },
                        "exact_text": raw_tafsir,
                    }
                ]
            },
        },
        {
            "id": "sem_licensed_tafsir",
            "response": {
                "blocks": [
                    {
                        "block_type": "SOURCE",
                        "citation": {
                            "source_kind": "TAFSIR",
                            "source_revision": {"source_id": "locally_licensed_tafsir"},
                        },
                        "exact_text": "This licensed tafsir text is retained.",
                    }
                ]
            },
        },
    ]
    artifact = tmp_path / "semantic_responses.jsonl"
    original_text = "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records)
    artifact.write_text(original_text, encoding="utf-8")

    assert sanitizer.sanitize_artifact(artifact) == 1
    first_sanitized = artifact.read_text(encoding="utf-8")
    sanitized_records = [json.loads(line) for line in first_sanitized.splitlines()]
    redacted_block = sanitized_records[0]["response"]["blocks"][0]
    assert redacted_block["exact_text"] == "[REDACTED_QF_TAFSIR_TEXT_STORE_DENIED]"
    assert redacted_block["qa_redaction"] == {
        "original_length": len(raw_tafsir),
        "original_sha256": hashlib.sha256(raw_tafsir.encode("utf-8")).hexdigest(),
        "reason": "Quran Foundation source policy STORE=DENIED",
    }
    assert sanitized_records[0]["events"] == events
    assert sanitized_records[1]["response"]["blocks"][0]["exact_text"] == (
        "This licensed tafsir text is retained."
    )
    assert [record["id"] for record in sanitized_records] == [
        "sem_quran_foundation",
        "sem_licensed_tafsir",
    ]
    assert first_sanitized.splitlines(keepends=True)[1] == (
        original_text.splitlines(keepends=True)[1]
    )

    assert sanitizer.sanitize_artifact(artifact) == 0
    assert artifact.read_text(encoding="utf-8") == first_sanitized
