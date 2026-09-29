from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "artifacts/qa/live-api-quality-results.json"
REDACTED_TEXT = "[REDACTED_QF_TAFSIR_TEXT_STORE_DENIED]"
REDACTION_REASON = "Quran Foundation source policy STORE=DENIED"


def _is_quran_foundation_tafsir_block(value: object) -> bool:
    if not isinstance(value, dict):
        return False
    citation = value.get("citation")
    if (
        value.get("block_type") != "SOURCE"
        or not isinstance(citation, dict)
        or citation.get("source_kind") != "TAFSIR"
    ):
        return False
    revision = citation.get("source_revision")
    return isinstance(revision, dict) and revision.get("source_id") == "quran_foundation"


def _redact_quran_foundation_tafsir_blocks(value: object) -> int:
    if isinstance(value, list):
        return sum(_redact_quran_foundation_tafsir_blocks(item) for item in value)
    if not isinstance(value, dict):
        return 0

    redactions = 0
    if _is_quran_foundation_tafsir_block(value):
        raw = value.get("exact_text", "")
        if raw != REDACTED_TEXT:
            raw_text = raw if isinstance(raw, str) else json.dumps(raw, ensure_ascii=False)
            value["exact_text"] = REDACTED_TEXT
            value["qa_redaction"] = {
                "original_length": len(raw_text),
                "original_sha256": hashlib.sha256(raw_text.encode("utf-8")).hexdigest(),
                "reason": REDACTION_REASON,
            }
            redactions += 1
    return redactions + sum(_redact_quran_foundation_tafsir_blocks(item) for item in value.values())


def _serialize_jsonl_line(record: object, ending: str) -> str:
    return json.dumps(record, ensure_ascii=False) + ending


def _sanitize_jsonl(text: str) -> tuple[str, int]:
    redactions = 0
    lines: list[str] = []
    for line in text.splitlines(keepends=True):
        ending = "\n" if line.endswith("\n") else ""
        content = line[: -len(ending)] if ending else line
        try:
            record = json.loads(content)
        except json.JSONDecodeError:
            lines.append(line)
            continue
        count = _redact_quran_foundation_tafsir_blocks(record)
        redactions += count
        lines.append(_serialize_jsonl_line(record, ending) if count else line)
    return "".join(lines), redactions


def _sanitize_json(text: str) -> tuple[str, int]:
    payload = json.loads(text)
    redactions = _redact_quran_foundation_tafsir_blocks(payload)
    if redactions and isinstance(payload, dict):
        payload["qa_artifact_redactions"] = {
            "qf_tafsir_source_blocks": redactions,
            "policy": "No exact Quran Foundation Tafsir text retained in QA artifact",
        }
    if not redactions:
        return text, 0
    return json.dumps(payload, ensure_ascii=False, indent=2) + "\n", redactions


def sanitize_artifact(path: Path) -> int:
    """Redact stored Quran Foundation tafsir text from a JSON or JSONL artifact."""

    text = path.read_text(encoding="utf-8")
    try:
        sanitized, redactions = _sanitize_json(text)
    except json.JSONDecodeError:
        sanitized, redactions = _sanitize_jsonl(text)
    if redactions:
        path.write_text(sanitized, encoding="utf-8")
    return redactions


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Redact stored Quran Foundation tafsir text from QA artifacts"
    )
    parser.add_argument("paths", nargs="*", type=Path)
    args = parser.parse_args()
    paths = args.paths or [TARGET]
    for path in paths:
        redactions = sanitize_artifact(path)
        if args.paths:
            print(f"{path}: redacted_qf_tafsir_source_blocks={redactions}")
        else:
            print(f"redacted_qf_tafsir_source_blocks={redactions}")


if __name__ == "__main__":
    main()
