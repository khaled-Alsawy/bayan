#!/usr/bin/env python3
"""Exercise every demo question against the live loopback API before recording."""

from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any
from uuid import uuid4

ROOT = Path(__file__).resolve().parent
API = "http://127.0.0.1:8000"


def post(url: str, payload: dict[str, Any], timeout: float) -> Any:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.status, response.read()


def ask(conversation_id: str, question: str, locale: str) -> dict[str, Any]:
    request = urllib.request.Request(
        f"{API}/v1/conversations/{conversation_id}/turns/stream",
        data=json.dumps(
            {"message": question, "locale": locale, "idempotency_key": str(uuid4())},
            ensure_ascii=False,
        ).encode("utf-8"),
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
        method="POST",
    )
    final: dict[str, Any] | None = None
    with urllib.request.urlopen(request, timeout=180) as response:
        event = ""
        data: list[str] = []
        for raw_line in response:
            line = raw_line.decode("utf-8").rstrip("\r\n")
            if line.startswith("event:"):
                event = line[6:].strip()
            elif line.startswith("data:"):
                data.append(line[5:].lstrip())
            elif not line and event:
                if event == "final" and data:
                    final = json.loads("\n".join(data))
                elif event == "error":
                    raise RuntimeError("API returned a safe failure while streaming")
                event, data = "", []
    if final is None:
        raise RuntimeError("API stream ended without a final turn")
    return final


def status_output_kind_and_claim_citations(turn: dict[str, Any]) -> tuple[str, str, bool]:
    response = turn.get("response", {})
    output_kind = str(response.get("output_kind", "UNKNOWN"))
    claim_has_quran_citation = any(
        block.get("block_type") == "CLAIM"
        and any(
            citation.get("source_kind") == "QURAN"
            for citation in block.get("citations", [])
        )
        for block in response.get("blocks", [])
    )
    status = str(response.get("status", "UNKNOWN"))
    return status, output_kind, claim_has_quran_citation


def outcome(status: str, output_kind: str, claim_has_quran_citation: bool) -> str:
    if (
        status in {"ANSWERABLE", "PARTIAL"}
        and output_kind != "SOURCE_ONLY"
        and claim_has_quran_citation
    ):
        return "grounded"
    if status in {"INSUFFICIENT_EVIDENCE", "UNSUPPORTED_SCOPE"}:
        return "honest_refusal"
    return status.lower()


def main() -> int:
    script = json.loads((ROOT / "script.json").read_text(encoding="utf-8"))
    _, created = post(f"{API}/v1/conversations", {"locale": "ar"}, 15)
    conversation_id = json.loads(created)["conversation_id"]
    rows: list[dict[str, Any]] = []
    selected: dict[str, str] = {}
    failed = False
    for scene in script["scenes"]:
        for index, step in enumerate(scene["steps"]):
            if step.get("act") != "ask":
                continue
            questions = step["candidates"] if step.get("candidates") else [step["text"]]
            chosen: str | None = None
            is_candidate_step = bool(step.get("candidates"))
            for question in questions:
                locale = "en" if all(ord(character) < 128 for character in question) else "ar"
                started = time.monotonic()
                turn = ask(conversation_id, question, locale)
                elapsed = time.monotonic() - started
                status, output_kind, claim_has_quran_citation = (
                    status_output_kind_and_claim_citations(turn)
                )
                actual = outcome(status, output_kind, claim_has_quran_citation)
                expected = step.get("expect", "")
                passed = actual == expected
                if is_candidate_step:
                    passed = passed or (expected == "honest_refusal" and actual == "honest_refusal")
                rows.append(
                    {
                        "scene_id": scene["id"],
                        "step_index": index,
                        "question": question,
                        "expect": expected,
                        "outcome": actual,
                        "status": status,
                        "quran_citation": claim_has_quran_citation,
                        "elapsed_s": round(elapsed, 2),
                        "passed": passed,
                    }
                )
                print(
                    f"{scene['id']}: {actual} | output_kind: {output_kind} "
                    f"| Quran claim citation: {'yes' if claim_has_quran_citation else 'no'} "
                    f"| {elapsed:.2f}s | {'PASS' if passed else 'FAIL'}",
                    flush=True,
                )
                if expected == "honest_refusal" and actual == "honest_refusal":
                    chosen = question
                    break
                if not passed and not is_candidate_step:
                    failed = True
            if step.get("candidates") and chosen:
                selected[f"{scene['id']}/{index}"] = chosen
            elif step.get("candidates") and chosen is None:
                failed = True
    output = ROOT / "out" / "preflight.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps({"rows": rows, "selected_questions": selected}, ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )
    return 1 if failed else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError, urllib.error.URLError, RuntimeError) as error:
        print(f"preflight: {error}", file=sys.stderr)
        sys.exit(1)
