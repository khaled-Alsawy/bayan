#!/usr/bin/env python3
"""Transcribe every narration clip and flag clips whose speech differs from the script.

Catches the TTS reading the style instruction aloud, dropped or invented words, and wrong
language. Runs local faster-whisper (WHISPER_MODEL overrides the small default):
  .runtime/whisper/bin/python demo/audit_audio.py
Results are cached by clip hash in demo/out/audio_audit.json so re-runs are instant.
"""

from __future__ import annotations

import difflib
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "audio" / "manifest.json"
SCRIPT = ROOT / "script.json"
OUT = ROOT / "out" / "audio_audit.json"


_MODEL = None


def transcribe(path: Path) -> str:
    """Local faster-whisper: no quota, no network once the model is cached."""

    global _MODEL
    if _MODEL is None:
        from faster_whisper import WhisperModel

        _MODEL = WhisperModel(os.environ.get("WHISPER_MODEL", "Systran/faster-whisper-small"),
                              device="cpu", compute_type="int8")
    segments, _ = _MODEL.transcribe(str(path), language="ar", beam_size=5, vad_filter=False,
                                    condition_on_previous_text=False)
    return " ".join(segment.text.strip() for segment in segments).strip()


def normalize(text: str) -> str:
    text = re.sub(r"[ً-ْـ]", "", text)  # tashkeel, tatweel
    for source, target in (("أ", "ا"), ("إ", "ا"), ("آ", "ا"), ("ة", "ه"), ("ى", "ي")):
        text = text.replace(source, target)
    text = re.sub(r"[^\w\s]", " ", text.casefold())
    return " ".join(text.split())


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--clip", help="audit only this manifest segment key")
    args = parser.parse_args()
    script = json.loads(SCRIPT.read_text(encoding="utf-8"))
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    style_words = {w for w in normalize(script["voice_style"]).split() if len(w) > 3}
    scenes = script["scenes"] if isinstance(script, dict) else script
    texts = {
        f"{s['id']}/{i}": st["say"]
        for s in scenes
        for i, st in enumerate(s["steps"])
        if "say" in st
    }
    cache = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
    results, failures = {}, 0
    selected = {args.clip} if args.clip else set(manifest)
    unknown = selected - set(manifest)
    if unknown:
        raise SystemExit(f"Unknown clip key: {', '.join(sorted(unknown))}")
    report_path = ROOT / "out" / "audio_audit_report.json"
    existing_report = (
        json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {}
    )
    results.update(existing_report)
    for segment, item in manifest.items():
        if segment not in selected:
            continue
        if not (ROOT / item["file"]).is_file():
            continue
        digest = item.get("cache_key") or Path(item["file"]).stem
        if digest not in cache:
            cache[digest] = transcribe(ROOT / item["file"])
        heard = cache[digest]
        expected = texts[segment]
        ratio = difflib.SequenceMatcher(None, normalize(expected), normalize(heard)).ratio()
        heard_n, expected_n = normalize(heard), normalize(expected)
        leaked = len({w for w in style_words if w in heard_n and w not in expected_n})
        # Whisper small spells English terms in Arabic letters ("السانتسيس"), so similarity on
        # term-heavy clips sits near 0.55 even when the take is clean. Similarity only guards
        # against wrong content; leaks are caught by style words and repeats by word count.
        word_ratio = len(heard_n.split()) / max(1, len(expected_n.split()))
        ok = ratio >= 0.45 and leaked < 2 and word_ratio <= 1.35
        failures += not ok
        results[segment] = {"ok": ok, "similarity": round(ratio, 3), "style_words_heard": leaked,
                            "word_ratio": round(word_ratio, 2),
                            "cache_key": digest,
                            "expected": expected, "heard": heard}
        print(
            f"{'OK  ' if ok else 'FAIL'} {segment:24s} sim={ratio:.2f} "
            f"style_words={leaked} words={word_ratio:.2f}"
        )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(cache, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"{failures} of {len(results)} clips failed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
