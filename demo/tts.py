#!/usr/bin/env python3
"""Generate cached narration WAVs from the human-edited demo script."""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
import wave
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
SCRIPT = ROOT / "script.json"
AUDIO = ROOT / "audio"
MANIFEST = AUDIO / "manifest.json"
TAKES = AUDIO / "takes.json"
MODEL = "gemini-3.8-flash-tts"
VOICE = "Bodi"
SAMPLE_RATE = 24_000
MIN_REQUEST_INTERVAL_SECONDS = 30.0
_last_request_at: float | None = None
RETAKE_PROMPT_FORMAT = (
    "Read ONLY the transcript below aloud. Do not read these directions.\n"
    "Voice direction: {style}\n"
    "TRANSCRIPT:\n"
    "{text}"
)


def load_script() -> dict[str, Any]:
    value = json.loads(SCRIPT.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not isinstance(value.get("scenes"), list):
        raise ValueError("script.json must contain a scenes array")
    if not isinstance(value.get("voice_style"), str):
        raise ValueError("script.json must contain voice_style")
    ids: set[str] = set()
    for scene in value["scenes"]:
        if not isinstance(scene, dict) or not isinstance(scene.get("id"), str):
            raise ValueError("Each scene needs a string id")
        if scene["id"] in ids:
            raise ValueError(f"Duplicate scene id: {scene['id']}")
        ids.add(scene["id"])
        if not isinstance(scene.get("steps"), list):
            raise ValueError(f"Scene {scene['id']} needs a steps array")
        for step in scene["steps"]:
            if "say" in step and (not isinstance(step["say"], str) or not step["say"].strip()):
                raise ValueError(f"Empty narration in scene {scene['id']}")
            if step.get("act") == "ask" and not (step.get("text") or step.get("candidates")):
                raise ValueError(f"Question step in scene {scene['id']} needs text or candidates")
    return value


def segments(script: dict[str, Any]):
    for scene in script["scenes"]:
        for index, step in enumerate(scene["steps"]):
            if "say" in step:
                yield f"{scene['id']}/{index}", step["say"]


def segment_hash(text: str, style: str, take: int = 1) -> str:
    parts = (MODEL, VOICE, style, text)
    if take >= 2:
        parts += (f"take:{take}", RETAKE_PROMPT_FORMAT)
    payload = "\0".join(parts).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def duration(path: Path) -> float:
    with wave.open(str(path), "rb") as audio:
        return audio.getnframes() / audio.getframerate()


def api_key() -> str:
    try:
        result = subprocess.run(
            [
                "security",
                "find-generic-password",
                "-a",
                "islamic-research-ai",
                "-s",
                "gemini-api-key",
                "-w",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("Could not read the Gemini key from macOS Keychain") from exc
    key = result.stdout.strip()
    if not key:
        raise RuntimeError("Gemini API key is empty in macOS Keychain")
    return key


def generate(text: str, style: str, key: str, take: int) -> bytes:
    global _last_request_at
    prompt = style + text if take == 1 else RETAKE_PROMPT_FORMAT.format(style=style, text=text)
    body = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "responseModalities": ["AUDIO"],
            "speechConfig": {"voiceConfig": {"prebuiltVoiceConfig": {"voiceName": VOICE}}},
        },
    }
    request = urllib.request.Request(
        f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent",
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json", "x-goog-api-key": key},
        method="POST",
    )
    if _last_request_at is not None:
        time.sleep(max(0.0, MIN_REQUEST_INTERVAL_SECONDS - (time.monotonic() - _last_request_at)))
    _last_request_at = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            payload = json.loads(response.read())
        data = payload["candidates"][0]["content"]["parts"][0]["inlineData"]["data"]
        return base64.b64decode(data, validate=True)
    except urllib.error.HTTPError as exc:
        detail = ""
        try:
            error_payload = json.loads(exc.read())
            detail = str(error_payload.get("error", {}).get("message", ""))
        except (ValueError, AttributeError, TypeError):
            pass
        detail = detail.replace(key, "[redacted]")[:240]
        suffix = f": {detail}" if detail else ""
        raise RuntimeError(f"Gemini TTS request failed with HTTP {exc.code}{suffix}") from None
    except (urllib.error.URLError, TimeoutError) as exc:
        raise RuntimeError(f"Gemini TTS request failed: {type(exc).__name__}") from None
    except (KeyError, IndexError, ValueError, TypeError) as exc:
        raise RuntimeError("Gemini TTS returned an invalid audio response") from exc


def write_wav(path: Path, pcm: bytes) -> None:
    # gemini-3.8-flash-tts returns a complete WAV file (RIFF header, data chunk, then a C2PA
    # provenance chunk), not bare PCM. Written as PCM, the header plays as a click and the C2PA
    # text as a loud burst at the end of every clip, so keep only the data chunk's frames.
    rate, channels = SAMPLE_RATE, 1
    if pcm.startswith(b"RIFF"):
        with wave.open(io.BytesIO(pcm), "rb") as inner:
            if inner.getsampwidth() != 2:
                raise ValueError("Gemini returned WAV audio that is not 16-bit")
            rate, channels = inner.getframerate(), inner.getnchannels()
            pcm = inner.readframes(inner.getnframes())
    if len(pcm) < 2 or len(pcm) % 2:
        raise ValueError("Gemini returned empty or malformed 16-bit PCM audio")
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as output:
        output.setnchannels(channels)
        output.setsampwidth(2)
        output.setframerate(rate)
        output.writeframes(pcm)


def repair_cached_wav(path: Path) -> bool:
    """Unwrap a clip cached before write_wav understood Gemini's WAV payload. Idempotent."""

    with wave.open(str(path), "rb") as outer:
        frames = outer.readframes(outer.getnframes())
    if not frames.startswith(b"RIFF"):
        return False
    write_wav(path, frames)
    return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dry-run", action="store_true", help="validate and report cache status only"
    )
    args = parser.parse_args()
    script = load_script()
    takes: dict[str, int] = json.loads(TAKES.read_text(encoding="utf-8")) if TAKES.exists() else {}
    if any(not isinstance(value, int) or value < 1 for value in takes.values()):
        raise ValueError("takes.json values must be positive integers")
    old = json.loads(MANIFEST.read_text(encoding="utf-8")) if MANIFEST.exists() else {}
    manifest: dict[str, dict[str, str | float]] = {}
    total = 0.0
    todo: list[tuple[str, str, str, Path]] = []
    for key, text in segments(script):
        take = takes.get(key, 1)
        digest = segment_hash(text, script["voice_style"], take)
        path = AUDIO / f"{digest}.wav"
        cached = path.exists()
        if cached:
            clip_duration = duration(path)
        elif (
            key in old
            and (ROOT / old[key]["file"]).exists()
            and old[key].get("cache_key") == digest
        ):
            path = ROOT / old[key]["file"]
            clip_duration = duration(path)
            cached = True
        else:
            clip_duration = 0.0
            todo.append((key, text, digest, path))
        manifest[key] = {"file": path.relative_to(ROOT).as_posix(), "duration_s": clip_duration}
        total += clip_duration
        print(f"{key}: {'cached' if cached else 'uncached'}")
    if args.dry_run:
        print(
            f"{len(manifest)} narration segments; {len(todo)} uncached; "
            f"cached duration {total:.2f}s"
        )
        return 0
    for segment_key, item in manifest.items():
        if (ROOT / str(item["file"])).exists() and repair_cached_wav(ROOT / str(item["file"])):
            print(f"Repaired cached clip {segment_key}: removed nested WAV header and metadata")
    key = api_key() if todo else ""
    whisper_python = ROOT.parent / ".runtime/whisper/bin/python"
    audit_available = whisper_python.is_file()
    if todo and not audit_available:
        print(f"Warning: Whisper unavailable at {whisper_python}; skipping leak gate")
    for todo_index, (segment_key, text, digest, path) in enumerate(todo):
        take = takes.get(segment_key, 1)
        for generated_take in range(take, take + 3):
            digest = segment_hash(text, script["voice_style"], generated_take)
            path = AUDIO / f"{digest}.wav"
            print(f"Generating {segment_key} take {generated_take}", flush=True)
            try:
                audio = generate(text, script["voice_style"], key, generated_take)
            except RuntimeError as exc:
                if "HTTP 429" not in str(exc):
                    raise
                waiting = [segment_key, *(entry[0] for entry in todo[todo_index + 1 :])]
                print(f"{exc}", file=sys.stderr)
                print("Waiting for quota: " + ", ".join(waiting), file=sys.stderr)
                return 2
            write_wav(path, audio)
            manifest[segment_key] = {
                "file": path.relative_to(ROOT).as_posix(),
                "duration_s": duration(path),
                "cache_key": digest,
            }
            AUDIO.mkdir(parents=True, exist_ok=True)
            temporary = MANIFEST.with_suffix(".json.tmp")
            temporary.write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            os.replace(temporary, MANIFEST)
            if not audit_available:
                print(f"{segment_key} take {generated_take}: audit skipped (Whisper unavailable)")
                takes[segment_key] = generated_take
                break
            audit_process = subprocess.run(
                [str(whisper_python), str(ROOT / "audit_audio.py"), "--clip", segment_key],
                cwd=ROOT.parent,
                env={**os.environ, "HF_HUB_OFFLINE": "1"},
                capture_output=True,
                text=True,
            )
            try:
                report_path = ROOT / "out/audio_audit_report.json"
                report = json.loads(report_path.read_text(encoding="utf-8"))
                result = report[segment_key]
            except (OSError, ValueError, KeyError):
                result = {}
            if result.get("cache_key") != digest:
                if audit_process.stderr:
                    print(audit_process.stderr, end="", file=sys.stderr)
                print(f"Warning: Whisper unavailable for {segment_key}; skipping leak gate")
                takes[segment_key] = generated_take
                audit_available = False
                break
            print(
                f"{segment_key} take {generated_take}: "
                f"{'PASS' if result['ok'] else 'FAIL'} "
                f"similarity={result['similarity']:.3f} "
                f"style_words={result['style_words_heard']}"
            )
            if result["ok"]:
                takes[segment_key] = generated_take
                break
            # Persist the next attempt before asking Gemini again, so a 429 leaves a
            # resumable take number and the already failed audio is never treated as final.
            takes[segment_key] = generated_take + 1
            TAKES.write_text(
                json.dumps(takes, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            if generated_take == take + 2:
                print(f"{segment_key}: failed after 2 retakes")
        TAKES.write_text(json.dumps(takes, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for segment_key, text in segments(script):
        digest = segment_hash(text, script["voice_style"], takes.get(segment_key, 1))
        if segment_key not in manifest or manifest[segment_key].get("cache_key") != digest:
            path = AUDIO / f"{digest}.wav"
            if not path.exists():
                raise RuntimeError(
                    f"Missing take {takes.get(segment_key, 1)} audio for {segment_key}"
                )
            manifest[segment_key] = {
                "file": path.relative_to(ROOT).as_posix(),
                "duration_s": duration(path),
                "cache_key": digest,
            }
    AUDIO.mkdir(parents=True, exist_ok=True)
    temporary = MANIFEST.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, MANIFEST)
    print(
        f"Narration duration: {sum(float(item['duration_s']) for item in manifest.values()):.2f}s"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as error:
        print(f"tts: {error}", file=sys.stderr)
        sys.exit(1)
