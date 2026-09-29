#!/usr/bin/env python3
"""Build a timestamped visual and transcript review page for the final demo."""

from __future__ import annotations

import html
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "out"


def ffmpeg_path() -> str:
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError as exc:
        candidate = shutil.which("ffmpeg")
        if candidate:
            return candidate
        raise RuntimeError(
            "ffmpeg is unavailable; install the demo extra or put ffmpeg on PATH"
        ) from exc


def extract(ffmpeg: str, video: Path, timestamp: float, image: Path) -> None:
    result = subprocess.run(
        [ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-ss", f"{timestamp:.3f}",
         "-i", str(video), "-frames:v", "1", "-vf", "scale=640:-2", str(image)],
        capture_output=True,
        text=True,
    )
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or f"Frame extraction failed at {timestamp:.3f}s")
    if not image.is_file() or image.stat().st_size == 0:
        raise RuntimeError(f"No frame extracted at {timestamp:.3f}s from {video.name}")


def main() -> int:
    video = OUT / "bayan-demo.mp4"
    manifest = json.loads((ROOT / "audio/manifest.json").read_text(encoding="utf-8"))
    timeline = json.loads((OUT / "sync_report.json").read_text(encoding="utf-8"))
    report = json.loads((OUT / "audio_audit_report.json").read_text(encoding="utf-8"))
    if not video.is_file():
        raise RuntimeError(f"Final video is missing: {video}")
    ffmpeg = ffmpeg_path()
    review_dir = OUT / "review"
    review_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    placed = {
        item["segment_key"]: float(item["placed_start_ms"]) / 1000
        for item in timeline["clips"]
    }
    for key, clip in manifest.items():
        if key not in placed:
            raise RuntimeError(f"Sync report has no placed time for {key}")
        audit = report.get(key)
        if audit is None:
            raise RuntimeError(f"Audio audit report has no result for {key}")
        start = placed[key]
        duration = float(clip["duration_s"])
        timestamps = [start + 0.4, start + duration / 2, start + duration - 0.3]
        images = []
        slug = key.replace("/", "-")
        for label, timestamp in zip(("start", "middle", "end"), timestamps, strict=True):
            if timestamp < 0:
                timestamp = 0
            name = f"{slug}-{label}.png"
            extract(ffmpeg, video, timestamp, review_dir / name)
            images.append((label, timestamp, name))
        passed = bool(audit.get("ok"))
        status = "PASS" if passed else "FAIL"
        image_html = "".join(
            f'<figure><img src="{html.escape(name)}" '
            f'alt="{html.escape(label)} frame at {timestamp:.2f}s">'
            f"<figcaption>{html.escape(label)} · {timestamp:.2f}s</figcaption></figure>"
            for label, timestamp, name in images
        )
        rows.append(
            f'<tr class="{"pass" if passed else "fail"}"><th>{html.escape(key)}<br>'
            f'<small>placed {start:.2f}s · take {clip.get("take", "—")}</small></th>'
            f'<td>{html.escape(status)} · similarity {float(audit.get("similarity", 0)):.3f} · '
            f'style words {int(audit.get("style_words_heard", 0))}</td>'
            '<td><details><summary>Expected</summary>'
            f'{html.escape(str(audit.get("expected", "")))}</details>'
            '<details><summary>Heard</summary>'
            f'{html.escape(str(audit.get("heard", "")))}</details></td>'
            f'<td class="frames">{image_html}</td></tr>'
        )
    document = """<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Bayān demo review</title>
<style>
body{font:15px/1.5 system-ui,sans-serif;margin:24px;background:#f4f3ee;color:#172522}
h1{margin:0 0 18px}
table{border-collapse:collapse;width:100%;background:#fff}
th,td{border:1px solid #d5d9d3;padding:12px;vertical-align:top;text-align:left}
th{width:170px}.pass{background:#e9f5e9}.fail{background:#ffe7e5}
.frames{display:flex;gap:10px;flex-wrap:wrap}
figure{margin:0;width:min(31%,640px)}
img{display:block;width:100%;height:auto;border:1px solid #aaa}
figcaption{font-size:12px;color:#555}
details{margin:0 0 8px;max-width:450px;white-space:pre-wrap}
small{color:#5b6561}
</style><h1>Bayān demo review</h1><table><thead>
<tr><th>Clip</th><th>Audio audit</th><th>Transcript</th><th>Final MP4 frames</th></tr></thead>
<tbody>__ROWS__</tbody></table></html>""".replace("__ROWS__", "\n".join(rows))
    (review_dir / "index.html").write_text(document, encoding="utf-8")
    print(f"Review page: {review_dir / 'index.html'} ({len(rows)} clips)")
    print(
        "Last clip frame timestamps: "
        f"{timestamps[0]:.2f}s, {timestamps[1]:.2f}s, {timestamps[2]:.2f}s"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError, RuntimeError, json.JSONDecodeError) as error:
        print(f"review: {error}", file=sys.stderr)
        sys.exit(1)
