#!/usr/bin/env python3
"""Synchronise cached narration with the Playwright recording and encode an MP4."""

from __future__ import annotations

import argparse
import array
import json
import math
import re
import shutil
import subprocess
import sys
import tempfile
import wave
from pathlib import Path
from typing import Any

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


def run(command: list[str], *, capture: bool = False) -> subprocess.CompletedProcess[bytes]:
    result = subprocess.run(
        command, stdout=subprocess.PIPE if capture else subprocess.DEVNULL, stderr=subprocess.PIPE
    )
    if result.returncode:
        message = result.stderr.decode("utf-8", errors="replace")[-4000:]
        raise RuntimeError(f"Command failed ({result.returncode}): {message}")
    return result


def ffprobe_data(ffprobe: str, path: Path) -> dict[str, Any]:
    result = run(
        [ffprobe, "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)],
        capture=True,
    )
    return json.loads(result.stdout)


def media_duration(ffmpeg: str, path: Path, ffprobe: str | None) -> float:
    if ffprobe:
        info = ffprobe_data(ffprobe, path)
        return float(info.get("format", {}).get("duration", 0))
    result = subprocess.run([ffmpeg, "-hide_banner", "-i", str(path)], capture_output=True)
    match = re.search(rb"Duration: (\d+):(\d+):(\d+(?:\.\d+)?)", result.stderr)
    if not match:
        raise RuntimeError(f"Could not read media duration for {path.name}")
    hours, minutes, seconds = (float(part) for part in match.groups())
    return hours * 3600 + minutes * 60 + seconds


def detect_magenta_markers(ffmpeg: str, path: Path) -> list[float]:
    result = run(
        [
            ffmpeg,
            "-v",
            "error",
            "-i",
            str(path),
            "-vf",
            "fps=30,scale=1:1:flags=area,format=rgb24",
            "-f",
            "rawvideo",
            "-",
        ],
        capture=True,
    )
    pixels = result.stdout
    if len(pixels) % 3:
        raise RuntimeError("Decoded marker scan returned a partial RGB frame")
    frames = len(pixels) // 3
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for index in range(frames):
        red, green, blue = pixels[index * 3 : index * 3 + 3]
        is_magenta = red > 190 and green < 75 and blue > 190
        if is_magenta and start is None:
            start = index
        elif not is_magenta and start is not None:
            runs.append((start, index))
            start = None
    if start is not None:
        runs.append((start, frames))
    return [begin / 30 for begin, end in runs if end - begin >= 8]


def write_sync_report(
    timeline: list[dict[str, Any]], detected: list[float], report_path: Path
) -> tuple[float, float, float, dict[str, float]]:
    marker_times = {
        item["marker"]: float(item["t_start_ms"]) for item in timeline if "marker" in item
    }
    if "start" not in marker_times or "end" not in marker_times or len(detected) < 2:
        raise RuntimeError("Could not detect both magenta clapper markers in raw.webm")
    planned_start, planned_end = marker_times["start"], marker_times["end"]
    detected_start, detected_end = detected[0], detected[-1]
    offset = detected_start * 1000 - planned_start
    drift = ((detected_end - detected_start) * 1000) - (planned_end - planned_start)
    if abs(drift) > 500:
        raise RuntimeError(f"Recording drift is {drift:.1f}ms; redo the recording")
    measured_scale = ((detected_end - detected_start) * 1000) / (planned_end - planned_start)
    scale = measured_scale if abs(drift) > 40 else 1.0
    placed: dict[str, float] = {}
    for item in timeline:
        key = item.get("segment_key")
        if not key:
            continue
        planned = float(item["t_start_ms"])
        relative = planned - planned_start
        raw_time = detected_start * 1000 + relative * scale
        # Remove the opening 500ms marker from the final picture.
        placed[key] = max(0.0, raw_time - detected_start * 1000 - 500)
    report = {
        "offset_ms": round(offset, 2),
        "drift_ms": round(drift, 2),
        "scale": round(scale, 8),
        "detected_markers_s": [round(detected_start, 4), round(detected_end, 4)],
        "logged_markers_ms": {key: round(value, 2) for key, value in marker_times.items()},
        "clips": [
            {
                "segment_key": item["segment_key"],
                "planned_start_ms": round(float(item["t_start_ms"]), 2),
                "placed_start_ms": round(placed[item["segment_key"]], 2),
            }
            for item in timeline
            if item.get("segment_key")
        ],
    }
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return offset, drift, scale, placed


def compose(
    raw: Path,
    timeline_path: Path,
    manifest_path: Path,
    output: Path,
    *,
    selftest: bool = False,
    trim_waits: bool = False,
    selftest_onsets: tuple[float, ...] = (1.0, 5.0),
) -> dict[str, Any]:
    ffmpeg = ffmpeg_path()
    ffprobe = shutil.which("ffprobe")
    timeline = json.loads(timeline_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    detected = detect_magenta_markers(ffmpeg, raw)
    report_path = output.parent / "sync_report.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    _, drift, _, placed = write_sync_report(timeline, detected, report_path)
    offset = json.loads(report_path.read_text())["offset_ms"]
    print(f"Sync offset: {offset:.1f}ms; drift: {drift:.1f}ms")

    start, end = detected[0], detected[-1]
    marker_end = min(media_duration(ffmpeg, raw, ffprobe), end + 0.5)
    clips = [item for item in timeline if item.get("segment_key")]
    if not clips:
        raise RuntimeError("Timeline contains no narration clips")
    base_duration_s = media_duration(ffmpeg, raw, ffprobe) - 1.0
    if base_duration_s <= 0:
        raise RuntimeError("Raw recording is too short")
    intervals = f"between(t,{start:.6f},{start + 0.5:.6f})+between(t,{end:.6f},{marker_end:.6f})"
    gaps: list[tuple[float, float]] = []
    audio_starts: dict[str, float] = {}
    cursor = 0.0
    for item in sorted(clips, key=lambda clip: placed[clip["segment_key"]]):
        key = item["segment_key"]
        clip_start = placed[key] / 1000
        clip_end = clip_start + float(manifest[key]["duration_s"])
        if trim_waits and clip_start - cursor > 4:
            gaps.append((cursor, clip_start))
        cursor = max(cursor, clip_end)
        audio_starts[key] = clip_start
    if trim_waits and base_duration_s - cursor > 4:
        gaps.append((cursor, base_duration_s))
    saved_by_gap = [0.75 * (gap_end - gap_start) for gap_start, gap_end in gaps]
    duration_s = base_duration_s - sum(saved_by_gap)
    if gaps:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        report["trim_waits"] = {
            "speed": "4x",
            "silence_gaps_ms": [[round(a * 1000, 2), round(b * 1000, 2)] for a, b in gaps],
            "removed_ms": round(sum(saved_by_gap) * 1000, 2),
        }
        for clip in report["clips"]:
            key = clip["segment_key"]
            shift = sum(
                savings
                for (gap_start, gap_end), savings in zip(gaps, saved_by_gap, strict=True)
                if audio_starts[key] >= gap_end
            )
            clip["placed_start_ms"] = round(audio_starts[key] * 1000 - shift * 1000, 2)
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    if gaps:
        source_pts = "(PTS-STARTPTS)"
        savings_terms = []
        for gap_start, gap_end in gaps:
            savings_terms.append(
                "0.75*if(lt({p}\\,{a}/TB)\\,0\\,if(lt({p}\\,{b}/TB)\\,"
                "({p}-{a}/TB)\\,({length}/TB)))".format(
                    p=source_pts,
                    a=f"{gap_start:.6f}",
                    b=f"{gap_end:.6f}",
                    length=f"{gap_end - gap_start:.6f}",
                )
            )
        pts_filter = f"setpts={source_pts}-({' + '.join(savings_terms)})"
    else:
        # Playwright records variable-frame-rate video (few frames while the page is static).
        # Keep real timestamps; numbering frames as N/30 compressed 272 s of video into 226 s
        # and pulled the picture progressively ahead of the narration.
        pts_filter = "setpts=PTS-STARTPTS"
    video_filter = (
        f"select='not({intervals})',{pts_filter},"
        "scale=1920:1080:force_original_aspect_ratio=decrease,"
        "pad=1920:1080:(ow-iw)/2:(oh-ih)/2,fps=30,format=yuv420p"
    )
    command = [ffmpeg, "-y", "-i", str(raw)]
    clip_paths: list[Path] = []
    clip_delays: list[int] = []
    cumulative_savings = 0.0
    for item in clips:
        key = item["segment_key"]
        if key not in manifest:
            raise RuntimeError(f"Audio manifest is missing {key}")
        audio_path = ROOT / manifest[key]["file"]
        if not audio_path.is_file():
            raise RuntimeError(f"Audio file is missing for {key}")
        clip_paths.append(audio_path)
        clip_start = audio_starts[key]
        cumulative_savings = sum(
            savings
            for (gap_start, gap_end), savings in zip(gaps, saved_by_gap, strict=True)
            if clip_start >= gap_end
        )
        clip_delays.append(round(clip_start * 1000 - cumulative_savings * 1000))
        command.extend(["-i", str(audio_path)])
    chains = [f"[0:v]{video_filter}[v]"]
    labels = []
    for index, delay in enumerate(clip_delays, start=1):
        label = f"a{index}"
        chains.append(
            f"[{index}:a]aformat=sample_rates=48000:channel_layouts=stereo,adelay={delay}:all=1[{label}]"
        )
        labels.append(f"[{label}]")
    chains.append(
        f"{''.join(labels)}amix=inputs={len(labels)}:duration=longest:normalize=0,"
        f"apad,atrim=duration={duration_s:.6f},loudnorm=I=-16:TP=-1.5:LRA=11[a]"
    )
    command.extend(
        [
            "-filter_complex",
            ";".join(chains),
            "-map",
            "[v]",
            "-map",
            "[a]",
            "-t",
            f"{duration_s:.6f}",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "20",
            "-pix_fmt",
            "yuv420p",
            "-r",
            "30",
            "-c:a",
            "aac",
            "-b:a",
            "160k",
            "-movflags",
            "+faststart",
            str(output),
        ]
    )
    run(command)
    if selftest:
        verify_selftest(output, ffmpeg, ffprobe, duration_s, selftest_onsets)
    return {"duration_s": media_duration(ffmpeg, output, ffprobe), "ffprobe": ffprobe is not None}


def make_tone(path: Path, frequency: int) -> None:
    with wave.open(str(path), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(48_000)
        samples = array.array(
            "h",
            (int(12_000 * math.sin(2 * math.pi * frequency * i / 48_000)) for i in range(24_000)),
        )
        audio.writeframes(samples.tobytes())


def decoded_audio(ffmpeg: str, path: Path) -> array.array:
    result = run(
        [ffmpeg, "-v", "error", "-i", str(path), "-f", "s16le", "-ac", "2", "-ar", "48000", "-"],
        capture=True,
    )
    values = array.array("h")
    values.frombytes(result.stdout)
    return values


def verify_selftest(
    path: Path,
    ffmpeg: str,
    ffprobe: str | None,
    expected_duration: float,
    expected_onsets: tuple[float, ...],
) -> None:
    if ffprobe:
        info = ffprobe_data(ffprobe, path)
        codecs = {stream.get("codec_name") for stream in info.get("streams", [])}
        if not {"h264", "aac"}.issubset(codecs):
            raise RuntimeError(f"Selftest expected H.264 and AAC, got {codecs}")
        actual_duration = float(info["format"]["duration"])
    else:
        # FFmpeg's own demuxer reports the stream codecs when ffprobe is not installed.
        probe = subprocess.run([ffmpeg, "-hide_banner", "-i", str(path)], capture_output=True)
        details = probe.stderr.decode("utf-8", errors="replace")
        if "Video: h264" not in details or "Audio: aac" not in details:
            raise RuntimeError("Selftest output is missing H.264 or AAC")
        actual_duration = media_duration(ffmpeg, path, None)
    if abs(actual_duration - expected_duration) > 0.15:
        raise RuntimeError(
            f"Selftest duration {actual_duration:.3f}s differs from {expected_duration:.3f}s"
        )
    audio = decoded_audio(ffmpeg, path)
    # Require non-silent decoded samples around the two scheduled half-second tones.
    for start_s in expected_onsets:
        begin = round(start_s * 48_000) * 2
        section = audio[begin : begin + 12_000 * 2]
        if not section or max(abs(value) for value in section) < 100:
            raise RuntimeError(f"Selftest audio is missing at expected offset {start_s:.1f}s")
    verifier = "ffprobe" if ffprobe else "ffmpeg demuxer"
    onset_text = " and ".join(f"{onset:.3f}s" for onset in expected_onsets)
    print(
        f"Selftest passed: H.264 + AAC, {actual_duration:.2f}s, audio at {onset_text} ({verifier})"
    )


def selftest() -> None:
    ffmpeg = ffmpeg_path()
    with tempfile.TemporaryDirectory(prefix="bayan-demo-selftest-") as temporary:
        temp = Path(temporary)
        raw = temp / "raw.webm"
        tone_a, tone_b = temp / "tone-a.wav", temp / "tone-b.wav"
        timeline = temp / "timeline.json"
        manifest = temp / "manifest.json"
        output = temp / "selftest.mp4"
        trimmed_output = temp / "selftest-trimmed.mp4"
        command = [
            ffmpeg,
            "-y",
            "-f",
            "lavfi",
            "-i",
            "smptebars=size=1920x1080:rate=30:duration=10",
            "-f",
            "lavfi",
            "-i",
            "color=c=magenta:s=1920x1080:r=30:d=0.5",
            "-filter_complex",
            "[1:v][0:v][1:v]concat=n=3:v=1:a=0,"
            "select='if(between(t,0.5,10.5),not(eq(mod(n,5),1)),1)',"
            "setpts=PTS-STARTPTS,format=yuv420p[v]",
            "-map",
            "[v]",
            "-c:v",
            "libvpx-vp9",
            "-deadline",
            "realtime",
            "-fps_mode",
            "vfr",
            str(raw),
        ]
        run(command)
        if shutil.which("ffprobe"):
            probe = run(
                [
                    shutil.which("ffprobe") or "ffprobe",
                    "-v",
                    "error",
                    "-select_streams",
                    "v:0",
                    "-show_entries",
                    "frame=best_effort_timestamp_time",
                    "-of",
                    "csv=p=0",
                    str(raw),
                ],
                capture=True,
            )
            pts = [float(value) for value in probe.stdout.split() if value]
            intervals = {round(b - a, 4) for a, b in zip(pts, pts[1:], strict=False)}
            if len(intervals) < 2:
                raise RuntimeError("Selftest synthetic recording is not variable-frame-rate")
            print(f"VFR selftest source confirmed: {len(intervals)} frame intervals")
        make_tone(tone_a, 440)
        make_tone(tone_b, 880)
        timeline.write_text(
            json.dumps(
                [
                    {"marker": "start", "t_start_ms": 0},
                    {"segment_key": "test/0", "t_start_ms": 1500},
                    {"segment_key": "test/1", "t_start_ms": 7500},
                    {"marker": "end", "t_start_ms": 10_500},
                ]
            ),
            encoding="utf-8",
        )
        manifest.write_text(
            json.dumps(
                {
                    "test/0": {"file": str(tone_a), "duration_s": 0.5},
                    "test/1": {"file": str(tone_b), "duration_s": 0.5},
                }
            ),
            encoding="utf-8",
        )
        # Selftest manifests use absolute synthetic WAV paths.
        OUT.mkdir(parents=True, exist_ok=True)
        result = compose(raw, timeline, manifest, output, selftest=True, selftest_onsets=(1.0, 7.0))
        print(f"compose selftest output: {result['duration_s']:.2f}s")
        trimmed = compose(
            raw,
            timeline,
            manifest,
            trimmed_output,
            selftest=True,
            trim_waits=True,
            selftest_onsets=(1.0, 2.875),
        )
        if abs(trimmed["duration_s"] - 5.875) > 0.15:
            raise RuntimeError("Trim-waits selftest did not shorten the silent gap at 4x")
        trim_report = json.loads((temp / "sync_report.json").read_text(encoding="utf-8"))
        trim_placements = {
            item["segment_key"]: float(item["placed_start_ms"])
            for item in trim_report["clips"]
        }
        if abs(trim_placements["test/0"] - 1000) > 100 or abs(
            trim_placements["test/1"] - 2875
        ) > 100:
            raise RuntimeError(
                f"Trim-waits sync report disagrees with VFR audio schedule: {trim_placements}"
            )
        if abs(float(trim_report["trim_waits"]["removed_ms"]) - 4125) > 100:
            raise RuntimeError("Trim-waits sync report has an inconsistent removed duration")
        print(f"trim-waits selftest output: {trimmed['duration_s']:.2f}s")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--trim-waits", action="store_true", help="speed silent gaps over 4s at 4x")
    args = parser.parse_args()
    if args.selftest:
        selftest()
        return 0
    raw = OUT / "raw.webm"
    result = compose(
        raw,
        OUT / "timeline.json",
        ROOT / "audio/manifest.json",
        OUT / "bayan-demo.mp4",
        trim_waits=args.trim_waits,
    )
    print(f"Final video: {result['duration_s']:.2f}s")
    if not result["ffprobe"]:
        print("ffprobe unavailable; stream checks used ffmpeg's demuxer")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError, RuntimeError, json.JSONDecodeError) as error:
        print(f"compose: {error}", file=sys.stderr)
        sys.exit(1)
