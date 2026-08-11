"""Detect shot boundaries and extract one midpoint keyframe per shot."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from videoops_rl.shots import Shot, build_shots  # noqa: E402


FRAME_RE = re.compile(r"pts_time:(?P<time>[0-9.]+)")
SCORE_RE = re.compile(r"lavfi\.scene_score=(?P<score>[0-9.]+)")


def run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, text=True, capture_output=True, check=True)


def probe_video(input_path: Path) -> tuple[float, float]:
    result = run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "format=duration:stream=avg_frame_rate",
            "-of",
            "json",
            str(input_path),
        ]
    )
    payload = json.loads(result.stdout)
    duration_s = float(payload["format"]["duration"])
    numerator, denominator = payload["streams"][0]["avg_frame_rate"].split("/")
    fps = float(numerator) / float(denominator)
    return duration_s, fps


def detect_transitions(input_path: Path, threshold: float) -> list[tuple[float, float]]:
    result = run(
        [
            "ffmpeg",
            "-hide_banner",
            "-nostats",
            "-i",
            str(input_path),
            "-vf",
            f"select=gt(scene\\,{threshold}),metadata=print",
            "-an",
            "-f",
            "null",
            "NUL" if sys.platform == "win32" else "/dev/null",
        ]
    )

    transitions: list[tuple[float, float]] = []
    pending_time: float | None = None
    for line in result.stderr.splitlines():
        frame_match = FRAME_RE.search(line)
        if frame_match:
            pending_time = float(frame_match.group("time"))
            continue
        score_match = SCORE_RE.search(line)
        if score_match and pending_time is not None:
            transitions.append((pending_time, float(score_match.group("score"))))
            pending_time = None
    if pending_time is not None:
        raise RuntimeError("FFmpeg returned a transition time without a scene score")
    return transitions


def extract_keyframes(
    input_path: Path,
    shots: list[Shot],
    output_dir: Path,
    fps: float,
    batch_size: int = 40,
) -> None:
    """Extract midpoint frames in bounded batches to avoid huge FFmpeg commands."""
    keyframe_dir = output_dir / "keyframes"
    keyframe_dir.mkdir(parents=True, exist_ok=True)
    frame_indexes = [max(0, round(shot.keyframe_s * fps)) for shot in shots]
    for old_image in keyframe_dir.glob("shot_*.jpg"):
        old_image.unlink()

    with tempfile.TemporaryDirectory(prefix="keyframe_batches_", dir=output_dir) as temporary:
        temporary_dir = Path(temporary)
        global_index = 1
        for batch_index, start in enumerate(range(0, len(frame_indexes), batch_size), start=1):
            batch = frame_indexes[start : start + batch_size]
            unique_frames = list(dict.fromkeys(batch))
            select_expression = "+".join(f"eq(n\\,{frame_index})" for frame_index in unique_frames)
            pattern = temporary_dir / f"batch_{batch_index:04d}_%04d.jpg"
            run(
                [
                    "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                    "-i", str(input_path), "-vf", f"select={select_expression}",
                    "-fps_mode", "vfr", "-q:v", "2", str(pattern),
                ]
            )
            batch_images = sorted(temporary_dir.glob(f"batch_{batch_index:04d}_*.jpg"))
            if len(batch_images) != len(unique_frames):
                # Some nominal frame indexes do not exist in variable-frame-rate
                # files. Fall back to timestamp seeking for this small batch.
                for shot in shots[start : start + batch_size]:
                    target = keyframe_dir / f"shot_{global_index:04d}.jpg"
                    run(
                        [
                            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                            "-ss", f"{shot.keyframe_s:.6f}", "-i", str(input_path),
                            "-frames:v", "1", "-q:v", "2", str(target),
                        ]
                    )
                    global_index += 1
                continue
            image_by_frame = dict(zip(unique_frames, batch_images))
            for frame_index in batch:
                shutil.copy2(image_by_frame[frame_index], keyframe_dir / f"shot_{global_index:04d}.jpg")
                global_index += 1

    images = sorted(keyframe_dir.glob("shot_*.jpg"))
    if len(images) != len(shots):
        raise RuntimeError(f"Expected {len(shots)} keyframes, found {len(images)}")


def write_outputs(
    input_path: Path,
    output_dir: Path,
    threshold: float,
    duration_s: float,
    fps: float,
    shots: list[Shot],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "shots.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for shot in shots:
            record = shot.to_dict()
            record["keyframe_path"] = f"keyframes/{shot.shot_id}.jpg"
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    durations = [shot.duration_s for shot in shots]
    summary = {
        "method": "ffmpeg_scene_score",
        "input": input_path.as_posix(),
        "threshold": threshold,
        "duration_s": duration_s,
        "fps": fps,
        "transition_count": max(0, len(shots) - 1),
        "shot_count": len(shots),
        "shot_duration_s": {
            "min": min(durations),
            "median": statistics.median(durations),
            "mean": statistics.mean(durations),
            "max": max(durations),
        },
        "boundary_semantics": "transition timestamp is the first frame of the new shot",
        "keyframe_policy": "temporal midpoint of each shot",
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--threshold", type=float, default=0.35)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input.is_file():
        raise FileNotFoundError(args.input)
    if not 0.0 < args.threshold < 1.0:
        raise ValueError("threshold must be between 0 and 1")
    for executable in ("ffmpeg", "ffprobe"):
        if shutil.which(executable) is None:
            raise RuntimeError(f"{executable} is not available on PATH")

    duration_s, fps = probe_video(args.input)
    transitions = detect_transitions(args.input, args.threshold)
    shots = build_shots(transitions, duration_s)
    extract_keyframes(args.input, shots, args.output_dir, fps)
    write_outputs(args.input, args.output_dir, args.threshold, duration_s, fps, shots)
    print(f"Detected {len(transitions)} transitions and wrote {len(shots)} shots")
    print(args.output_dir / "shots.jsonl")


if __name__ == "__main__":
    main()
