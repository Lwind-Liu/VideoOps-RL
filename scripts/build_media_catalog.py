"""Create media.json and audit all S1 artifacts before downstream use."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from videoops_rl.media_audit import AuditCheck, check_records  # noqa: E402


def read_jsonl(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def probe(path: Path) -> dict[str, object]:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_format",
            "-show_streams",
            "-of",
            "json",
            str(path),
        ],
        text=True,
        capture_output=True,
        check=True,
    )
    return json.loads(result.stdout)


def relative(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()


def add_check(checks: list[AuditCheck], name: str, passed: bool, details: str) -> None:
    checks.append(AuditCheck(name, passed, details))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-dir", type=Path, required=True)
    parser.add_argument("--shot-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sample_dir = args.sample_dir.resolve()
    shot_dir = args.shot_dir.resolve()
    manifest_path = args.manifest.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    duration_ms = round(float(manifest["sample_timeline"]["end_seconds"]) * 1000)

    paths = {
        "sample": sample_dir / "sample_20s_500s.mp4",
        "video": sample_dir / "video_only.mp4",
        "audio_asr": sample_dir / "audio_asr_16k_mono.wav",
        "subtitles": sample_dir / "subtitles_en.srt",
        "shots": shot_dir / "shots.jsonl",
        "utterances": shot_dir / "utterances.jsonl",
        "evidence_units": shot_dir / "evidence_units.jsonl",
        "keyframes": shot_dir / "keyframes",
        "shot_summary": shot_dir / "summary.json",
    }
    missing = [name for name, path in paths.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required artifacts: {missing}")

    shots = read_jsonl(paths["shots"])
    utterances = read_jsonl(paths["utterances"])
    evidence_units = read_jsonl(paths["evidence_units"])
    checks = check_records(shots, utterances, evidence_units, shot_dir, duration_ms)

    sample_probe = probe(paths["sample"])
    video_probe = probe(paths["video"])
    audio_probe = probe(paths["audio_asr"])

    def duration_ok(payload: dict[str, object]) -> bool:
        return abs(float(payload["format"]["duration"]) * 1000 - duration_ms) <= 100

    add_check(
        checks,
        "media_durations_match_timeline",
        all(duration_ok(item) for item in (sample_probe, video_probe, audio_probe)),
        "sample, video-only, and ASR audio are within 100 ms of 480000 ms",
    )

    sample_types = [stream["codec_type"] for stream in sample_probe["streams"]]
    video_types = [stream["codec_type"] for stream in video_probe["streams"]]
    audio_stream = audio_probe["streams"][0]
    add_check(
        checks,
        "stream_layout_expected",
        sample_types.count("video") == 1
        and sample_types.count("audio") == 1
        and video_types == ["video"]
        and audio_stream["codec_type"] == "audio"
        and int(audio_stream["sample_rate"]) == 16_000
        and int(audio_stream["channels"]) == 1,
        f"sample={sample_types}, video_only={video_types}, asr=16000Hz mono",
    )

    manifest_hashes = {item["path"]: item["sha256"] for item in manifest["outputs"]}
    hashed_artifacts: dict[str, dict[str, object]] = {}
    hashes_match = True
    for name in ("sample", "video", "audio_asr", "subtitles"):
        path = paths[name]
        digest = sha256(path)
        relative_path = relative(path)
        hashed_artifacts[name] = {
            "path": relative_path,
            "bytes": path.stat().st_size,
            "sha256": digest,
        }
        hashes_match = hashes_match and manifest_hashes.get(relative_path) == digest
    add_check(
        checks,
        "manifest_hashes_match",
        hashes_match,
        "sample, video, audio, and subtitle hashes match frozen S1 manifest",
    )

    keyframes = sorted(paths["keyframes"].glob("shot_*.jpg"))
    shot_summary = json.loads(paths["shot_summary"].read_text(encoding="utf-8"))
    aggregate = hashlib.sha256()
    for path in keyframes:
        aggregate.update(path.name.encode("utf-8"))
        aggregate.update(bytes.fromhex(sha256(path)))

    catalog = {
        "schema_version": "1.0",
        "media_id": "tears_of_steel_s1_20s_500s",
        "parent_media_id": manifest["parent_media_id"],
        "timeline": {"start_ms": 0, "end_ms": duration_ms, "duration_ms": duration_ms},
        "artifacts": hashed_artifacts,
        "indexes": {
            "shots": relative(paths["shots"]),
            "utterances": relative(paths["utterances"]),
            "evidence_units": relative(paths["evidence_units"]),
            "keyframe_dir": relative(paths["keyframes"]),
        },
        "counts": {
            "shots": len(shots),
            "utterances": len(utterances),
            "evidence_units": len(evidence_units),
            "keyframes": len(keyframes),
        },
        "processing": {
            "shot_detector": shot_summary["method"],
            "shot_threshold": shot_summary["threshold"],
            "keyframe_policy": shot_summary["keyframe_policy"],
            "alignment": "half_open_interval_overlap",
        },
        "keyframes_sha256_aggregate": aggregate.hexdigest().upper(),
        "source_manifest": relative(manifest_path),
    }

    all_passed = all(check.passed for check in checks)
    report = {
        "schema_version": "1.0",
        "media_id": catalog["media_id"],
        "all_passed": all_passed,
        "passed": sum(check.passed for check in checks),
        "total": len(checks),
        "checks": [check.to_dict() for check in checks],
    }
    (sample_dir / "s1_audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"S1 audit: {report['passed']}/{report['total']} checks passed")
    if not all_passed:
        raise SystemExit(1)
    (sample_dir / "media.json").write_text(
        json.dumps(catalog, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(sample_dir / "media.json")


if __name__ == "__main__":
    main()
