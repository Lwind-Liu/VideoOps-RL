"""Convert the existing real-video smoke sample into the formal P1 protocol."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from videoops_rl.dataset_protocol import (  # noqa: E402
    HARD_LIMIT_BYTES, TARGET_LIMIT_BYTES, audit_manifest, package_inventory, read_jsonl, sha256_file,
)


def artifact(path: Path) -> dict:
    return {"path": path.relative_to(REPO_ROOT).as_posix(), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def main() -> None:
    media_path = REPO_ROOT / "data/processed/tears_of_steel/s1/media.json"
    media = json.loads(media_path.read_text(encoding="utf-8"))
    evidence_path = REPO_ROOT / media["indexes"]["evidence_units"]
    units = read_jsonl(evidence_path)
    source_tasks = read_jsonl(REPO_ROOT / "data/annotations/highlight_tasks.jsonl")
    video_id = media["media_id"]

    registry_dir = REPO_ROOT / "data/registry"
    registry_dir.mkdir(parents=True, exist_ok=True)
    normalized_task_path = registry_dir / "p1_smoke_tasks.jsonl"
    normalized: list[dict] = []
    for task in source_tasks:
        shot_ids = [unit["shot_id"] for unit in units if unit["end_ms"] > task["target_start_ms"] and unit["start_ms"] < task["target_end_ms"]]
        normalized.append({
            "schema_version": "videoops.task.v1",
            "task_id": task["task_id"], "video_id": video_id, "split": "smoke", "query": task["query"],
            "target_segments": [{"start_ms": task["target_start_ms"], "end_ms": task["target_end_ms"]}],
            "evidence": {"search_terms": task["search_terms"], "shot_ids": shot_ids, "required_modalities": ["text", "image"]},
            "provenance": {
                "source": "manually_reviewed_open_film_demo", "original_split_ignored": task["split"],
                "claim_boundary": "smoke only; one source video cannot measure cross-video generalization",
            },
        })
    normalized_task_path.write_text("".join(json.dumps(task, ensure_ascii=False, separators=(",", ":")) + "\n" for task in normalized), encoding="utf-8")

    sample_path = REPO_ROOT / media["artifacts"]["sample"]["path"]
    keyframes = REPO_ROOT / media["indexes"]["keyframe_dir"]
    keyframe_files = sorted(keyframes.glob("*.jpg"))
    manifest = {
        "schema_version": "videoops.dataset_manifest.v1", "dataset_id": "videoops_p1_smoke_v1",
        "scope": {
            "role": "smoke", "business_task": "query_driven_highlight_localization",
            "claim_boundary": "protocol and pipeline validation on one open film; not benchmark generalization",
        },
        "capacity": {
            "hard_limit_bytes": HARD_LIMIT_BYTES, "target_limit_bytes": TARGET_LIMIT_BYTES,
            "policy": "final offline package must contain code, data, model and environment without server downloads",
        },
        "videos": [{
            "video_id": video_id, "split": "smoke", "duration_ms": media["timeline"]["duration_ms"], "license": "CC BY 3.0",
            "artifacts": [
                artifact(sample_path), artifact(REPO_ROOT / media["artifacts"]["subtitles"]["path"]),
                artifact(evidence_path), artifact(REPO_ROOT / media["indexes"]["utterances"]),
            ],
            "keyframe_inventory": {
                "directory": keyframes.relative_to(REPO_ROOT).as_posix(), "count": len(keyframe_files),
                "bytes": sum(path.stat().st_size for path in keyframe_files),
                "aggregate_sha256": media["keyframes_sha256_aggregate"],
            },
        }],
        "tasks": {"path": normalized_task_path.relative_to(REPO_ROOT).as_posix(), "count": len(normalized), "schema": "schemas/videoops_task_v1.schema.json"},
        "split_policy": {
            "unit": "video_id", "production_splits": ["train", "val", "test"],
            "rule": "one video_id may appear in exactly one production split", "smoke_excluded_from_metrics": True,
        },
        "current_inventory": package_inventory(REPO_ROOT),
    }
    manifest_path = registry_dir / "p1_dataset_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    audit = audit_manifest(manifest, REPO_ROOT, verify_hashes=True)
    report_dir = REPO_ROOT / "outputs/reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "p1_data_audit.json"
    report_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"passed": audit["passed"], "checks": len(audit["checks"]), "inventory_gib": audit["inventory"]["gib"]}, indent=2))
    print(manifest_path)
    print(report_path)
    if not audit["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
