"""Versioned dataset protocol and offline-package audit for VideoOps-RL."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from jsonschema import Draft202012Validator

GIB = 1024**3
HARD_LIMIT_BYTES = 50 * GIB
TARGET_LIMIT_BYTES = 45 * GIB
PRODUCTION_SPLITS = {"train", "val", "test"}
EXCLUDED_DIRS = {".git", "__pycache__", "tmp"}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest().upper()


def safe_repo_path(repo_root: Path, relative_path: str) -> Path:
    candidate = (repo_root / relative_path).resolve()
    root = repo_root.resolve()
    if candidate != root and root not in candidate.parents:
        raise ValueError(f"path escapes repository: {relative_path}")
    return candidate


def iter_package_files(repo_root: Path) -> Iterable[Path]:
    for path in repo_root.rglob("*"):
        if path.is_file() and not any(part in EXCLUDED_DIRS for part in path.relative_to(repo_root).parts):
            yield path


def package_inventory(repo_root: Path) -> dict[str, Any]:
    categories: dict[str, int] = defaultdict(int)
    total = 0
    file_count = 0
    for path in iter_package_files(repo_root):
        size = path.stat().st_size
        relative = path.relative_to(repo_root)
        first = relative.parts[0] if len(relative.parts) > 1 else "project_root"
        categories[first] += size
        total += size
        file_count += 1
    return {
        "bytes": total,
        "gib": round(total / GIB, 4),
        "file_count": file_count,
        "by_top_level": {
            key: {"bytes": value, "gib": round(value / GIB, 4)}
            for key, value in sorted(categories.items(), key=lambda item: item[1], reverse=True)
        },
    }


def audit_task_records(tasks: list[dict[str, Any]], video_durations: dict[str, int]) -> list[dict[str, Any]]:
    seen_task_ids: set[str] = set()
    video_splits: dict[str, set[str]] = defaultdict(set)
    errors: list[str] = []
    for task in tasks:
        task_id, video_id, split = task.get("task_id"), task.get("video_id"), task.get("split")
        if task_id in seen_task_ids:
            errors.append(f"duplicate task_id: {task_id}")
        seen_task_ids.add(task_id)
        if video_id not in video_durations:
            errors.append(f"{task_id}: unknown video_id {video_id}")
            continue
        if split not in {"smoke", *PRODUCTION_SPLITS}:
            errors.append(f"{task_id}: invalid split {split}")
        if split in PRODUCTION_SPLITS:
            video_splits[video_id].add(split)
        if not str(task.get("query", "")).strip():
            errors.append(f"{task_id}: empty query")
        for segment in task.get("target_segments", []):
            start, end = segment.get("start_ms"), segment.get("end_ms")
            if not isinstance(start, int) or not isinstance(end, int) or not (0 <= start < end <= video_durations[video_id]):
                errors.append(f"{task_id}: invalid target [{start}, {end})")
        if not task.get("evidence", {}).get("shot_ids"):
            errors.append(f"{task_id}: no grounded shot evidence")
    leaking = {video_id: sorted(splits) for video_id, splits in video_splits.items() if len(splits) > 1}
    if leaking:
        errors.append(f"video-level split leakage: {leaking}")
    return [
        {"name": "task_schema_and_ranges", "passed": not errors, "errors": errors},
        {"name": "video_level_split_isolation", "passed": not leaking, "leaking_videos": leaking},
    ]


def audit_manifest(manifest: dict[str, Any], repo_root: Path, verify_hashes: bool = True) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    manifest_schema = json.loads((repo_root / "schemas/videoops_dataset_manifest_v1.schema.json").read_text(encoding="utf-8"))
    manifest_schema_errors = [error.message for error in Draft202012Validator(manifest_schema).iter_errors(manifest)]
    checks.append({"name": "manifest_json_schema", "passed": not manifest_schema_errors, "errors": manifest_schema_errors})
    inventory = package_inventory(repo_root)
    hard_limit = int(manifest["capacity"]["hard_limit_bytes"])
    checks.append({
        "name": "offline_package_capacity", "passed": inventory["bytes"] <= hard_limit,
        "actual_bytes": inventory["bytes"], "hard_limit_bytes": hard_limit,
        "remaining_bytes": hard_limit - inventory["bytes"],
    })
    path_errors: list[str] = []
    hash_errors: list[str] = []
    durations: dict[str, int] = {}
    for video in manifest["videos"]:
        durations[video["video_id"]] = int(video["duration_ms"])
        for item in video["artifacts"]:
            try:
                path = safe_repo_path(repo_root, item["path"])
            except ValueError as error:
                path_errors.append(str(error))
                continue
            if not path.is_file():
                path_errors.append(f"missing: {item['path']}")
                continue
            if path.stat().st_size != item["bytes"]:
                path_errors.append(f"size mismatch: {item['path']}")
            if verify_hashes and sha256_file(path) != item["sha256"]:
                hash_errors.append(f"sha256 mismatch: {item['path']}")
    checks.append({"name": "artifact_paths_and_sizes", "passed": not path_errors, "errors": path_errors})
    checks.append({"name": "artifact_hashes", "passed": not hash_errors, "errors": hash_errors})
    task_path = safe_repo_path(repo_root, manifest["tasks"]["path"])
    tasks = read_jsonl(task_path)
    task_schema = json.loads(safe_repo_path(repo_root, manifest["tasks"]["schema"]).read_text(encoding="utf-8"))
    task_validator = Draft202012Validator(task_schema)
    task_schema_errors = [
        f"{task.get('task_id', f'line_{index}')}: {error.message}"
        for index, task in enumerate(tasks, start=1)
        for error in task_validator.iter_errors(task)
    ]
    checks.append({"name": "task_json_schema", "passed": not task_schema_errors, "errors": task_schema_errors})
    checks.extend(audit_task_records(tasks, durations))
    expected = int(manifest["tasks"]["count"])
    checks.append({"name": "task_count", "passed": len(tasks) == expected, "expected": expected, "actual": len(tasks)})
    return {
        "schema_version": "videoops.data_audit.v1", "dataset_id": manifest["dataset_id"],
        "passed": all(check["passed"] for check in checks), "checks": checks, "inventory": inventory,
    }
