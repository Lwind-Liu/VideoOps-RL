"""Convert official QVHighlights human annotations into VideoOps task records."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def convert(item: dict, split: str) -> dict:
    relevant = sorted(set(int(index) for index in item.get("relevant_clip_ids", [])))
    windows = [{"start_ms": int(round(start * 1000)), "end_ms": int(round(end * 1000))} for start, end in item["relevant_windows"]]
    terms = [token for token in re.findall(r"[a-z0-9']+", item["query"].lower()) if len(token) > 2][:12]
    return {"schema_version": "videoops.task.v1", "task_id": f"qvh_{item['qid']}", "video_id": f"qvh:{item['vid']}", "split": split, "query": item["query"], "target_segments": windows, "evidence": {"search_terms": terms, "shot_ids": [f"clip_{index:04d}" for index in relevant], "required_modalities": ["video"]}, "task_type": "qvhighlights_human_query", "difficulty": "multi_moment" if len(windows) > 1 else "single_moment", "duration_ms": int(item["duration"] * 1000), "qv_vid": item["vid"], "saliency_scores": item.get("saliency_scores", [])}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=ROOT / "external/moment_detr/data")
    args = parser.parse_args()
    mapping = {"train": "highlight_train_release.jsonl", "val": "highlight_val_release.jsonl", "test": "highlight_test_with_gt.jsonl"}
    output = ROOT / "data/external/qvhighlights"
    annotations = output / "annotations"
    annotations.mkdir(parents=True, exist_ok=True)
    all_tasks, summary, split_videos = [], {}, {}
    for split, filename in mapping.items():
        rows = read_jsonl(args.source / filename)
        tasks = [convert(item, split) for item in rows if item.get("relevant_windows")]
        path = annotations / f"tasks_{split}_v1.jsonl"
        path.write_text("\n".join(json.dumps(task, ensure_ascii=False) for task in tasks) + "\n", encoding="utf-8")
        all_tasks.extend(tasks)
        split_videos[split] = {task["video_id"] for task in tasks}
        summary[split] = {"tasks": len(tasks), "videos": len(split_videos[split]), "windows": sum(len(task["target_segments"]) for task in tasks)}
    leakage = {f"{left}-{right}": len(split_videos[left] & split_videos[right]) for left, right in (("train", "val"), ("train", "test"), ("val", "test"))}
    manifest = {"schema_version": "videoops.qvhighlights.v1", "dataset": "QVHighlights", "license": "CC BY-NC-SA 4.0 annotations", "source": "https://github.com/jayleicn/moment_detr", "clip_seconds": 2, "summary": summary, "video_level_leakage": leakage, "total_tasks": len(all_tasks), "total_videos": len({task["video_id"] for task in all_tasks}), "feature_root": "data/external/qvhighlights/features/clip_features", "feature_type": "official public OpenAI CLIP 2-second video features", "source_archive_bytes": 11842201723, "source_archive_sha256": "88152E47F7B744BE2E9C3A5F1BE4572BAFD9B41C482DB1C0FCC7E28853AE97A9", "claim_boundary": "human query and temporal annotations with public pre-extracted features; raw videos are not redistributed in this package"}
    (output / "qvhighlights_manifest_v1.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (output / "LICENSE").write_text((args.source / "LICENSE").read_text(encoding="utf-8"), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    if any(leakage.values()):
        raise SystemExit("official split video leakage detected")


if __name__ == "__main__":
    main()
