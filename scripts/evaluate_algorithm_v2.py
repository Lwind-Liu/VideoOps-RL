"""Held-out retrieval and temporal-decoder ablations on QVHighlights."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from videoops_rl.dataset_protocol import read_jsonl
from videoops_rl.multivideo_env import temporal_iou
from videoops_rl.qv_env import decode_temporal_proposals
from videoops_rl.retrieval import load_qv_query_index


def segment(left: int, right: int) -> list[dict[str, int]]:
    return [{"start_ms": left * 2000, "end_ms": right * 2000}]


def aggregate(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float32)
    return {
        "mean_iou": round(float(array.mean()), 6),
        "r1_iou_0.3": round(float((array >= 0.3).mean()), 6),
        "r1_iou_0.5": round(float((array >= 0.5).mean()), 6),
        "r1_iou_0.7": round(float((array >= 0.7).mean()), 6),
    }


def evaluate(split: str, max_tasks: int = 0) -> dict:
    tasks = read_jsonl(ROOT / f"data/external/qvhighlights/annotations/tasks_{split}_v1.jsonl")
    if max_tasks and len(tasks) > max_tasks:
        tasks = random.Random(42).sample(tasks, max_tasks)
    queries = load_qv_query_index(str(ROOT))
    rows, missing = [], []
    for index, task in enumerate(tasks, 1):
        path = ROOT / "data/external/qvhighlights/features/clip_features" / f"{task['qv_vid']}.npz"
        if not path.is_file() or task["task_id"] not in queries:
            missing.append(task["task_id"])
            continue
        features = np.load(path)["features"].astype(np.float32)
        features /= np.maximum(np.linalg.norm(features, axis=1, keepdims=True), 1e-8)
        cosine = features @ queries[task["task_id"]]
        scores = np.clip((cosine - 0.10) / 0.25, 0.0, 1.0)
        peak = int(scores.argmax())
        proposal = decode_temporal_proposals(scores)[0]
        predictions = {
            "peak_clip": segment(peak, peak + 1),
            "fixed_10s": segment(max(0, peak - 2), min(len(scores), peak + 3)),
            "adaptive_temporal": [{"start_ms": proposal.start_ms, "end_ms": proposal.end_ms}],
        }
        rows.append({name: temporal_iou(prediction, task["target_segments"]) for name, prediction in predictions.items()})
        if index % 500 == 0:
            print(f"evaluated {split} {index}/{len(tasks)}")
    metrics = {name: aggregate([row[name] for row in rows]) for name in rows[0]} if rows else {}
    return {"split": split, "requested_tasks": len(tasks), "evaluated_tasks": len(rows), "missing": missing, "metrics": metrics}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-tasks", type=int, default=0)
    args = parser.parse_args()
    report = {
        "schema_version": "videoops.algorithm_eval.v2",
        "protocol": "one prediction per human query; R@1-style temporal IoU thresholds; not the official QVHighlights mAP implementation",
        "label_boundary": "all predictions use only CLIP video/query features; annotations are read only by the metric",
        "ablations": {
            "peak_clip": "highest-scoring 2-second clip",
            "fixed_10s": "five clips centered on the peak",
            "adaptive_temporal": "smoothed adaptive threshold, connected components, padding and proposal ranking",
        },
        "results": [evaluate(split, args.max_tasks) for split in ("val", "test")],
    }
    output = ROOT / "outputs/reports/algorithm_v2_qvhighlights_eval.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
