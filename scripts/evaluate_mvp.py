"""Evaluate fixed, single-agent, rule multi-agent, and RL Coordinator methods."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from videoops_rl.business_env import RealHighlightEnv, load_jsonl, load_tasks, normalize, temporal_iou  # noqa: E402
from videoops_rl.coordinator import CoordinatorPolicy, rollout_policy, run_sequence  # noqa: E402


def fixed_chunk_result(task, utterances, chunk_ms: int = 30_000):
    best_start, best_score = 0, -1
    for start in range(0, 480_000, chunk_ms):
        end = min(480_000, start + chunk_ms)
        text = normalize(" ".join(item["text"] for item in utterances if item["end_ms"] > start and item["start_ms"] < end))
        score = sum(len(normalize(term).split()) for term in task.search_terms if normalize(term) in text)
        if score > best_score:
            best_start, best_score = start, score
    end = min(480_000, best_start + chunk_ms)
    iou = temporal_iou(best_start, end, task.target_start_ms, task.target_end_ms)
    return {
        "predicted_start_ms": best_start,
        "predicted_end_ms": end,
        "target_start_ms": task.target_start_ms,
        "target_end_ms": task.target_end_ms,
        "temporal_iou": iou,
        "evidence_supported": False,
        "audit_passed": False,
        "success": False,
        "tool_calls": 1,
    }, [{"step": 1, "action": "uniform_30s_retrieval", "score": best_score}]


def summarize(records):
    count = len(records)
    return {
        "task_count": count,
        "mean_temporal_iou": sum(item["temporal_iou"] for item in records) / count,
        "temporal_success_rate_iou_0_5": sum(item["temporal_iou"] >= 0.5 for item in records) / count,
        "evidence_support_rate": sum(bool(item.get("evidence_supported")) for item in records) / count,
        "audit_pass_rate": sum(bool(item.get("audit_passed")) for item in records) / count,
        "business_success_rate": sum(bool(item.get("success")) for item in records) / count,
        "avg_tool_calls": sum(item["tool_calls"] for item in records) / count,
    }


def main() -> None:
    tasks = load_tasks(REPO_ROOT / "data/annotations/highlight_tasks.jsonl")
    evidence_dir = REPO_ROOT / "data/processed/tears_of_steel/s1/shots_ffmpeg_t035"
    units = load_jsonl(evidence_dir / "evidence_units.jsonl")
    utterances = load_jsonl(evidence_dir / "utterances.jsonl")
    visual = json.loads((REPO_ROOT / "data/annotations/visual_evidence.json").read_text(encoding="utf-8"))
    checkpoint_path = REPO_ROOT / "artifacts/checkpoints/coordinator_group_relative.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    policy = CoordinatorPolicy()
    policy.load_state_dict(checkpoint["state_dict"])
    policy.eval()

    trajectories_dir = REPO_ROOT / "data/trajectories/mvp"
    trajectories_dir.mkdir(parents=True, exist_ok=True)
    methods: dict[str, list[dict]] = {
        "uniform_30s": [],
        "single_agent": [],
        "rule_multi_agent": [],
        "rl_multi_agent": [],
    }
    task_records: list[dict] = []

    for task in tasks:
        fixed_result, fixed_trajectory = fixed_chunk_result(task, utterances)
        runs = {"uniform_30s": (fixed_result, fixed_trajectory)}

        env = RealHighlightEnv(units, utterances, visual, task)
        episode = run_sequence(env, ["search", "inspect", "submit"])
        runs["single_agent"] = (episode.result, episode.trajectory)

        env = RealHighlightEnv(units, utterances, visual, task)
        episode = run_sequence(env, ["search", "inspect", "audit", "submit"])
        runs["rule_multi_agent"] = (episode.result, episode.trajectory)

        env = RealHighlightEnv(units, utterances, visual, task)
        episode = rollout_policy(policy, env, torch.device("cpu"), deterministic=True)
        runs["rl_multi_agent"] = (episode.result, episode.trajectory)

        for method, (result, trajectory) in runs.items():
            record = {
                "method": method,
                "task_id": task.task_id,
                "split": task.split,
                "query": task.query,
                **result,
                "tool_calls": len(trajectory),
            }
            methods[method].append(record)
            task_records.append(record)
            (trajectories_dir / f"{method}_{task.task_id}.json").write_text(
                json.dumps({"task": task.__dict__, "trajectory": trajectory, "result": result}, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )

    report = {
        "project": "VideoOps-RL business MVP",
        "task": "query-driven highlight localization",
        "data_boundary": "10 manually reviewed tasks on one open 8-minute film; demo evidence, not benchmark generalization",
        "policy_boundary": checkpoint["training"]["boundary"],
        "methods": {name: summarize(records) for name, records in methods.items()},
        "eval_split": {name: summarize([item for item in records if item["split"] == "eval"]) for name, records in methods.items()},
        "tasks": task_records,
    }
    output_dir = REPO_ROOT / "outputs/reports"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "mvp_evaluation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (output_dir / "mvp_task_results.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=["method", "task_id", "split", "temporal_iou", "evidence_supported", "audit_passed", "success", "tool_calls", "predicted_start_ms", "predicted_end_ms"])
        writer.writeheader()
        for item in task_records:
            writer.writerow({key: item.get(key) for key in writer.fieldnames})
    print(json.dumps(report["methods"], indent=2))
    print(output_dir / "mvp_evaluation.json")


if __name__ == "__main__":
    main()
