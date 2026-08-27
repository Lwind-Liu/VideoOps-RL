"""Summarize optimization metrics and agent trajectories into one run report."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev

ROOT = Path(__file__).resolve().parents[1]


def read_jsonl(paths: list[Path]) -> list[dict]:
    rows: list[dict] = []
    for path in paths:
        if not path.is_file():
            continue
        with path.open(encoding="utf-8") as handle:
            rows.extend(json.loads(line) for line in handle if line.strip())
    return rows


def metric_summary(rows: list[dict]) -> dict:
    series: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        for key, value in row.get("metrics", {}).items():
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                series[key].append(float(value))
    return {
        key: {"count": len(values), "first": values[0], "last": values[-1], "min": min(values), "max": max(values)}
        for key, values in sorted(series.items())
    }


def trace_summary(rows: list[dict], group_size: int) -> dict:
    episodes: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        episodes[row["episode_id"]].append(row)
    completed = []
    for episode_id, calls in episodes.items():
        calls.sort(key=lambda item: item["request_id"])
        submit = next((item for item in reversed(calls) if item["tool"] == "submit"), None)
        observation = {} if submit is None else submit["observation"]
        completed.append({
            "episode_id": episode_id,
            "task_id": calls[0]["task_id"],
            "tool_calls": len(calls),
            "invalid_calls": sum(item["status"] != "ok" for item in calls),
            "submitted": submit is not None,
            "success": bool(observation.get("success", False)),
            "audit_passed": bool(observation.get("audit_passed", False)),
            "reward": observation.get("reward"),
            "reward_parts": observation.get("reward_parts", {}),
            "cost_units": sum(float(item["cost_units"]) for item in calls),
            "latency_ms": sum(float(item["latency_ms"]) for item in calls),
            "tools": [item["tool"] for item in calls],
        })
    reward_values = [float(item["reward"]) for item in completed if item["reward"] is not None]
    task_rewards: dict[str, list[float]] = defaultdict(list)
    for item in completed:
        if item["reward"] is not None:
            task_rewards[item["task_id"]].append(float(item["reward"]))
    approximate_groups = []
    for task_id, values in task_rewards.items():
        for start in range(0, len(values), group_size):
            group = values[start:start + group_size]
            if len(group) == group_size:
                approximate_groups.append({"task_id": task_id, "reward_std": pstdev(group)})
    part_names = sorted({key for item in completed for key in item["reward_parts"]})
    return {
        "calls": len(rows),
        "episodes": len(completed),
        "submitted_rate": mean([item["submitted"] for item in completed]) if completed else 0.0,
        "success_rate": mean([item["success"] for item in completed]) if completed else 0.0,
        "audit_pass_rate": mean([item["audit_passed"] for item in completed]) if completed else 0.0,
        "mean_tool_calls": mean([item["tool_calls"] for item in completed]) if completed else 0.0,
        "invalid_call_rate": sum(item["invalid_calls"] for item in completed) / max(1, len(rows)),
        "mean_reward": mean(reward_values) if reward_values else None,
        "reward_std": pstdev(reward_values) if len(reward_values) > 1 else 0.0,
        "mean_cost_units": mean([item["cost_units"] for item in completed]) if completed else 0.0,
        "mean_latency_ms": mean([item["latency_ms"] for item in completed]) if completed else 0.0,
        "reward_parts_mean": {
            name: mean([float(item["reward_parts"].get(name, 0.0)) for item in completed])
            for name in part_names
        },
        "approximate_grouping_note": "Grouped by task and arrival order; trainer reward_std is authoritative for exact update groups.",
        "approximate_group_reward_std_mean": mean([item["reward_std"] for item in approximate_groups]) if approximate_groups else None,
        "approximate_zero_advantage_group_rate": mean([item["reward_std"] == 0.0 for item in approximate_groups]) if approximate_groups else None,
        "episodes_detail": completed,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--run-id", default="manual")
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--require-traces", action="store_true")
    args = parser.parse_args()
    metric_rows = read_jsonl(sorted((ROOT / "outputs/metrics").glob(f"*_{args.run_id}_metrics.jsonl")))
    trace_rows = read_jsonl(sorted((ROOT / "outputs/traces" / args.run_id).glob("tool_trace_*.jsonl")))
    traces = trace_summary(trace_rows, args.group_size)
    trainer_metrics = metric_summary(metric_rows)
    exact_group_keys = [
        key for key in trainer_metrics
        if "reward_std" in key or "zero_std" in key or ("zero" in key and "adv" in key)
    ]
    exact_zero_keys = [key for key in exact_group_keys if "zero_std" in key or ("zero" in key and "adv" in key)]
    exact_zero_gate = not exact_zero_keys or max(trainer_metrics[key]["last"] for key in exact_zero_keys) < 0.95
    essential_pass = bool(trace_rows) and traces["submitted_rate"] > 0.0 and traces["reward_std"] > 0.0 and exact_zero_gate
    report = {
        "run_mode": args.run_mode,
        "run_id": args.run_id,
        "passed": essential_pass if args.require_traces else True,
        "gates": {
            "has_tool_traces": bool(trace_rows),
            "has_submission": traces["submitted_rate"] > 0.0,
            "reward_has_variance": traces["reward_std"] > 0.0,
            "exact_zero_advantage_rate_below_95pct_when_available": exact_zero_gate,
        },
        "trainer_metrics": trainer_metrics,
        "exact_group_metric_keys": exact_group_keys,
        "tool_trajectory_summary": traces,
    }
    output = ROOT / f"outputs/reports/training_signal_{args.run_id}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    concise_traces = {key: value for key, value in traces.items() if key != "episodes_detail"}
    print(json.dumps({"passed": report["passed"], "gates": report["gates"], "exact_group_metric_keys": exact_group_keys, "tool_trajectory_summary": concise_traces}, indent=2, ensure_ascii=False))
    if args.require_traces and not report["passed"]:
        raise SystemExit("training-signal gate failed; inspect the generated report before full training")


if __name__ == "__main__":
    main()
