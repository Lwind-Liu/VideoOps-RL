"""Evaluate fixed, single-agent and multi-agent policies before GPU training."""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from videoops_rl.agents import fixed_chunk_baseline, multi_agent_expert, single_agent
from videoops_rl.dataset_protocol import read_jsonl
from videoops_rl.multivideo_env import MultiVideoHighlightEnv


def main() -> None:
    tasks = read_jsonl(ROOT / "data/registry/formal_tasks_v1.jsonl")
    policies = {"fixed_chunk": fixed_chunk_baseline, "single_agent": single_agent, "multi_agent": multi_agent_expert}
    rows, trajectories = [], []
    for task in tasks:
        for name, policy in policies.items():
            env = MultiVideoHighlightEnv(ROOT, task)
            result = policy(env)
            row = {"policy": name, "split": task["split"], "task_id": task["task_id"], "task_type": task["task_type"], "iou": result["temporal_iou"], "success": int(result["success"]), "evidence": int(result["evidence_supported"]), "audit": int(result["audit_passed"]), "tool_calls": env.state.tool_calls, "reward": result["reward"]}
            rows.append(row)
            trajectories.append({"policy": name, "task": task, "episode": env.export_episode(), "result": result})
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["policy"], row["split"])].append(row)
    summary = []
    for (policy, split), values in sorted(grouped.items()):
        summary.append({"policy": policy, "split": split, "tasks": len(values), "mean_iou": round(sum(item["iou"] for item in values) / len(values), 4), "success_rate": round(sum(item["success"] for item in values) / len(values), 4), "evidence_rate": round(sum(item["evidence"] for item in values) / len(values), 4), "audit_rate": round(sum(item["audit"] for item in values) / len(values), 4), "mean_tool_calls": round(sum(item["tool_calls"] for item in values) / len(values), 2), "mean_reward": round(sum(item["reward"] for item in values) / len(values), 4)})
    out = ROOT / "outputs/reports"
    out.mkdir(parents=True, exist_ok=True)
    (out / "pretraining_stack_eval_v1.json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=2), encoding="utf-8")
    (out / "expert_trajectories_v1.jsonl").write_text("\n".join(json.dumps(item, ensure_ascii=False) for item in trajectories if item["policy"] == "multi_agent") + "\n", encoding="utf-8")
    with (out / "pretraining_stack_eval_v1.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

