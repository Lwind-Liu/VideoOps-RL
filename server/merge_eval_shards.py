"""Merge deterministic multi-GPU checkpoint-evaluation shards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def aggregate(rows: list[dict]) -> dict:
    count = max(1, len(rows))
    return {
        "tasks": len(rows),
        "mean_iou": sum(row["iou"] for row in rows) / count,
        "success_rate": sum(row["success"] for row in rows) / count,
        "mean_reward": sum(row["reward"] for row in rows) / count,
        "mean_tool_calls": sum(row["tool_calls"] for row in rows) / count,
        "invalid_call_rate": sum(row.get("invalid_calls", 0) for row in rows) / max(1, sum(row["tool_calls"] for row in rows)),
        "repeated_call_rate": sum(row.get("repeated_calls", 0) for row in rows) / max(1, sum(row["tool_calls"] for row in rows)),
        "audit_pass_rate": sum(row.get("audit_passed", False) for row in rows) / count,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["formal", "qvhighlights", "all"], default="all")
    parser.add_argument("--split", choices=["val", "test"], required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    args = parser.parse_args()
    reports = []
    for index in range(args.num_shards):
        path = ROOT / f"outputs/reports/checkpoint_{args.dataset}_{args.split}_eval_shard{index:02d}-of-{args.num_shards:02d}.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        reports.append(json.loads(path.read_text(encoding="utf-8")))
    rows = [row for report in reports for row in report["rows"]]
    task_ids = [row["task_id"] for row in rows]
    if len(task_ids) != len(set(task_ids)):
        raise RuntimeError("duplicate task IDs across evaluation shards")
    result = {
        "model": reports[0]["model"], "dataset": args.dataset, "split": args.split,
        "shards": args.num_shards, **aggregate(rows),
        "by_source": {source: aggregate([row for row in rows if row["source"] == source]) for source in sorted({row["source"] for row in rows})},
        "parse_error_rate": sum(row["parse_errors"] for row in rows) / max(1, sum(row["tool_calls"] + row["parse_errors"] for row in rows)),
        "rows": rows,
    }
    output = ROOT / f"outputs/reports/checkpoint_{args.dataset}_{args.split}_eval.json"
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()
