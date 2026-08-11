"""Greedy held-out evaluation for a trained tool-using checkpoint."""

from __future__ import annotations

import argparse
import json
import os
import re
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from videoops_rl.dataset_protocol import read_jsonl
from videoops_rl.multivideo_env import MultiVideoHighlightEnv
from videoops_rl.qv_env import QVHighlightsEnv

SYSTEM = "Use one tool at a time. Answer only JSON: {\"tool\": NAME, \"arguments\": OBJECT}. Finish with submit."


def parse_call(text: str) -> dict | None:
    matches = re.findall(r"\{.*\}", text, flags=re.DOTALL)
    for candidate in reversed(matches):
        try:
            call = json.loads(candidate)
            if isinstance(call, dict) and "tool" in call and isinstance(call.get("arguments"), dict):
                return call
        except json.JSONDecodeError:
            continue
    return None


def dispatch(env: MultiVideoHighlightEnv | QVHighlightsEnv, call: dict) -> dict:
    allowed = {name: getattr(env, name) for name in ("search_transcript", "search_visual", "inspect_keyframe", "expand_context", "request_audit", "submit")}
    if call["tool"] not in allowed:
        env.state.invalid_calls += 1
        return {"error": "unknown_tool"}
    try:
        return allowed[call["tool"]](**call["arguments"])
    except (TypeError, ValueError) as error:
        env.state.invalid_calls += 1
        return {"error": "invalid_arguments", "detail": str(error)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=str(ROOT / "artifacts/grpo_qwen3vl2b"))
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--dataset", choices=["formal", "qvhighlights", "all"], default="all")
    parser.add_argument("--max-tasks", type=int, default=0, help="Zero evaluates the complete selected split.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    args = parser.parse_args()
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForImageTextToText.from_pretrained(args.model, torch_dtype=torch.bfloat16, device_map="auto", local_files_only=True).eval()
    formal = [item for item in read_jsonl(ROOT / "data/registry/formal_tasks_v1.jsonl") if item["split"] == args.split]
    qv = read_jsonl(ROOT / f"data/external/qvhighlights/annotations/tasks_{args.split}_v1.jsonl")
    tasks = formal if args.dataset == "formal" else qv if args.dataset == "qvhighlights" else formal + qv
    if args.max_tasks and len(tasks) > args.max_tasks:
        tasks = random.Random(args.seed).sample(tasks, args.max_tasks)
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("shard-index must be in [0, num-shards)")
    tasks = tasks[args.shard_index::args.num_shards]
    rows = []
    for task in tasks:
        source = "qvhighlights" if task["video_id"].startswith("qvh:") else "formal"
        env = QVHighlightsEnv(ROOT, task) if source == "qvhighlights" else MultiVideoHighlightEnv(ROOT, task)
        messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": json.dumps(env.public_prompt)}]
        parse_errors = 0
        for _ in range(env.max_steps):
            inputs = processor.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True,
                return_dict=True, return_tensors="pt",
            ).to(model.device)
            with torch.inference_mode():
                generated = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
            text = processor.batch_decode(generated[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]
            call = parse_call(text)
            if call is None:
                parse_errors += 1
                messages.extend([{"role": "assistant", "content": text}, {"role": "tool", "content": "Invalid JSON. Try one valid tool call."}])
                continue
            observation = dispatch(env, call)
            tool_content: str | list[dict] = json.dumps(observation)
            if observation.get("keyframe_path"):
                tool_content = [{"type": "image", "image": str(ROOT / observation["keyframe_path"])}, {"type": "text", "text": json.dumps(observation)}]
            messages.extend([{"role": "assistant", "content": json.dumps(call)}, {"role": "tool", "content": tool_content}])
            if env.state.done:
                break
        result = next((step["observation"] for step in reversed(env.trajectory) if step["tool"] == "submit"), {"temporal_iou": 0.0, "success": False, "reward": -1.0})
        rows.append({"task_id": task["task_id"], "source": source, "iou": result["temporal_iou"], "success": result["success"], "reward": result["reward"], "parse_errors": parse_errors, "tool_calls": env.state.tool_calls})
    def aggregate(items: list[dict]) -> dict:
        return {"tasks": len(items), "mean_iou": sum(row["iou"] for row in items) / max(1, len(items)), "success_rate": sum(row["success"] for row in items) / max(1, len(items)), "mean_reward": sum(row["reward"] for row in items) / max(1, len(items))}
    report = {"model": args.model, "split": args.split, "dataset": args.dataset, "sampling": {"max_tasks": args.max_tasks, "seed": args.seed, "complete_split": args.max_tasks == 0, "num_shards": args.num_shards, "shard_index": args.shard_index}, **aggregate(rows), "by_source": {source: aggregate([row for row in rows if row["source"] == source]) for source in sorted({row["source"] for row in rows})}, "parse_error_rate": sum(row["parse_errors"] for row in rows) / max(1, sum(row["tool_calls"] + row["parse_errors"] for row in rows)), "rows": rows}
    suffix = f"_shard{args.shard_index:02d}-of-{args.num_shards:02d}" if args.num_shards > 1 else ""
    path = ROOT / f"outputs/reports/checkpoint_{args.dataset}_{args.split}_eval{suffix}.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()
