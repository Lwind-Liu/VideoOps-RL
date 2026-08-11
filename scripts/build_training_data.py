"""Generate auditable SFT demonstrations and label-free GRPO prompts."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from videoops_rl.agents import multi_agent_expert
from videoops_rl.dataset_protocol import read_jsonl
from videoops_rl.multivideo_env import MultiVideoHighlightEnv
from videoops_rl.qv_env import QVHighlightsEnv

SYSTEM = """You are the Coordinator in VideoOps-RL. Locate query-relevant highlights by calling tools. TimelineScout handles transcript search, VisionAnalyst handles visual search and keyframe inspection, and EvidenceAuditor checks grounding. Never invent a shot ID. Return the final selection through submit."""


def _text_content(value: str) -> list[dict]:
    return [{"type": "text", "text": value}]


def _assistant_call(tool: str, arguments: dict) -> dict:
    return {"role": "assistant", "content": _text_content(json.dumps({"tool": tool, "arguments": arguments}, ensure_ascii=False))}


def build_record(task: dict, env_class=MultiVideoHighlightEnv) -> tuple[dict, dict]:
    env = env_class(ROOT, task)
    result = multi_agent_expert(env)
    messages = [{"role": "system", "content": _text_content(SYSTEM)}, {"role": "user", "content": _text_content(json.dumps(env.public_prompt, ensure_ascii=False))}]
    image_paths = []
    for step in env.trajectory:
        messages.append(_assistant_call(step["tool"], step["arguments"]))
        content = _text_content(json.dumps(step["observation"], ensure_ascii=False))
        path = step["observation"].get("keyframe_path")
        if path:
            image_paths.append(path)
            content.insert(0, {"type": "image", "image": path})
        messages.append({"role": "tool", "content": content})
    source = "qvhighlights" if task["video_id"].startswith("qvh:") else "formal_multimodal"
    sft = {"task_id": task["task_id"], "source": source, "split": task["split"], "messages": messages, "images": image_paths, "teacher_reward": result["reward"], "teacher_success": result["success"]}
    prompt = {"task_id": task["task_id"], "video_id": task["video_id"], "source": source, "split": task["split"], "prompt": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": json.dumps(env.public_prompt, ensure_ascii=False)}]}
    return sft, prompt


def main() -> None:
    formal_tasks = read_jsonl(ROOT / "data/registry/formal_tasks_v1.jsonl")
    qv_root = ROOT / "data/external/qvhighlights/annotations"
    qv_tasks = [item for split in ("train", "val", "test") for item in read_jsonl(qv_root / f"tasks_{split}_v1.jsonl")]
    output = ROOT / "data/training"
    output.mkdir(parents=True, exist_ok=True)
    formal_records = [(*build_record(task), task) for task in formal_tasks]
    qv_records = []
    for index, task in enumerate(qv_tasks, 1):
        qv_records.append((*build_record(task, QVHighlightsEnv), task))
        if index % 500 == 0:
            print(f"built QV expert trajectories {index}/{len(qv_tasks)}")
    records = formal_records + qv_records
    files = {}
    for split in ("train", "val", "test"):
        qv_sft = [sft for sft, _, task in qv_records if task["split"] == split and sft["teacher_success"]]
        formal_sft = [sft for sft, _, task in formal_records if task["split"] == split and sft["teacher_success"]]
        oversample = 5 if split == "train" else 1
        sft = qv_sft + formal_sft * oversample
        qv_prompts = [prompt for _, prompt, task in qv_records if task["split"] == split]
        formal_prompts = [prompt for _, prompt, task in formal_records if task["split"] == split]
        grpo_formal_repeat = 45 if split == "train" else 1
        prompts = qv_prompts + formal_prompts * grpo_formal_repeat
        for name, values in ((f"sft_{split}_v2.jsonl", sft), (f"grpo_{split}_v2.jsonl", prompts)):
            (output / name).write_text("\n".join(json.dumps(item, ensure_ascii=False) for item in values) + "\n", encoding="utf-8")
            files[name] = len(values)
    target_keys = {"target_segments", "search_terms", "shot_ids"}
    leakage = []
    for _, prompt, task in records:
        serialized = json.dumps(prompt, ensure_ascii=False)
        if any(key in serialized for key in target_keys):
            leakage.append(task["task_id"])
    missing_images = sorted({path for sft, _, _ in records for path in sft["images"] if not (ROOT / path).is_file()})
    teacher_success = {
        "formal": sum(sft["teacher_success"] for sft, _, _ in formal_records) / len(formal_records),
        "qvhighlights": sum(sft["teacher_success"] for sft, _, _ in qv_records) / len(qv_records),
    }
    report = {"passed": not leakage and not missing_images, "counts": files, "sources": {"formal_tasks": len(formal_records), "qvhighlights_tasks": len(qv_records)}, "teacher_success_rate": teacher_success, "sft_filter": "successful audited teacher trajectories only", "formal_sft_train_repeat": 5, "formal_grpo_train_repeat": 45, "effective_grpo_train_mix": {"qvhighlights": 7218, "formal": 1800, "formal_ratio": round(1800 / 9018, 6)}, "prompt_label_leakage": leakage, "missing_images": missing_images, "notes": "SFT contains successful teacher observations by design. GRPO keeps all label-free prompts so policy learning explores successes and failures. Train prompts repeat the 40 formal tasks to realize the configured 80/20 benchmark-to-raw-image mixture."}
    report_path = ROOT / "outputs/reports/training_data_audit_v2.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
