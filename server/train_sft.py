"""Stage 1: LoRA SFT on audited expert tool trajectories."""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path

try:
    from server.tool_schemas import TOOL_SCHEMAS, wrap_offline_observation
    from server.metric_logging import build_metric_callback
except ModuleNotFoundError:  # Direct execution: python server/train_sft.py
    from tool_schemas import TOOL_SCHEMAS, wrap_offline_observation
    from metric_logging import build_metric_callback

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


def _text_content(content: object) -> str | None:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                return block.get("text")
    return None


def _replace_text_content(content: object, text: str) -> object:
    if isinstance(content, str):
        return text
    if isinstance(content, list):
        replaced = False
        blocks = []
        for block in content:
            current = copy.deepcopy(block)
            if isinstance(current, dict) and current.get("type") == "text" and not replaced:
                current["text"] = text
                replaced = True
            blocks.append(current)
        if not replaced:
            blocks.append({"type": "text", "text": text})
        return blocks
    return text


def normalize_tool_record(record: dict) -> dict:
    """Convert legacy JSON action text into TRL's native tool-calling format."""
    normalized = copy.deepcopy(record)
    messages = []
    pending_tool: str | None = None
    pending_arguments: dict = {}
    native_calls = 0
    for message in normalized["messages"]:
        current = copy.deepcopy(message)
        if current.get("role") == "assistant":
            text = _text_content(current.get("content"))
            try:
                call = json.loads(text) if text else None
            except json.JSONDecodeError:
                call = None
            if isinstance(call, dict) and isinstance(call.get("tool"), str) and isinstance(call.get("arguments"), dict):
                pending_tool = call["tool"]
                pending_arguments = call["arguments"]
                current = {
                    "role": "assistant",
                    "tool_calls": [{
                        "type": "function",
                        "function": {"name": pending_tool, "arguments": call["arguments"]},
                    }],
                }
                native_calls += 1
        elif current.get("role") == "tool":
            if pending_tool is None:
                raise ValueError(f"tool response without assistant call in {record.get('task_id')}")
            current["name"] = pending_tool
            text = _text_content(current.get("content"))
            try:
                observation = json.loads(text) if text else None
            except json.JSONDecodeError:
                observation = None
            if isinstance(observation, dict) and "observation" not in observation:
                wrapped = wrap_offline_observation(
                    pending_tool,
                    pending_arguments,
                    observation,
                    str(record.get("task_id", "unknown")),
                    native_calls,
                )
                current["content"] = _replace_text_content(
                    current.get("content"), json.dumps(wrapped, ensure_ascii=False)
                )
            pending_tool = None
            pending_arguments = {}
        messages.append(current)
    if native_calls == 0:
        raise ValueError(f"no tool calls found in {record.get('task_id')}")
    normalized["messages"] = messages
    normalized["tools"] = copy.deepcopy(TOOL_SCHEMAS)
    return normalized


def load_sft_records(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [normalize_tool_record(json.loads(line)) for line in handle if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=str(ROOT / "models/Qwen3-VL-2B-Instruct"))
    parser.add_argument("--output-dir", default=str(ROOT / "artifacts/sft_qwen3vl2b"))
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    args = parser.parse_args()
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoProcessor
    from trl import SFTConfig, SFTTrainer

    dataset = {
        "train": Dataset.from_list(load_sft_records(ROOT / "data/training/sft_train_v2.jsonl"), on_mixed_types="use_json"),
        "validation": Dataset.from_list(load_sft_records(ROOT / "data/training/sft_val_v2.jsonl"), on_mixed_types="use_json"),
    }
    config = SFTConfig(
        output_dir=args.output_dir, num_train_epochs=args.epochs, learning_rate=2e-5,
        per_device_train_batch_size=1, gradient_accumulation_steps=args.gradient_accumulation_steps, bf16=True,
        gradient_checkpointing=True, logging_steps=1, save_strategy="epoch",
        eval_strategy="epoch", report_to="none", max_length=None,
    )
    peft = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules="all-linear", task_type="CAUSAL_LM")
    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
    trainer = SFTTrainer(
        model=args.model,
        args=config,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        peft_config=peft,
        processing_class=processor,
        callbacks=[build_metric_callback("sft", ROOT / "outputs")],
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    trainer.save_state()
    processor.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
