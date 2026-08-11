"""Stage 1: LoRA SFT on audited expert tool trajectories."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=str(ROOT / "models/Qwen3-VL-2B-Instruct"))
    parser.add_argument("--output-dir", default=str(ROOT / "artifacts/sft_qwen3vl2b"))
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    args = parser.parse_args()
    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import AutoProcessor
    from trl import SFTConfig, SFTTrainer

    dataset = load_dataset("json", data_files={"train": str(ROOT / "data/training/sft_train_v2.jsonl"), "validation": str(ROOT / "data/training/sft_val_v2.jsonl")})
    config = SFTConfig(
        output_dir=args.output_dir, num_train_epochs=args.epochs, learning_rate=2e-5,
        per_device_train_batch_size=1, gradient_accumulation_steps=args.gradient_accumulation_steps, bf16=True,
        gradient_checkpointing=True, logging_steps=1, save_strategy="epoch",
        eval_strategy="epoch", report_to="none", max_length=None,
    )
    peft = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules="all-linear", task_type="CAUSAL_LM")
    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
    trainer = SFTTrainer(model=args.model, args=config, train_dataset=dataset["train"], eval_dataset=dataset["validation"], peft_config=peft, processing_class=processor)
    trainer.train()
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
