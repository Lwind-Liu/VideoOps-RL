"""Merge the SFT LoRA adapter into the local base VLM for vLLM and GRPO."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default=str(ROOT / "models/Qwen3-VL-2B-Instruct"))
    parser.add_argument("--adapter", default=str(ROOT / "artifacts/sft_qwen3vl2b"))
    parser.add_argument("--output", default=str(ROOT / "artifacts/sft_qwen3vl2b_merged"))
    args = parser.parse_args()
    import torch
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText, AutoProcessor

    base = AutoModelForImageTextToText.from_pretrained(args.base, torch_dtype=torch.bfloat16, device_map="auto", local_files_only=True)
    merged = PeftModel.from_pretrained(base, args.adapter, local_files_only=True).merge_and_unload()
    merged.save_pretrained(args.output, safe_serialization=True, max_shard_size="5GB")
    AutoProcessor.from_pretrained(args.base, local_files_only=True).save_pretrained(args.output)
    print(f"merged model written to {args.output}")


if __name__ == "__main__":
    main()
