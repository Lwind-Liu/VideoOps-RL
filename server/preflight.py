"""Fail-fast checks before consuming H200 time."""

from __future__ import annotations

import json
import argparse
import importlib.metadata
import platform
import shutil
import inspect
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_BYTES = 4_255_140_312


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--required-gpus", type=int, default=8)
    args = parser.parse_args()
    qv_features = ROOT / "data/external/qvhighlights/features/clip_features"
    model_weight = ROOT / "models/Qwen3-VL-2B-Instruct/model.safetensors"
    required_data = [
        "data/training/sft_train_v2.jsonl",
        "data/training/sft_val_v2.jsonl",
        "data/training/grpo_train_v2.jsonl",
        "data/registry/formal_tasks_v1.jsonl",
        "data/external/qvhighlights/annotations/tasks_train_v1.jsonl",
        "data/external/qvhighlights/annotations/tasks_val_v1.jsonl",
        "data/external/qvhighlights/annotations/tasks_test_v1.jsonl",
    ]
    checks = {
        "python": sys.version.split()[0],
        "python_supported": (3, 11) <= sys.version_info[:2] < (3, 13),
        "platform": platform.platform(),
        "model_present": (ROOT / "models/Qwen3-VL-2B-Instruct/config.json").is_file(),
        "model_weight_bytes": model_weight.stat().st_size if model_weight.is_file() else 0,
        "clip_present": (ROOT / "models/clip/ViT-B-32.pt").is_file(),
        "clip_indexes": len(list((ROOT / "data/indexes/clip_vit_b32").glob("*.npz"))),
        "qv_feature_files": len(list(qv_features.glob("*.npz"))),
        "qv_query_index": (ROOT / "data/external/qvhighlights/query_embeddings_vit_b32.npz").is_file(),
        "required_data": {path: (ROOT / path).is_file() for path in required_data},
        "commands": {name: shutil.which(name) is not None for name in ("accelerate", "trl", "curl")},
        "free_gib": round(shutil.disk_usage(ROOT).free / 1024**3, 2),
    }
    try:
        import torch
        checks.update({"torch": torch.__version__, "cuda": torch.cuda.is_available(), "gpu_count": torch.cuda.device_count(), "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]})
    except Exception as error:
        checks["torch_error"] = repr(error)
    try:
        from trl import GRPOTrainer
        checks["trl"] = importlib.metadata.version("trl")
        checks["transformers"] = importlib.metadata.version("transformers")
        checks["vllm"] = importlib.metadata.version("vllm")
        checks["deepspeed"] = importlib.metadata.version("deepspeed")
        checks["versions_supported"] = (
            checks["trl"] == "1.9.2"
            and checks["transformers"] == "5.15.0"
            and checks["vllm"] == "0.25.1"
            and checks["deepspeed"] == "0.19.5"
        )
        checks["trl_environment_factory"] = "environment_factory" in inspect.signature(GRPOTrainer.__init__).parameters
    except Exception as error:
        checks["trl_error"] = repr(error)
    try:
        import jmespath
        checks["jmespath"] = jmespath.__version__
    except Exception as error:
        checks["jmespath_error"] = repr(error)
    try:
        from transformers import AutoProcessor
        processor = AutoProcessor.from_pretrained(ROOT / "models/Qwen3-VL-2B-Instruct", local_files_only=True)
        try:
            from server.tool_schemas import TOOL_SCHEMAS
        except ModuleNotFoundError:  # Direct execution: python server/preflight.py
            from tool_schemas import TOOL_SCHEMAS
        rendered = processor.apply_chat_template(
            [{"role": "user", "content": "Submit grounded evidence."}], tokenize=False,
            tools=TOOL_SCHEMAS,
        )
        checks["tool_chat_template"] = bool(rendered)
    except Exception as error:
        checks["tool_chat_template_error"] = repr(error)
    checks["required_gpus"] = args.required_gpus
    checks["passed"] = (
        checks["python_supported"]
        and checks["model_present"]
        and checks["model_weight_bytes"] == MODEL_BYTES
        and checks["clip_present"]
        and checks["clip_indexes"] >= 3
        and checks["qv_feature_files"] >= 10_000
        and checks["qv_query_index"]
        and all(checks["required_data"].values())
        and all(checks["commands"].values())
        and checks.get("cuda", False)
        and checks.get("gpu_count", 0) >= args.required_gpus
        and checks.get("versions_supported", False)
        and checks.get("trl_environment_factory", False)
        and "jmespath" in checks
        and checks.get("tool_chat_template", False)
        and checks["free_gib"] >= 20
    )
    print(json.dumps(checks, indent=2))
    if not checks["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
