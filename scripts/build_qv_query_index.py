"""Precompute official QVHighlights query embeddings for fast offline rollouts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()
    import open_clip
    import torch

    tasks = []
    source = ROOT / "data/external/qvhighlights/annotations"
    for split in ("train", "val", "test"):
        tasks.extend(read_jsonl(source / f"tasks_{split}_v1.jsonl"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    weights = ROOT / "models/clip/ViT-B-32.pt"
    model, _, _ = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained=str(weights), cache_dir=str(weights.parent), device=device, weights_only=False
    )
    model.eval()
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    chunks = []
    with torch.inference_mode():
        for start in range(0, len(tasks), args.batch_size):
            tokens = tokenizer([item["query"] for item in tasks[start:start + args.batch_size]]).to(device)
            vectors = model.encode_text(tokens)
            vectors = vectors / vectors.norm(dim=-1, keepdim=True)
            chunks.append(vectors.float().cpu().numpy().astype(np.float16))
            print(f"encoded {min(start + args.batch_size, len(tasks))}/{len(tasks)}")
    output = ROOT / "data/external/qvhighlights/query_embeddings_vit_b32.npz"
    np.savez_compressed(output, task_ids=np.asarray([item["task_id"] for item in tasks]), embeddings=np.concatenate(chunks))
    print(json.dumps({"output": str(output), "tasks": len(tasks), "shape": [len(tasks), 512], "device": device}, indent=2))


if __name__ == "__main__":
    main()
