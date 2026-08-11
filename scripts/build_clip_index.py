"""Build a real CLIP image index over all keyframes in the formal corpus."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from videoops_rl.dataset_protocol import read_jsonl, sha256_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="ViT-B-32")
    parser.add_argument("--pretrained", default="openai")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    import open_clip

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cache = ROOT / "models/clip"
    cache.mkdir(parents=True, exist_ok=True)
    weights_path = cache / "ViT-B-32.pt"
    if not weights_path.is_file():
        raise FileNotFoundError(f"download OpenAI CLIP weights first: {weights_path}")
    model, _, preprocess = open_clip.create_model_and_transforms(args.model, pretrained=str(weights_path), cache_dir=str(cache), weights_only=False)
    model = model.to(device).eval()
    manifest = json.loads((ROOT / "data/registry/formal_dataset_manifest_v1.json").read_text(encoding="utf-8"))
    out = ROOT / "data/indexes/clip_vit_b32"
    out.mkdir(parents=True, exist_ok=True)
    report = {"schema_version": "videoops.clip_index.v1", "model": args.model, "pretrained": args.pretrained, "device": device, "videos": []}
    for video in manifest["videos"]:
        evidence_root = ROOT / video["evidence_root"]
        units = read_jsonl(evidence_root / "evidence_units.jsonl")
        vectors = []
        with torch.inference_mode():
            for start in range(0, len(units), args.batch_size):
                batch = units[start:start + args.batch_size]
                images = torch.stack([preprocess(Image.open(evidence_root / unit["keyframe_path"]).convert("RGB")) for unit in batch]).to(device)
                features = model.encode_image(images)
                features = features / features.norm(dim=-1, keepdim=True)
                vectors.append(features.float().cpu().numpy())
        matrix = np.concatenate(vectors).astype(np.float16)
        path = out / f"{video['video_id']}.npz"
        np.savez_compressed(path, shot_ids=np.array([unit["shot_id"] for unit in units]), embeddings=matrix)
        report["videos"].append({"video_id": video["video_id"], "shots": len(units), "dimensions": int(matrix.shape[1]), "path": path.relative_to(ROOT).as_posix(), "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    weights = [weights_path]
    report["weights"] = [{"path": path.relative_to(ROOT).as_posix(), "bytes": path.stat().st_size, "sha256": sha256_file(path)} for path in weights]
    report_path = ROOT / "outputs/reports/clip_index_audit_v1.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
