"""Build and verify the <=50 GiB offline server handoff archive."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HARD_LIMIT = 50 * 1024**3
CODE_DIRS = ("configs", "docs", "schemas", "scripts", "server", "src", "tests", "data/registry", "data/training", "data/indexes", "data/external/qvhighlights", "models/Qwen3-VL-2B-Instruct", "models/clip")
ROOT_FILES = ("README.md", "pyproject.toml", ".gitignore", "videoops_rl_notes.tex", "videoops_rl_notes.pdf")
REPORTS = ("formal_data_audit_v1.json", "training_data_audit_v1.json", "training_data_audit_v2.json", "pretraining_stack_eval_v1.json", "pretraining_stack_eval_v1.csv", "algorithm_v2_qvhighlights_eval.json", "clip_index_audit_v1.json", "expert_trajectories_v1.jsonl")
SKIP_PARTS = {"__pycache__", ".cache", ".git", "dist"}
SKIP_SUFFIXES = {".pyc", ".aux", ".log", ".out", ".aria2", ".lock", ".incomplete"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(4 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest().upper()


def collect() -> list[Path]:
    selected: set[Path] = set()
    for directory in CODE_DIRS:
        root = ROOT / directory
        if root.exists():
            selected.update(path for path in root.rglob("*") if path.is_file())
    for name in ROOT_FILES:
        if (ROOT / name).is_file():
            selected.add(ROOT / name)
    for name in REPORTS:
        if (ROOT / "outputs/reports" / name).is_file():
            selected.add(ROOT / "outputs/reports" / name)
    manifest = json.loads((ROOT / "data/registry/formal_dataset_manifest_v1.json").read_text(encoding="utf-8"))
    for video in manifest["videos"]:
        evidence_root = ROOT / video["evidence_root"]
        selected.update(path for path in evidence_root.rglob("*") if path.is_file())
        for artifact in video["artifacts"]:
            selected.add(ROOT / artifact["path"])
    return sorted(path for path in selected if not (set(path.relative_to(ROOT).parts) & SKIP_PARTS) and path.suffix.lower() not in SKIP_SUFFIXES)


def validate_model() -> dict:
    model = ROOT / "models/Qwen3-VL-2B-Instruct/model.safetensors"
    expected = 4_255_140_312
    if not model.is_file() or model.stat().st_size != expected:
        raise RuntimeError(f"model is incomplete: expected {expected} bytes at {model}")
    with model.open("rb") as handle:
        header_length = int.from_bytes(handle.read(8), "little")
        header = json.loads(handle.read(header_length))
    return {"path": model.relative_to(ROOT).as_posix(), "bytes": model.stat().st_size, "tensors": len(header) - int("__metadata__" in header), "sha256": sha256(model)}


def validate_algorithm_assets() -> dict:
    feature_root = ROOT / "data/external/qvhighlights/features/clip_features"
    features = list(feature_root.glob("*.npz"))
    query_index = ROOT / "data/external/qvhighlights/query_embeddings_vit_b32.npz"
    clip = ROOT / "models/clip/ViT-B-32.pt"
    training = ROOT / "data/training/sft_train_v2.jsonl"
    if len(features) < 10_000 or not query_index.is_file() or not clip.is_file() or not training.is_file():
        raise RuntimeError("algorithm v2 assets are incomplete")
    return {"qv_feature_files": len(features), "query_index_bytes": query_index.stat().st_size, "clip_bytes": clip.stat().st_size, "sft_train_bytes": training.stat().st_size}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "dist/VideoOps-RL-offline-server.zip")
    args = parser.parse_args()
    model = validate_model()
    algorithm_assets = validate_algorithm_assets()
    files = collect()
    total = sum(path.stat().st_size for path in files)
    if total > HARD_LIMIT:
        raise RuntimeError(f"uncompressed package exceeds 50 GiB: {total}")
    manifest = {"package": "VideoOps-RL-offline-server-v2", "policy": "formal videos/keyframes, QVHighlights annotations/CLIP features, model weights, code, configs and reports included; no server-side downloads", "uncompressed_bytes": total, "hard_limit_bytes": HARD_LIMIT, "model": model, "algorithm_assets": algorithm_assets, "files": [{"path": path.relative_to(ROOT).as_posix(), "bytes": path.stat().st_size, "sha256": sha256(path)} for path in files]}
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", allowZip64=True) as archive:
        for path in files:
            relative = path.relative_to(ROOT).as_posix()
            compress = zipfile.ZIP_STORED if path.suffix.lower() in {".mp4", ".mov", ".jpg", ".png", ".safetensors", ".pt", ".npz"} else zipfile.ZIP_DEFLATED
            archive.write(path, f"VideoOps-RL/{relative}", compress_type=compress)
        archive.writestr("VideoOps-RL/PACKAGE_MANIFEST.json", json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", compress_type=zipfile.ZIP_DEFLATED)
    with zipfile.ZipFile(output) as archive:
        bad = archive.testzip()
        names = archive.namelist()
    if bad or "VideoOps-RL/models/Qwen3-VL-2B-Instruct/model.safetensors" not in names:
        raise RuntimeError(f"archive verification failed: {bad}")
    report = {"passed": True, "output": str(output), "zip_bytes": output.stat().st_size, "zip_gib": round(output.stat().st_size / 1024**3, 3), "uncompressed_gib": round(total / 1024**3, 3), "file_count": len(files), "model": model, "algorithm_assets": algorithm_assets}
    (ROOT / "outputs/reports/offline_package_audit_v1.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
