"""BM25 + CLIP retrieval primitives shared by agents and the RL environment."""

from __future__ import annotations

import math
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from .training_data_quality import validate_feature_matrix


def tokens(text: str) -> list[str]:
    return [token for token in re.findall(r"[a-z0-9']+", text.lower()) if token not in {"a", "an", "and", "at", "by", "find", "in", "is", "it", "of", "on", "scene", "the", "to", "when", "where", "with"}]


class BM25Index:
    def __init__(self, records: list[dict[str, Any]], text_key: str = "transcript"):
        self.records = records
        self.docs = [tokens(record.get(text_key, "")) for record in records]
        self.counts = [Counter(doc) for doc in self.docs]
        self.avgdl = sum(map(len, self.docs)) / max(1, len(self.docs))
        df = Counter(token for doc in self.docs for token in set(doc))
        self.idf = {term: math.log(1 + (len(self.docs) - freq + 0.5) / (freq + 0.5)) for term, freq in df.items()}

    def scores(self, query: str, k1: float = 1.5, b: float = 0.75) -> np.ndarray:
        result = np.zeros(len(self.records), dtype=np.float32)
        for index, (doc, counts) in enumerate(zip(self.docs, self.counts)):
            for term in tokens(query):
                frequency = counts.get(term, 0)
                if not frequency:
                    continue
                denominator = frequency + k1 * (1 - b + b * len(doc) / max(self.avgdl, 1e-6))
                result[index] += self.idf.get(term, 0.0) * frequency * (k1 + 1) / denominator
        maximum = float(result.max(initial=0.0))
        return result / maximum if maximum else result


class CLIPTextEncoder:
    """Lazy singleton CPU text encoder; image embeddings are precomputed."""

    def __init__(self, repo_root: Path):
        import open_clip
        weights = repo_root / "models/clip/ViT-B-32.pt"
        self.model, _, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained=str(weights), cache_dir=str(repo_root / "models/clip"), device="cpu", weights_only=False)
        self.model.eval()
        self.tokenizer = open_clip.get_tokenizer("ViT-B-32")

    def encode(self, query: str) -> np.ndarray:
        import torch
        with torch.inference_mode():
            vector = self.model.encode_text(self.tokenizer([query]))
            vector = vector / vector.norm(dim=-1, keepdim=True)
        return vector[0].float().cpu().numpy()


@lru_cache(maxsize=2)
def clip_encoder(repo_root: str) -> CLIPTextEncoder:
    return CLIPTextEncoder(Path(repo_root))


def load_clip_index(repo_root: Path, video_id: str) -> tuple[list[str], np.ndarray]:
    path = repo_root / "data/indexes/clip_vit_b32" / f"{video_id}.npz"
    data = np.load(path, allow_pickle=False)
    shot_ids, embeddings = data["shot_ids"].tolist(), data["embeddings"]
    validate_feature_matrix(embeddings, path.as_posix())
    if len(shot_ids) != len(embeddings) or len(shot_ids) != len(set(shot_ids)):
        raise ValueError(f"invalid shot ID index in {path}")
    return shot_ids, embeddings.astype(np.float32)


@lru_cache(maxsize=1)
def load_qv_query_index(repo_root: str) -> dict[str, np.ndarray]:
    path = Path(repo_root) / "data/external/qvhighlights/query_embeddings_vit_b32.npz"
    if not path.is_file():
        return {}
    data = np.load(path, allow_pickle=False)
    task_ids, embeddings = data["task_ids"].tolist(), data["embeddings"]
    validate_feature_matrix(embeddings, path.as_posix())
    if len(task_ids) != len(embeddings) or len(task_ids) != len(set(task_ids)):
        raise ValueError(f"invalid task ID index in {path}")
    return {str(task_id): vector.astype(np.float32) for task_id, vector in zip(task_ids, embeddings)}
