"""Temporal evidence graph and query-conditioned candidate belief state."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


@dataclass
class CandidateNode:
    shot_id: str
    start_ms: int
    end_ms: int
    text_score: float = 0.0
    visual_score: float = 0.0
    context_score: float = 0.0
    inspected: bool = False
    sources: set[str] = field(default_factory=set)
    fusion_weights: tuple[float, float, float] = (0.52, 0.38, 0.10)

    @property
    def fused_score(self) -> float:
        # Noisy-OR rewards agreement without allowing one modality to erase another.
        components = tuple(weight * score for weight, score in zip(self.fusion_weights, (self.text_score, self.visual_score, self.context_score)))
        return 1.0 - math.prod(1.0 - max(0.0, min(1.0, value)) for value in components)


class TemporalEvidenceGraph:
    def __init__(self, units: list[dict[str, Any]], fusion_weights: dict[str, float] | None = None, context_decay: float = 0.7):
        weights = fusion_weights or {"text": 0.52, "image": 0.38, "context": 0.10}
        weight_tuple = (weights["text"], weights["image"], weights["context"])
        self.context_decay = context_decay
        self.order = [unit["shot_id"] for unit in units]
        self.nodes = {unit["shot_id"]: CandidateNode(unit["shot_id"], unit["start_ms"], unit["end_ms"], fusion_weights=weight_tuple) for unit in units}

    def update(self, modality: str, scores: dict[str, float]) -> None:
        attribute = {"text": "text_score", "image": "visual_score"}[modality]
        for shot_id, score in scores.items():
            if shot_id not in self.nodes:
                continue
            node = self.nodes[shot_id]
            setattr(node, attribute, max(getattr(node, attribute), float(score)))
            if score > 0:
                node.sources.add(modality)

    def inspect(self, shot_id: str) -> None:
        self.nodes[shot_id].inspected = True

    def propagate(self, center_shot: str, radius: int) -> list[str]:
        center = self.order.index(center_shot)
        selected = self.order[max(0, center - radius):center + radius + 1]
        center_score = max(0.1, self.nodes[center_shot].fused_score)
        for shot_id in selected:
            distance = abs(self.order.index(shot_id) - center)
            self.nodes[shot_id].context_score = max(self.nodes[shot_id].context_score, center_score * math.exp(-self.context_decay * distance))
            self.nodes[shot_id].sources.add("context")
        return selected

    def ranked(self, limit: int | None = None) -> list[CandidateNode]:
        values = [node for node in self.nodes.values() if node.sources]
        values.sort(key=lambda node: (-node.fused_score, node.start_ms))
        return values[:limit] if limit else values

    def reciprocal_rank(self, relevant: set[str]) -> float:
        for rank, node in enumerate(self.ranked(), start=1):
            if node.shot_id in relevant:
                return 1.0 / rank
        return 0.0

    def top_ids(self, limit: int = 8) -> list[str]:
        return [node.shot_id for node in self.ranked(limit)]

    def observation(self, node: CandidateNode, include_scores: bool = True) -> dict[str, Any]:
        item = {"shot_id": node.shot_id, "start_ms": node.start_ms, "end_ms": node.end_ms, "sources": sorted(node.sources)}
        if include_scores:
            item.update({"text_score": round(node.text_score, 4), "visual_score": round(node.visual_score, 4), "context_score": round(node.context_score, 4), "fused_score": round(node.fused_score, 4)})
        return item
