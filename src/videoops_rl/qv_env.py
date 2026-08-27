"""Feature-grounded QVHighlights environment with temporal proposal decoding."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .evidence_graph import TemporalEvidenceGraph
from .multivideo_env import EpisodeState, MultiVideoHighlightEnv
from .retrieval import clip_encoder, load_qv_query_index
from .training_data_quality import validate_feature_matrix


@dataclass(frozen=True)
class TemporalProposal:
    proposal_id: str
    shot_ids: list[str]
    start_ms: int
    end_ms: int
    score: float
    anchor_shot_id: str


def decode_temporal_proposals(scores: np.ndarray, clip_ms: int = 2000, limit: int = 3) -> list[TemporalProposal]:
    """Decode dense similarities into padded contiguous moment proposals.

    A short smoothing kernel suppresses isolated peaks. The adaptive threshold
    combines a video-relative quantile with a peak-relative floor, then connected
    components become variable-length candidate moments.
    """
    values = np.asarray(scores, dtype=np.float32)
    if not len(values):
        return []
    smooth = np.convolve(values, np.ones(5, dtype=np.float32) / 5.0, mode="same")
    threshold = max(float(np.quantile(smooth, 0.72)), 0.52 * float(smooth.max()))
    active = smooth >= threshold
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for index, flag in enumerate(active.tolist() + [False]):
        if flag and start is None:
            start = index
        elif not flag and start is not None:
            runs.append((max(0, start - 2), min(len(values), index + 2)))
            start = None
    if not runs:
        peak = int(smooth.argmax())
        runs = [(max(0, peak - 2), min(len(values), peak + 3))]
    merged: list[tuple[int, int]] = []
    for left, right in runs:
        if merged and left <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], right))
        else:
            merged.append((left, right))
    ranked = sorted(
        merged,
        key=lambda span: 0.55 * float(smooth[slice(*span)].mean()) + 0.45 * float(smooth[slice(*span)].max()),
        reverse=True,
    )[:limit]
    proposals = []
    for rank, (left, right) in enumerate(ranked, 1):
        local = smooth[left:right]
        anchor = left + int(local.argmax())
        proposals.append(TemporalProposal(
            proposal_id=f"proposal_{rank:02d}",
            shot_ids=[f"clip_{index:04d}" for index in range(left, right)],
            start_ms=left * clip_ms,
            end_ms=right * clip_ms,
            score=round(0.55 * float(local.mean()) + 0.45 * float(local.max()), 6),
            anchor_shot_id=f"clip_{anchor:04d}",
        ))
    return proposals


class QVHighlightsEnv(MultiVideoHighlightEnv):
    """Agentic RL environment over official 2-second QVHighlights CLIP features."""

    def __init__(self, repo_root: Path, task: dict[str, Any], max_steps: int = 10):
        self.repo_root, self.task = repo_root.resolve(), task
        self.config = yaml.safe_load((self.repo_root / "configs/algorithm_v2.yaml").read_text(encoding="utf-8"))
        feature_path = self.repo_root / "data/external/qvhighlights/features/clip_features" / f"{task['qv_vid']}.npz"
        if not feature_path.is_file():
            raise FileNotFoundError(f"QVHighlights feature missing: {feature_path}")
        features = np.load(feature_path, allow_pickle=False)["features"]
        validate_feature_matrix(features, feature_path.as_posix())
        features = features.astype(np.float32)
        features /= np.maximum(np.linalg.norm(features, axis=1, keepdims=True), 1e-8)
        self.features = features
        clip_ms = int(self.config["qvhighlights"]["clip_seconds"] * 1000)
        self.duration_ms = int(task.get("duration_ms", len(features) * clip_ms))
        self.units = [
            {"shot_id": f"clip_{index:04d}", "start_ms": index * clip_ms,
             "end_ms": min((index + 1) * clip_ms, self.duration_ms), "transcript": "", "has_dialogue": False}
            for index in range(len(features))
        ]
        self.by_shot = {unit["shot_id"]: unit for unit in self.units}
        weights = dict(self.config["retrieval"]["fusion_weights"])
        weights["text"] = 0.0
        weights["image"] += self.config["retrieval"]["fusion_weights"]["text"]
        self.graph = TemporalEvidenceGraph(self.units, weights, self.config["graph"]["context_decay"])
        self.has_dialogue = False
        self.state, self.max_steps = EpisodeState(), max_steps
        self.trajectory: list[dict[str, Any]] = []
        self.call_signatures: set[str] = set()
        self._relevant = set(task["evidence"]["shot_ids"])
        self.proposals: list[TemporalProposal] = []

    @property
    def public_prompt(self) -> dict[str, Any]:
        return {
            "task_id": self.task["task_id"], "video_id": self.task["video_id"], "query": self.task["query"],
            "video_has_dialogue": False, "tool_budget": self.max_steps,
            "evidence_type": "precomputed_2s_video_features",
            "rules": ["search before inspection", "inspect a proposal anchor", "expand radius 0 to accept its adaptive temporal proposal", "audit only inspected or expanded evidence", "submit only grounded clips"],
            "available_tools": ["search_visual", "inspect_keyframe", "expand_context", "request_audit", "submit"],
        }

    def valid_tools(self) -> list[str]:
        if self.state.done:
            return []
        tools = ["search_visual"]
        if self.graph.top_ids():
            tools.append("inspect_keyframe")
        if self.state.inspected_shot_ids:
            tools.extend(["expand_context", "request_audit", "submit"])
        return tools

    def search_transcript(self, query: str, top_k: int = 5) -> dict[str, Any]:
        return self._record("search_transcript", {"query": query, "top_k": top_k}, {"error": "transcript_unavailable_for_feature_benchmark"}, self.config["process_reward"]["invalid_action"])

    def search_visual(self, query: str, top_k: int = 5) -> dict[str, Any]:
        previous = self.graph.reciprocal_rank(self._relevant)
        cached = load_qv_query_index(str(self.repo_root)).get(self.task["task_id"])
        vector = cached if cached is not None and query.strip() == self.task["query"].strip() else clip_encoder(str(self.repo_root)).encode(query)
        cosine = self.features @ vector
        qv = self.config["qvhighlights"]
        scores = np.clip((cosine - qv["feature_score_floor"]) / qv["feature_score_range"], 0.0, 1.0)
        self.graph.update("image", {unit["shot_id"]: float(score) for unit, score in zip(self.units, scores)})
        self.state.searched_modalities.append("image")
        self.proposals = decode_temporal_proposals(scores, int(qv["clip_seconds"] * 1000))
        limit = min(max(1, int(top_k)), int(qv["candidate_limit"]))
        hits = [self.graph.observation(node) for node in self.graph.ranked(limit)]
        proposals = [proposal.__dict__ for proposal in self.proposals]
        return self._record("search_visual", {"query": query, "top_k": top_k}, {
            "agent": "VisionAnalyst", "retrieval": "normalized OpenAI CLIP cosine + adaptive temporal decoder",
            "query_embedding": "offline_cache" if cached is not None and query.strip() == self.task["query"].strip() else "online_encoder",
            "hits": hits, "temporal_proposals": proposals,
        }, self._retrieval_delta(previous))

    def inspect_keyframe(self, shot_id: str) -> dict[str, Any]:
        candidate_ids = set(self.graph.top_ids(int(self.config["qvhighlights"]["candidate_limit"])))
        candidate_ids.update(proposal.anchor_shot_id for proposal in self.proposals)
        if shot_id not in candidate_ids:
            return self._record("inspect_keyframe", {"shot_id": shot_id}, {"error": "shot_not_in_retrieved_candidates"}, self.config["process_reward"]["invalid_action"])
        self.graph.inspect(shot_id)
        if shot_id not in self.state.inspected_shot_ids:
            self.state.inspected_shot_ids.append(shot_id)
        node = self.graph.nodes[shot_id]
        delta = self.config["process_reward"]["inspect_correct"] if shot_id in self._relevant else self.config["process_reward"]["inspect_wrong"]
        return self._record("inspect_keyframe", {"shot_id": shot_id}, {
            "agent": "VisionAnalyst", **self.graph.observation(node), "feature_only": True,
            "feature_norm": round(float(np.linalg.norm(self.features[self.graph.order.index(shot_id)])), 4),
            "note": "Raw video is not redistributed; inspection is grounded in the official clip feature.",
        }, delta)

    def expand_context(self, shot_id: str, radius: int = 0) -> dict[str, Any]:
        if shot_id not in self.state.inspected_shot_ids:
            return self._record("expand_context", {"shot_id": shot_id, "radius": radius}, {"error": "inspect_before_expand"}, self.config["process_reward"]["invalid_action"])
        before = len(set(self.state.expanded_shot_ids) & self._relevant) / max(1, len(self._relevant))
        proposal = next((item for item in self.proposals if item.anchor_shot_id == shot_id), None)
        if int(radius) == 0 and proposal is not None:
            selected = proposal.shot_ids
            mode = "adaptive_temporal_proposal"
        else:
            selected = self.graph.propagate(shot_id, max(1, min(int(radius), 20)))
            mode = "fixed_radius"
        self.state.expanded_shot_ids = sorted(set(self.state.expanded_shot_ids + selected), key=self.graph.order.index)
        after = len(set(self.state.expanded_shot_ids) & self._relevant) / max(1, len(self._relevant))
        return self._record("expand_context", {"shot_id": shot_id, "radius": radius}, {
            "agent": "TimelineScout", "mode": mode, "shot_ids": selected,
            "segment": {"start_ms": self.by_shot[selected[0]]["start_ms"], "end_ms": self.by_shot[selected[-1]]["end_ms"]},
        }, self.config["process_reward"]["context_recall_gain"] * max(0.0, after - before))

    def _modality_coverage(self) -> float:
        return float("image" in self.state.searched_modalities and bool(self.state.inspected_shot_ids))

    def request_audit(self, shot_ids: list[str]) -> dict[str, Any]:
        grounded = set(self.state.expanded_shot_ids) | set(self.state.inspected_shot_ids)
        valid = bool(shot_ids) and set(shot_ids).issubset(grounded)
        inspected_anchor = any(item in self.state.inspected_shot_ids for item in shot_ids)
        scores = [self.graph.nodes[item].fused_score for item in shot_ids if item in self.graph.nodes]
        semantic = float(np.mean(scores)) if scores else 0.0
        indices = sorted(self.graph.order.index(item) for item in shot_ids if item in self.graph.nodes)
        continuity = float(bool(indices) and all(right - left == 1 for left, right in zip(indices, indices[1:])))
        modality = self._modality_coverage()
        cfg = self.config["audit"]
        audit_score = cfg["semantic_weight"] * semantic + cfg["modality_weight"] * modality + cfg["continuity_weight"] * continuity
        passed = valid and inspected_anchor and modality == 1.0 and audit_score >= cfg["pass_threshold"]
        self.state.audited, self.state.audit_passed, self.state.audit_score = True, passed, audit_score
        self.state.audited_shot_ids = list(shot_ids) if passed else []
        hidden_correct = bool(set(shot_ids) & self._relevant)
        delta = self.config["process_reward"]["correct_audit"] if passed and hidden_correct else (self.config["process_reward"]["false_positive_audit"] if passed else 0.0)
        return self._record("request_audit", {"shot_ids": shot_ids}, {
            "agent": "EvidenceAuditor", "passed": passed, "audit_score": round(audit_score, 4),
            "semantic_consistency": round(semantic, 4), "modality_coverage": modality,
            "temporal_continuity": continuity, "inspected_anchor": inspected_anchor,
        }, delta)

    def submit(self, shot_ids: list[str]) -> dict[str, Any]:
        result = super().submit(shot_ids)
        if "invalid_submission" in result.get("reward_parts", {}):
            return result
        saliency_by_shot = {
            shot_id: float(np.mean(scores)) / 4.0
            for shot_id, scores in zip(self.task["evidence"]["shot_ids"], self.task.get("saliency_scores", []))
        }
        saliency = float(np.mean([saliency_by_shot.get(item, 0.0) for item in result["shot_ids"]])) if result["shot_ids"] else 0.0
        bonus = self.config["qvhighlights"]["saliency_reward_weight"] * saliency
        result["saliency_quality"] = round(saliency, 6)
        result["reward_parts"]["saliency"] = bonus
        result["reward"] = round(float(result["reward"]) + bonus, 6)
        return result
