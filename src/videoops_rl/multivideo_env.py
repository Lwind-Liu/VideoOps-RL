"""Evidence-constrained multimodal environment for SFT and Agentic GRPO."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .evidence_graph import TemporalEvidenceGraph
from .retrieval import BM25Index, clip_encoder, load_clip_index


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def temporal_iou(predicted: list[dict[str, int]], target: list[dict[str, int]]) -> float:
    points = sorted({value for segment in predicted + target for value in (segment["start_ms"], segment["end_ms"])})
    intersection = union = 0
    for start, end in zip(points, points[1:]):
        in_pred = any(segment["start_ms"] < end and segment["end_ms"] > start for segment in predicted)
        in_target = any(segment["start_ms"] < end and segment["end_ms"] > start for segment in target)
        union += (end - start) * int(in_pred or in_target)
        intersection += (end - start) * int(in_pred and in_target)
    return intersection / union if union else 0.0


@dataclass
class EpisodeState:
    searched_modalities: list[str] = field(default_factory=list)
    inspected_shot_ids: list[str] = field(default_factory=list)
    expanded_shot_ids: list[str] = field(default_factory=list)
    audited: bool = False
    audit_passed: bool = False
    audit_score: float = 0.0
    audited_shot_ids: list[str] = field(default_factory=list)
    invalid_calls: int = 0
    repeated_calls: int = 0
    tool_calls: int = 0
    process_reward: float = 0.0
    done: bool = False


class MultiVideoHighlightEnv:
    """Grounded environment with hidden labels, dense credit and hard constraints."""

    def __init__(self, repo_root: Path, task: dict[str, Any], max_steps: int = 10):
        self.repo_root, self.task = repo_root.resolve(), task
        manifest = json.loads((self.repo_root / "data/registry/formal_dataset_manifest_v1.json").read_text(encoding="utf-8"))
        video = next(item for item in manifest["videos"] if item["video_id"] == task["video_id"])
        self.duration_ms = int(video["duration_ms"])
        self.evidence_root = self.repo_root / video["evidence_root"]
        self.units = _jsonl(self.evidence_root / "evidence_units.jsonl")
        self.by_shot = {unit["shot_id"]: unit for unit in self.units}
        self.config = yaml.safe_load((self.repo_root / "configs/algorithm_v2.yaml").read_text(encoding="utf-8"))
        self.graph = TemporalEvidenceGraph(self.units, self.config["retrieval"]["fusion_weights"], self.config["graph"]["context_decay"])
        self.bm25 = BM25Index(self.units)
        all_tags = json.loads((self.repo_root / "data/registry/visual_tags_v1.json").read_text(encoding="utf-8"))
        prefix = f"{task['video_id']}:"
        tags = {key[len(prefix):]: value for key, value in all_tags.items() if key.startswith(prefix)}
        self.visual_metadata = BM25Index([{**unit, "visual_text": " ".join(tags.get(unit["shot_id"], []))} for unit in self.units], text_key="visual_text")
        self.clip_shot_ids, self.clip_embeddings = load_clip_index(self.repo_root, task["video_id"])
        self.has_dialogue = any(unit.get("has_dialogue") for unit in self.units)
        self.state, self.max_steps = EpisodeState(), max_steps
        self.trajectory: list[dict[str, Any]] = []
        self.call_signatures: set[str] = set()
        self._relevant = set(task["evidence"]["shot_ids"])

    @property
    def public_prompt(self) -> dict[str, Any]:
        return {"task_id": self.task["task_id"], "video_id": self.task["video_id"], "query": self.task["query"], "video_has_dialogue": self.has_dialogue, "tool_budget": self.max_steps, "rules": ["inspect only returned candidates", "audit only inspected shots", "submit only grounded candidates", "multiple contiguous shots are allowed"], "available_tools": ["search_transcript", "search_visual", "inspect_keyframe", "expand_context", "request_audit", "submit"]}

    def valid_tools(self) -> list[str]:
        if self.state.done:
            return []
        tools = ["search_transcript", "search_visual"]
        if self.graph.top_ids():
            tools += ["inspect_keyframe"]
        if self.state.inspected_shot_ids:
            tools += ["expand_context", "request_audit", "submit"]
        return tools

    def _record(self, tool: str, arguments: dict[str, Any], observation: dict[str, Any], hidden_delta: float = 0.0) -> dict[str, Any]:
        signature = json.dumps({"tool": tool, "arguments": arguments}, sort_keys=True)
        if signature in self.call_signatures:
            self.state.repeated_calls += 1
        self.call_signatures.add(signature)
        self.state.process_reward += hidden_delta
        self.state.tool_calls += 1
        if "error" in observation:
            self.state.invalid_calls += 1
        observation["valid_next_tools"] = self.valid_tools()
        if self.state.tool_calls >= self.max_steps and tool != "submit":
            self.state.done = True
            observation["truncated"] = True
        self.trajectory.append({"tool": tool, "arguments": arguments, "observation": observation, "hidden_process_delta": round(hidden_delta, 6)})
        return observation

    def _retrieval_delta(self, previous_rr: float) -> float:
        return self.config["process_reward"]["reciprocal_rank_gain"] * max(0.0, self.graph.reciprocal_rank(self._relevant) - previous_rr)

    def search_transcript(self, query: str, top_k: int = 5) -> dict[str, Any]:
        previous = self.graph.reciprocal_rank(self._relevant)
        scores = self.bm25.scores(query)
        self.graph.update("text", {unit["shot_id"]: float(score) for unit, score in zip(self.units, scores)})
        self.state.searched_modalities.append("text")
        hits = []
        limit = self.config["retrieval"]["candidate_limit"]
        for node in self.graph.ranked(max(1, min(int(top_k), limit))):
            item = self.graph.observation(node)
            item["transcript"] = self.by_shot[node.shot_id].get("transcript", "")
            hits.append(item)
        return self._record("search_transcript", {"query": query, "top_k": top_k}, {"agent": "TimelineScout", "hits": hits}, self._retrieval_delta(previous))

    def search_visual(self, query: str, top_k: int = 5) -> dict[str, Any]:
        previous = self.graph.reciprocal_rank(self._relevant)
        vector = clip_encoder(str(self.repo_root)).encode(query)
        raw = self.clip_embeddings @ vector
        retrieval = self.config["retrieval"]
        clip_scores = np.clip((raw - retrieval["clip_score_floor"]) / retrieval["clip_score_range"], 0.0, 1.0)
        metadata_scores = self.visual_metadata.scores(query)
        # CLIP is the primary signal; sparse media metadata is an explicit auxiliary channel.
        scores = retrieval["visual_clip_weight"] * clip_scores + retrieval["visual_metadata_weight"] * metadata_scores
        self.graph.update("image", {shot_id: float(score) for shot_id, score in zip(self.clip_shot_ids, scores)})
        self.state.searched_modalities.append("image")
        hits = [self.graph.observation(node) for node in self.graph.ranked(max(1, min(int(top_k), retrieval["candidate_limit"])))]
        return self._record("search_visual", {"query": query, "top_k": top_k}, {"agent": "VisionAnalyst", "retrieval": f"{retrieval['visual_clip_weight']}*CLIP+{retrieval['visual_metadata_weight']}*media_metadata", "hits": hits}, self._retrieval_delta(previous))

    def inspect_keyframe(self, shot_id: str) -> dict[str, Any]:
        if shot_id not in self.graph.top_ids(self.config["retrieval"]["candidate_limit"]):
            return self._record("inspect_keyframe", {"shot_id": shot_id}, {"error": "shot_not_in_retrieved_candidates"}, self.config["process_reward"]["invalid_action"])
        unit, node = self.by_shot[shot_id], self.graph.nodes[shot_id]
        self.graph.inspect(shot_id)
        if shot_id not in self.state.inspected_shot_ids:
            self.state.inspected_shot_ids.append(shot_id)
        path = (self.evidence_root / unit["keyframe_path"]).resolve()
        delta = self.config["process_reward"]["inspect_correct"] if shot_id in self._relevant else self.config["process_reward"]["inspect_wrong"]
        return self._record("inspect_keyframe", {"shot_id": shot_id}, {"agent": "VisionAnalyst", **self.graph.observation(node), "keyframe_path": path.relative_to(self.repo_root).as_posix(), "transcript": unit.get("transcript", "")}, delta)

    def expand_context(self, shot_id: str, radius: int = 1) -> dict[str, Any]:
        if shot_id not in self.state.inspected_shot_ids:
            return self._record("expand_context", {"shot_id": shot_id, "radius": radius}, {"error": "inspect_before_expand"}, self.config["process_reward"]["invalid_action"])
        before = len(set(self.state.expanded_shot_ids) & self._relevant) / max(1, len(self._relevant))
        selected = self.graph.propagate(shot_id, max(1, min(int(radius), 3)))
        self.state.expanded_shot_ids = sorted(set(self.state.expanded_shot_ids + selected), key=self.graph.order.index)
        after = len(set(self.state.expanded_shot_ids) & self._relevant) / max(1, len(self._relevant))
        nodes = [self.graph.observation(self.graph.nodes[item]) for item in selected]
        return self._record("expand_context", {"shot_id": shot_id, "radius": radius}, {"agent": "TimelineScout", "neighbors": nodes}, self.config["process_reward"]["context_recall_gain"] * max(0.0, after - before))

    def _modality_coverage(self) -> float:
        required = set(self.task["evidence"]["required_modalities"])
        observed = set(self.state.searched_modalities)
        if self.state.inspected_shot_ids:
            observed.add("image")
        return len(required & observed) / max(1, len(required))

    def request_audit(self, shot_ids: list[str]) -> dict[str, Any]:
        valid = bool(shot_ids) and all(shot_id in self.graph.nodes for shot_id in shot_ids)
        inspected = valid and all(shot_id in self.state.inspected_shot_ids for shot_id in shot_ids)
        scores = [self.graph.nodes[shot_id].fused_score for shot_id in shot_ids if shot_id in self.graph.nodes]
        semantic = float(sum(scores) / max(1, len(scores)))
        ordered = sorted((self.graph.order.index(shot_id) for shot_id in shot_ids if shot_id in self.graph.nodes))
        continuity = 1.0 if not ordered or all(right - left <= 1 for left, right in zip(ordered, ordered[1:])) else 0.0
        modality = self._modality_coverage()
        audit_config = self.config["audit"]
        audit_score = audit_config["semantic_weight"] * semantic + audit_config["modality_weight"] * modality + audit_config["continuity_weight"] * continuity
        passed = valid and inspected and modality == 1.0 and audit_score >= audit_config["pass_threshold"]
        self.state.audited, self.state.audit_passed, self.state.audit_score = True, passed, audit_score
        self.state.audited_shot_ids = list(shot_ids) if passed else []
        hidden_correct = bool(set(shot_ids) & self._relevant)
        delta = self.config["process_reward"]["correct_audit"] if passed and hidden_correct else (self.config["process_reward"]["false_positive_audit"] if passed and not hidden_correct else 0.0)
        return self._record("request_audit", {"shot_ids": shot_ids}, {"agent": "EvidenceAuditor", "passed": passed, "audit_score": round(audit_score, 4), "semantic_consistency": round(semantic, 4), "modality_coverage": modality, "temporal_continuity": continuity, "inspected": inspected}, delta)

    def submit(self, shot_ids: list[str]) -> dict[str, Any]:
        grounded = set(self.graph.top_ids(self.config["retrieval"]["candidate_limit"])) | set(self.state.expanded_shot_ids)
        valid = list(dict.fromkeys(shot_id for shot_id in shot_ids if shot_id in grounded and shot_id in self.graph.nodes))
        if not valid or not set(valid).issubset(set(self.state.inspected_shot_ids) | set(self.state.expanded_shot_ids)):
            self.state.done = True
            result = {"shot_ids": valid, "predicted_segments": [], "temporal_iou": 0.0, "evidence_precision": 0.0, "shot_f1": 0.0, "evidence_supported": False, "audit_passed": False, "success": False, "reward_parts": {"invalid_submission": -1.0}, "reward": -1.0}
            return self._record("submit", {"shot_ids": shot_ids}, result, self.config["process_reward"]["invalid_submission"])
        predicted = [{"start_ms": self.by_shot[item]["start_ms"], "end_ms": self.by_shot[item]["end_ms"]} for item in valid]
        iou = temporal_iou(predicted, self.task["target_segments"])
        relevant, selected = self._relevant, set(valid)
        precision = len(selected & relevant) / max(1, len(selected))
        recall = len(selected & relevant) / max(1, len(relevant))
        shot_f1 = 2 * precision * recall / max(1e-8, precision + recall)
        evidence_supported = all(item in self.state.inspected_shot_ids or item in self.state.expanded_shot_ids for item in valid)
        audit_applies = self.state.audit_passed and selected.issubset(set(self.state.audited_shot_ids))
        reward = self.config["terminal_reward"]
        parts = {"temporal": reward["temporal_iou"] * iou, "shot_set": reward["shot_f1"] * shot_f1, "semantic_audit": reward["semantic_audit"] * self.state.audit_score if audit_applies else 0.0, "modality": reward["modality_coverage"] * self._modality_coverage(), "process": self.state.process_reward, "tool_cost": -reward["tool_cost"] * self.state.tool_calls, "repeat_cost": -reward["repeated_call_cost"] * self.state.repeated_calls, "invalid_cost": -reward["invalid_call_cost"] * self.state.invalid_calls}
        result = {"shot_ids": valid, "predicted_segments": predicted, "temporal_iou": round(iou, 6), "evidence_precision": round(precision, 6), "shot_f1": round(shot_f1, 6), "evidence_supported": evidence_supported, "audit_passed": audit_applies, "audit_score": round(self.state.audit_score, 6), "success": iou >= 0.5 and evidence_supported and audit_applies, "reward_parts": parts, "reward": round(sum(parts.values()), 6)}
        self.state.done = True
        return self._record("submit", {"shot_ids": shot_ids}, result)

    def export_episode(self) -> dict[str, Any]:
        return {"prompt": self.public_prompt, "trajectory": self.trajectory, "state": asdict(self.state)}
