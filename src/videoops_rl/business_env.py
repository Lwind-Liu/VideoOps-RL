"""Runnable query-based highlight environment over real S1 evidence units."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


ACTIONS = ("search", "inspect", "expand", "audit", "submit")


@dataclass(frozen=True)
class HighlightTask:
    task_id: str
    query: str
    search_terms: tuple[str, ...]
    target_start_ms: int
    target_end_ms: int
    split: str


@dataclass
class HighlightState:
    searched: bool = False
    inspected: bool = False
    expanded: bool = False
    audited: bool = False
    audit_passed: bool = False
    candidate_start_ms: int | None = None
    candidate_end_ms: int | None = None
    candidate_shot_ids: list[str] = field(default_factory=list)
    confidence: float = 0.0
    inspected_shot_id: str | None = None
    inspected_keyframe: str | None = None
    visual_tags: list[str] = field(default_factory=list)
    transcript_evidence: list[str] = field(default_factory=list)
    tool_calls: int = 0
    done: bool = False


def temporal_iou(
    first_start: int, first_end: int, second_start: int, second_end: int
) -> float:
    intersection = max(0, min(first_end, second_end) - max(first_start, second_start))
    union = max(first_end, second_end) - min(first_start, second_start)
    return intersection / union if union > 0 else 0.0


def load_tasks(path: Path) -> list[HighlightTask]:
    tasks = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        item["search_terms"] = tuple(item["search_terms"])
        tasks.append(HighlightTask(**item))
    return tasks


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def normalize(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9']+", text.lower()))


class RealHighlightEnv:
    """Tool environment whose observations come from real processed video artifacts."""

    def __init__(
        self,
        evidence_units: list[dict[str, Any]],
        utterances: list[dict[str, Any]],
        visual_annotations: dict[str, list[str]],
        task: HighlightTask,
        max_steps: int = 7,
    ):
        self.units = evidence_units
        self.utterances = utterances
        self.by_shot = {unit["shot_id"]: unit for unit in evidence_units}
        self.shot_order = [unit["shot_id"] for unit in evidence_units]
        self.visual_annotations = visual_annotations
        self.task = task
        self.max_steps = max_steps
        self.state = HighlightState()
        self.trajectory: list[dict[str, Any]] = []

    def valid_action_mask(self) -> list[bool]:
        s = self.state
        if s.done:
            return [False] * len(ACTIONS)
        return [
            not s.searched,
            s.searched and not s.inspected,
            s.searched and not s.expanded,
            s.inspected and not s.audited,
            s.searched,
        ]

    def features(self) -> list[float]:
        s = self.state
        span = (
            (s.candidate_end_ms - s.candidate_start_ms) / 60_000
            if s.candidate_start_ms is not None and s.candidate_end_ms is not None
            else 0.0
        )
        remaining = max(0, self.max_steps - s.tool_calls) / self.max_steps
        return [
            1.0,
            float(s.searched),
            float(s.inspected),
            float(s.expanded),
            float(s.audited),
            float(s.audit_passed),
            s.confidence,
            min(span, 2.0),
            remaining,
        ]

    def _search(self) -> dict[str, Any]:
        scored: list[tuple[int, float, dict[str, Any], list[str]]] = []
        normalized_terms = [(term, normalize(term)) for term in self.task.search_terms]
        for index, utterance in enumerate(self.utterances):
            transcript = normalize(utterance["text"])
            matched = [original for original, term in normalized_terms if term and term in transcript]
            if matched:
                score = sum(max(1, len(normalize(term).split())) for term in matched)
                scored.append((index, float(score), utterance, matched))

        clusters: list[list[tuple[int, float, dict[str, Any], list[str]]]] = []
        for item in scored:
            if not clusters or item[2]["start_ms"] - clusters[-1][-1][2]["end_ms"] > 15_000:
                clusters.append([item])
            else:
                clusters[-1].append(item)
        if not clusters:
            return {"hits": 0}
        cluster = max(clusters, key=lambda values: (sum(item[1] for item in values), -values[0][0]))
        unique_terms = {term for item in cluster for term in item[3]}
        candidate_start = max(0, min(item[2]["start_ms"] for item in cluster) - 2_000)
        candidate_end = min(480_000, max(item[2]["end_ms"] for item in cluster) + 2_000)
        self.state.candidate_start_ms = candidate_start
        self.state.candidate_end_ms = candidate_end
        self.state.candidate_shot_ids = [
            unit["shot_id"]
            for unit in self.units
            if unit["end_ms"] > candidate_start and unit["start_ms"] < candidate_end
        ]
        self.state.confidence = len(unique_terms) / len(self.task.search_terms)
        self.state.transcript_evidence = [item[2]["text"] for item in cluster]
        return {"hits": len(cluster), "matched_terms": sorted(unique_terms)}

    def _inspect(self) -> dict[str, Any]:
        if not self.state.candidate_shot_ids:
            return {"error": "no_candidate"}
        shot_id = self.state.candidate_shot_ids[len(self.state.candidate_shot_ids) // 2]
        unit = self.by_shot[shot_id]
        self.state.inspected_shot_id = shot_id
        self.state.inspected_keyframe = unit["keyframe_path"]
        self.state.visual_tags = list(self.visual_annotations.get(shot_id, []))
        return {
            "shot_id": shot_id,
            "keyframe_path": unit["keyframe_path"],
            "visual_tags": self.state.visual_tags,
        }

    def _expand(self) -> dict[str, Any]:
        if not self.state.candidate_shot_ids:
            return {"error": "no_candidate"}
        self.state.candidate_start_ms = max(0, self.state.candidate_start_ms - 5_000)
        self.state.candidate_end_ms = min(480_000, self.state.candidate_end_ms + 5_000)
        expanded_ids = [
            unit["shot_id"]
            for unit in self.units
            if unit["end_ms"] > self.state.candidate_start_ms
            and unit["start_ms"] < self.state.candidate_end_ms
        ]
        self.state.candidate_shot_ids = expanded_ids
        return {"expanded_shot_ids": expanded_ids}

    def _audit(self) -> dict[str, Any]:
        transcript_ok = bool(self.state.transcript_evidence)
        keyframe_ok = bool(self.state.inspected_keyframe)
        in_candidate = self.state.inspected_shot_id in self.state.candidate_shot_ids
        self.state.audit_passed = transcript_ok and keyframe_ok and in_candidate
        return {
            "agent": "EvidenceAuditor",
            "passed": self.state.audit_passed,
            "transcript_ok": transcript_ok,
            "keyframe_ok": keyframe_ok,
        }

    def step(self, action: str) -> tuple[float, bool, dict[str, Any]]:
        if action not in ACTIONS:
            raise ValueError(f"unknown action: {action}")
        if self.state.done:
            raise RuntimeError("episode already finished")
        valid = self.valid_action_mask()[ACTIONS.index(action)]
        before = asdict(self.state)
        reward = -0.05
        result: dict[str, Any]
        if not valid:
            reward -= 0.25
            result = {"error": "invalid_action"}
        elif action == "search":
            self.state.searched = True
            result = self._search()
        elif action == "inspect":
            self.state.inspected = True
            result = self._inspect()
            if self.state.inspected_keyframe:
                reward += 0.05
        elif action == "expand":
            self.state.expanded = True
            result = self._expand()
        elif action == "audit":
            self.state.audited = True
            result = self._audit()
            if self.state.audit_passed:
                reward += 0.10
        else:
            result = self._submit_result()
            reward = result["task_reward"]
            self.state.done = True

        self.state.tool_calls += 1
        if self.state.tool_calls >= self.max_steps and not self.state.done:
            reward -= 0.5
            self.state.done = True
            result["truncated"] = True
        self.trajectory.append(
            {"step": len(self.trajectory) + 1, "action": action, "state_before": before, "result": result, "reward": reward}
        )
        return reward, self.state.done, result

    def _submit_result(self) -> dict[str, Any]:
        if self.state.candidate_start_ms is None or self.state.candidate_end_ms is None:
            return {"task_reward": -1.0, "temporal_iou": 0.0, "success": False}
        iou = temporal_iou(
            self.state.candidate_start_ms,
            self.state.candidate_end_ms,
            self.task.target_start_ms,
            self.task.target_end_ms,
        )
        evidence_supported = self.state.inspected and bool(self.state.inspected_keyframe)
        task_reward = 2.0 * iou + 0.4 * evidence_supported + 0.3 * self.state.audit_passed
        return {
            "task_reward": task_reward,
            "predicted_start_ms": self.state.candidate_start_ms,
            "predicted_end_ms": self.state.candidate_end_ms,
            "target_start_ms": self.task.target_start_ms,
            "target_end_ms": self.task.target_end_ms,
            "temporal_iou": iou,
            "evidence_supported": evidence_supported,
            "audit_passed": self.state.audit_passed,
            "success": iou >= 0.5 and evidence_supported,
            "evidence_shot_ids": self.state.candidate_shot_ids,
            "keyframe_path": self.state.inspected_keyframe,
        }

    def final_result(self) -> dict[str, Any]:
        for step in reversed(self.trajectory):
            if step["action"] == "submit":
                return step["result"]
        return {"success": False, "temporal_iou": 0.0}
