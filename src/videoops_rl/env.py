from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .schemas import Observation, Segment, VideoTask


@dataclass(frozen=True)
class StepResult:
    observation: dict[str, Any]
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, Any]


class SyntheticVideoEnv:
    """Dependency-free environment for validating the first agentic-RL loop.

    It deliberately exposes only evidence-acquisition actions. Real video
    adapters will later implement the same contract over decoded media.
    """

    def __init__(self, segments: list[Segment], tasks: list[VideoTask]):
        self.segments = {segment.segment_id: segment for segment in segments}
        self.tasks = {task.task_id: task for task in tasks}
        self.task: VideoTask | None = None
        self.state: Observation | None = None
        self.done = False

    def reset(self, task_id: str) -> dict[str, Any]:
        if task_id not in self.tasks:
            raise KeyError(f"unknown task: {task_id}")
        self.task = self.tasks[task_id]
        self.state = Observation(
            task_id=task_id,
            query=self.task.query,
            remaining_steps=self.task.max_steps,
        )
        self.done = False
        return self.state.as_dict()

    def step(self, action: dict[str, Any]) -> StepResult:
        if self.task is None or self.state is None:
            raise RuntimeError("call reset(task_id) before step(action)")
        if self.done:
            raise RuntimeError("episode is finished; call reset(task_id)")

        self.state.remaining_steps -= 1
        self.state.last_action = dict(action)
        action_type = action.get("type")
        reward = -0.05
        info: dict[str, Any] = {"action_type": action_type}
        reward_components = {"tool_cost": -0.05}

        if action_type == "search_transcript":
            query = str(action.get("query", ""))
            hits = {
                segment_id
                for segment_id, segment in self.segments.items()
                if segment.contains_query(query)
            }
            self.state.transcript_hits.update(hits)
            target_ids = set(self.task.target_segment_ids)
            true_positives = len(hits & target_ids)
            precision = true_positives / len(hits) if hits else 0.0
            recall = true_positives / len(target_ids) if target_ids else 0.0
            f1 = (
                2 * precision * recall / (precision + recall)
                if precision + recall > 0
                else 0.0
            )
            search_quality_bonus = 0.10 * f1
            reward += search_quality_bonus
            reward_components["search_f1_bonus"] = search_quality_bonus
            if not hits:
                reward -= 0.05
                reward_components["no_hit_penalty"] = -0.05
            info["hits"] = sorted(hits)
            info["search_metrics"] = {
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }

        elif action_type == "inspect_segment":
            segment_id = str(action.get("segment_id", ""))
            segment = self.segments.get(segment_id)
            if segment is None:
                reward -= 0.15
                reward_components["invalid_segment_penalty"] = -0.15
                info["error"] = "unknown_segment"
            elif segment_id in self.state.inspected_segment_ids:
                reward -= 0.10
                reward_components["duplicate_inspection_penalty"] = -0.10
                info["error"] = "duplicate_inspection"
            else:
                self.state.inspected_segment_ids.add(segment_id)
                self.state.visual_observations[segment_id] = segment.visual_tags
                if segment_id in self.task.target_segment_ids:
                    reward += 0.20
                    reward_components["target_evidence_bonus"] = 0.20

        elif action_type == "answer":
            submitted = {str(item) for item in action.get("segment_ids", [])}
            correct = submitted == set(self.task.target_segment_ids)
            evidence = submitted.issubset(self.state.inspected_segment_ids)
            if correct and evidence:
                reward = 1.0
                reward_components = {"grounded_success": 1.0}
                info["success"] = True
            else:
                reward = -0.5
                reward_components = {"failed_submission": -0.5}
                info.update({"success": False, "correct": correct, "evidence": evidence})
            self.done = True

        else:
            reward -= 0.15
            reward_components["unknown_action_penalty"] = -0.15
            info["error"] = "unknown_action"

        truncated = self.state.remaining_steps <= 0 and not self.done
        if truncated:
            reward -= 0.20
            reward_components["budget_exhausted_penalty"] = -0.20
            self.done = True
            info["error"] = "step_budget_exhausted"

        info["reward_components"] = reward_components

        return StepResult(
            observation=self.state.as_dict(),
            reward=reward,
            terminated=self.done and not truncated,
            truncated=truncated,
            info=info,
        )
