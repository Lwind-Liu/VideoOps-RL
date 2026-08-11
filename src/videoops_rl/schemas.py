from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Segment:
    """A time-aligned multimodal unit used by the agent environment."""

    segment_id: str
    start_ms: int
    end_ms: int
    transcript: str = ""
    visual_tags: tuple[str, ...] = ()

    def contains_query(self, query: str) -> bool:
        terms = {term.lower() for term in query.split() if term.strip()}
        text = self.transcript.lower()
        return bool(terms) and all(term in text for term in terms)


@dataclass(frozen=True)
class VideoTask:
    task_id: str
    query: str
    target_segment_ids: frozenset[str]
    answer: str
    max_steps: int = 8


@dataclass
class Observation:
    task_id: str
    query: str
    inspected_segment_ids: set[str] = field(default_factory=set)
    transcript_hits: set[str] = field(default_factory=set)
    visual_observations: dict[str, tuple[str, ...]] = field(default_factory=dict)
    last_action: dict[str, Any] | None = None
    remaining_steps: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "query": self.query,
            "inspected_segment_ids": sorted(self.inspected_segment_ids),
            "transcript_hits": sorted(self.transcript_hits),
            "visual_observations": {
                key: list(value) for key, value in self.visual_observations.items()
            },
            "last_action": self.last_action,
            "remaining_steps": self.remaining_steps,
        }
