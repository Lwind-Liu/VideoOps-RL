"""Shot-boundary data structures shared by preprocessing scripts and tests."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Shot:
    shot_id: str
    start_s: float
    end_s: float
    duration_s: float
    keyframe_s: float
    start_scene_score: float | None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def build_shots(
    transitions: list[tuple[float, float]], duration_s: float
) -> list[Shot]:
    """Convert transition timestamps into contiguous shots.

    A transition timestamp is the first frame of the new shot. The first shot
    begins at zero, and every keyframe is chosen at its shot midpoint.
    """
    if duration_s <= 0:
        raise ValueError("duration_s must be positive")

    scores_by_time: dict[float, float] = {}
    for timestamp, score in transitions:
        timestamp = round(float(timestamp), 6)
        if 0.0 < timestamp < duration_s:
            scores_by_time[timestamp] = float(score)

    boundaries = [0.0, *sorted(scores_by_time), round(float(duration_s), 6)]
    shots: list[Shot] = []
    for index, (start_s, end_s) in enumerate(zip(boundaries, boundaries[1:]), 1):
        if end_s <= start_s:
            continue
        shots.append(
            Shot(
                shot_id=f"shot_{index:04d}",
                start_s=start_s,
                end_s=end_s,
                duration_s=round(end_s - start_s, 6),
                keyframe_s=round((start_s + end_s) / 2.0, 6),
                start_scene_score=scores_by_time.get(start_s),
            )
        )
    return shots
