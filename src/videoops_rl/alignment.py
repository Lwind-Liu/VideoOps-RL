"""Time-overlap alignment for shots, subtitles, and multimodal evidence units."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from .srt import SubtitleCue


@dataclass(frozen=True)
class Utterance:
    utterance_id: str
    start_ms: int
    end_ms: int
    text: str
    language: str
    speaker: str | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ShotInterval:
    shot_id: str
    start_ms: int
    end_ms: int
    keyframe_ms: int
    keyframe_path: str


def cues_to_utterances(cues: list[SubtitleCue], language: str) -> list[Utterance]:
    return [
        Utterance(
            utterance_id=f"utt_{index:04d}",
            start_ms=cue.start_ms,
            end_ms=cue.end_ms,
            text=cue.text,
            language=language,
        )
        for index, cue in enumerate(cues, 1)
    ]


def overlap_interval(
    first_start: int, first_end: int, second_start: int, second_end: int
) -> tuple[int, int] | None:
    """Return intersection of two half-open intervals, or None."""
    start = max(first_start, second_start)
    end = min(first_end, second_end)
    return (start, end) if start < end else None


def align_shots_and_utterances(
    shots: list[ShotInterval], utterances: list[Utterance]
) -> list[dict[str, object]]:
    evidence_units: list[dict[str, object]] = []
    for shot in shots:
        references: list[dict[str, object]] = []
        transcript_parts: list[str] = []
        for utterance in utterances:
            overlap = overlap_interval(
                shot.start_ms, shot.end_ms, utterance.start_ms, utterance.end_ms
            )
            if overlap is None:
                continue
            overlap_start, overlap_end = overlap
            references.append(
                {
                    "utterance_id": utterance.utterance_id,
                    "overlap_start_ms": overlap_start,
                    "overlap_end_ms": overlap_end,
                    "overlap_ms": overlap_end - overlap_start,
                }
            )
            transcript_parts.append(" ".join(utterance.text.split()))

        evidence_units.append(
            {
                "evidence_id": f"ev_{shot.shot_id}",
                "shot_id": shot.shot_id,
                "start_ms": shot.start_ms,
                "end_ms": shot.end_ms,
                "keyframe_ms": shot.keyframe_ms,
                "keyframe_path": shot.keyframe_path,
                "utterance_refs": references,
                "transcript": " ".join(transcript_parts),
                "has_dialogue": bool(references),
            }
        )
    return evidence_units
