from __future__ import annotations

import re
from dataclasses import dataclass


TIMESTAMP_PATTERN = re.compile(
    r"^(?P<start>\d{2}:\d{2}:\d{2},\d{3})\s+-->\s+"
    r"(?P<end>\d{2}:\d{2}:\d{2},\d{3})$"
)


@dataclass(frozen=True)
class SubtitleCue:
    start_ms: int
    end_ms: int
    text: str


def timestamp_to_ms(timestamp: str) -> int:
    hours, minutes, rest = timestamp.split(":")
    seconds, milliseconds = rest.split(",")
    return (
        int(hours) * 3_600_000
        + int(minutes) * 60_000
        + int(seconds) * 1_000
        + int(milliseconds)
    )


def ms_to_timestamp(value_ms: int) -> str:
    hours, remainder = divmod(value_ms, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    seconds, milliseconds = divmod(remainder, 1_000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"


def parse_srt(text: str) -> list[SubtitleCue]:
    cues: list[SubtitleCue] = []
    normalized = text.replace("\r\n", "\n").strip()
    for block in re.split(r"\n\s*\n", normalized):
        lines = [line.rstrip() for line in block.splitlines()]
        if len(lines) < 2:
            continue
        timestamp_index = 1 if lines[0].strip().isdigit() else 0
        match = TIMESTAMP_PATTERN.match(lines[timestamp_index].strip())
        if not match:
            raise ValueError(f"invalid SRT timestamp: {lines[timestamp_index]!r}")
        cue_text = "\n".join(lines[timestamp_index + 1 :]).strip()
        cues.append(
            SubtitleCue(
                start_ms=timestamp_to_ms(match.group("start")),
                end_ms=timestamp_to_ms(match.group("end")),
                text=cue_text,
            )
        )
    return cues


def clip_cues(
    cues: list[SubtitleCue], start_ms: int, end_ms: int
) -> list[SubtitleCue]:
    if start_ms < 0 or end_ms <= start_ms:
        raise ValueError("clip interval must satisfy 0 <= start_ms < end_ms")

    clipped: list[SubtitleCue] = []
    for cue in cues:
        if cue.end_ms <= start_ms or cue.start_ms >= end_ms:
            continue
        clipped.append(
            SubtitleCue(
                start_ms=max(cue.start_ms, start_ms) - start_ms,
                end_ms=min(cue.end_ms, end_ms) - start_ms,
                text=cue.text,
            )
        )
    return clipped


def render_srt(cues: list[SubtitleCue]) -> str:
    blocks = []
    for index, cue in enumerate(cues, start=1):
        blocks.append(
            f"{index}\n{ms_to_timestamp(cue.start_ms)} --> "
            f"{ms_to_timestamp(cue.end_ms)}\n{cue.text}"
        )
    return "\n\n".join(blocks) + "\n"

