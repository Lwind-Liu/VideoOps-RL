"""Build utterance records and shot-aligned multimodal evidence units."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from videoops_rl.alignment import (  # noqa: E402
    ShotInterval,
    align_shots_and_utterances,
    cues_to_utterances,
)
from videoops_rl.srt import parse_srt  # noqa: E402


def load_shots(path: Path) -> list[ShotInterval]:
    shots: list[ShotInterval] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            shots.append(
                ShotInterval(
                    shot_id=record["shot_id"],
                    start_ms=round(record["start_s"] * 1000),
                    end_ms=round(record["end_s"] * 1000),
                    keyframe_ms=round(record["keyframe_s"] * 1000),
                    keyframe_path=record["keyframe_path"],
                )
            )
    return shots


def write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shots", required=True, type=Path)
    parser.add_argument("--subtitles", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--language", default="en")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for path in (args.shots, args.subtitles):
        if not path.is_file():
            raise FileNotFoundError(path)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    shots = load_shots(args.shots)
    cues = parse_srt(args.subtitles.read_text(encoding="utf-8-sig"))
    utterances = cues_to_utterances(cues, args.language)
    evidence_units = align_shots_and_utterances(shots, utterances)

    utterance_records = [utterance.to_dict() for utterance in utterances]
    write_jsonl(args.output_dir / "utterances.jsonl", utterance_records)
    write_jsonl(args.output_dir / "evidence_units.jsonl", evidence_units)

    link_counts = Counter(
        reference["utterance_id"]
        for unit in evidence_units
        for reference in unit["utterance_refs"]
    )
    dialogue_counts = [len(unit["utterance_refs"]) for unit in evidence_units]
    summary = {
        "method": "half_open_interval_overlap",
        "interval_semantics": "[start_ms, end_ms)",
        "shots_source": args.shots.as_posix(),
        "subtitles_source": args.subtitles.as_posix(),
        "language": args.language,
        "shot_count": len(shots),
        "utterance_count": len(utterances),
        "shot_utterance_link_count": sum(dialogue_counts),
        "shots_with_dialogue": sum(count > 0 for count in dialogue_counts),
        "shots_without_dialogue": sum(count == 0 for count in dialogue_counts),
        "max_utterances_per_shot": max(dialogue_counts, default=0),
        "median_utterances_per_shot": statistics.median(dialogue_counts),
        "utterances_crossing_shot_boundaries": sum(count > 1 for count in link_counts.values()),
        "unlinked_utterances": sorted(
            utterance.utterance_id
            for utterance in utterances
            if utterance.utterance_id not in link_counts
        ),
    }
    (args.output_dir / "alignment_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"Aligned {len(utterances)} utterances to {len(shots)} shots; "
        f"created {sum(dialogue_counts)} links"
    )
    print(args.output_dir / "evidence_units.jsonl")


if __name__ == "__main__":
    main()
