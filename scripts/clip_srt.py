from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from videoops_rl.srt import clip_cues, parse_srt, render_srt


def main() -> None:
    parser = argparse.ArgumentParser(description="Clip and zero-base an SRT file.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--start-seconds", required=True, type=float)
    parser.add_argument("--duration-seconds", required=True, type=float)
    args = parser.parse_args()

    start_ms = round(args.start_seconds * 1000)
    end_ms = start_ms + round(args.duration_seconds * 1000)
    source_cues = parse_srt(args.input.read_text(encoding="utf-8-sig"))
    clipped_cues = clip_cues(source_cues, start_ms, end_ms)
    args.output.write_text(render_srt(clipped_cues), encoding="utf-8")

    if clipped_cues and clipped_cues[-1].end_ms > end_ms - start_ms:
        raise RuntimeError("clipped subtitle exceeds requested duration")
    print(f"wrote {len(clipped_cues)} cues to {args.output}")


if __name__ == "__main__":
    main()

