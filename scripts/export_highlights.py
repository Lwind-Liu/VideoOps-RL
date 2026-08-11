"""Export real MP4 clips and evidence cards from evaluated RL predictions."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    report = json.loads((REPO_ROOT / "outputs/reports/mvp_evaluation.json").read_text(encoding="utf-8"))
    source = REPO_ROOT / "data/processed/tears_of_steel/s1/sample_20s_500s.mp4"
    output_dir = REPO_ROOT / "outputs/highlights"
    output_dir.mkdir(parents=True, exist_ok=True)
    selected = [item for item in report["tasks"] if item["method"] == "rl_multi_agent" and item.get("success")]
    for item in selected:
        start_s = item["predicted_start_ms"] / 1000
        duration_s = (item["predicted_end_ms"] - item["predicted_start_ms"]) / 1000
        output = output_dir / f"{item['task_id']}.mp4"
        subprocess.run(
            [
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                "-ss", f"{start_s:.3f}", "-i", str(source), "-t", f"{duration_s:.3f}",
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
                "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart", str(output),
            ],
            check=True,
        )
        evidence = {
            "task_id": item["task_id"],
            "query": item["query"],
            "clip_path": output.relative_to(REPO_ROOT).as_posix(),
            "start_ms": item["predicted_start_ms"],
            "end_ms": item["predicted_end_ms"],
            "temporal_iou": item["temporal_iou"],
            "audit_passed": item["audit_passed"],
            "evidence_shot_ids": item.get("evidence_shot_ids", []),
            "keyframe_path": item.get("keyframe_path"),
        }
        (output_dir / f"{item['task_id']}.json").write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"exported={len(selected)} clips to {output_dir}")


if __name__ == "__main__":
    main()
