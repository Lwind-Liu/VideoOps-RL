"""Run one end-to-end learned-policy demo and print the auditable result."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from videoops_rl.business_env import RealHighlightEnv, load_jsonl, load_tasks  # noqa: E402
from videoops_rl.coordinator import CoordinatorPolicy, rollout_policy  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="VideoOps-RL query-highlight demo")
    parser.add_argument("--task-id", default="tos_hl_008")
    args = parser.parse_args()

    tasks = {task.task_id: task for task in load_tasks(REPO_ROOT / "data/annotations/highlight_tasks.jsonl")}
    if args.task_id not in tasks:
        raise SystemExit(f"unknown task id: {args.task_id}; choose from {', '.join(tasks)}")
    task = tasks[args.task_id]
    evidence_dir = REPO_ROOT / "data/processed/tears_of_steel/s1/shots_ffmpeg_t035"
    units = load_jsonl(evidence_dir / "evidence_units.jsonl")
    utterances = load_jsonl(evidence_dir / "utterances.jsonl")
    visual = json.loads((REPO_ROOT / "data/annotations/visual_evidence.json").read_text(encoding="utf-8"))

    checkpoint = torch.load(
        REPO_ROOT / "artifacts/checkpoints/coordinator_group_relative.pt",
        map_location="cpu",
        weights_only=False,
    )
    policy = CoordinatorPolicy()
    policy.load_state_dict(checkpoint["state_dict"])
    policy.eval()
    episode = rollout_policy(
        policy,
        RealHighlightEnv(units, utterances, visual, task),
        torch.device("cpu"),
        deterministic=True,
    )

    clip = REPO_ROOT / "outputs/highlights" / f"{task.task_id}.mp4"
    card = {
        "task_id": task.task_id,
        "query": task.query,
        "learned_action_sequence": [step["action"] for step in episode.trajectory],
        "predicted_segment_seconds": [
            episode.result["predicted_start_ms"] / 1000,
            episode.result["predicted_end_ms"] / 1000,
        ],
        "temporal_iou": round(episode.result["temporal_iou"], 4),
        "evidence_supported": episode.result["evidence_supported"],
        "audit_passed": episode.result["audit_passed"],
        "keyframe_path": episode.result["keyframe_path"],
        "exported_clip": str(clip) if clip.exists() else "run scripts/export_highlights.py first",
        "claim_boundary": "single open-film POC; not a production benchmark",
    }
    print(json.dumps(card, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
