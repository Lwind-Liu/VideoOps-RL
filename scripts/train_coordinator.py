"""Train the lightweight group-relative Coordinator policy."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from videoops_rl.business_env import RealHighlightEnv, load_jsonl, load_tasks  # noqa: E402
from videoops_rl.coordinator import CoordinatorPolicy, train_group_relative_policy  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device if args.device != "auto" else "cpu"
    )
    tasks = load_tasks(REPO_ROOT / "data/annotations/highlight_tasks.jsonl")
    train_tasks = [task for task in tasks if task.split == "train"]
    evidence_dir = REPO_ROOT / "data/processed/tears_of_steel/s1/shots_ffmpeg_t035"
    evidence_units = load_jsonl(evidence_dir / "evidence_units.jsonl")
    utterances = load_jsonl(evidence_dir / "utterances.jsonl")
    visual = json.loads((REPO_ROOT / "data/annotations/visual_evidence.json").read_text(encoding="utf-8"))

    def env_factory(task):
        return RealHighlightEnv(evidence_units, utterances, visual, task)

    torch.manual_seed(args.seed)
    policy = CoordinatorPolicy()
    history = train_group_relative_policy(
        policy,
        env_factory,
        train_tasks,
        device,
        epochs=args.epochs,
        group_size=args.group_size,
        seed=args.seed,
    )

    output_dir = REPO_ROOT / "artifacts/checkpoints"
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / "coordinator_group_relative.pt"
    torch.save(
        {
            "state_dict": policy.cpu().state_dict(),
            "feature_dim": 9,
            "actions": ["search", "inspect", "expand", "audit", "submit"],
            "training": {
                "algorithm": "structured_group_relative_clipped_policy_gradient",
                "epochs": args.epochs,
                "group_size": args.group_size,
                "seed": args.seed,
                "train_task_ids": [task.task_id for task in train_tasks],
                "device": str(device),
                "boundary": "structured policy prototype; not token-level LLM GRPO",
            },
        },
        checkpoint,
    )
    (output_dir / "coordinator_training_log.json").write_text(
        json.dumps(history, indent=2) + "\n", encoding="utf-8"
    )
    print(f"trained_on={device} tasks={len(train_tasks)} epochs={args.epochs}")
    print(checkpoint)
    print(json.dumps(history[-1], indent=2))


if __name__ == "__main__":
    main()
