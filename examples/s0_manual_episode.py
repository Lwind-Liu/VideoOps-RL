from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from videoops_rl import Segment, SyntheticVideoEnv, VideoTask


def build_env() -> SyntheticVideoEnv:
    segments = [
        Segment("seg_01", 0, 5000, "主角走进房间", ("person", "room")),
        Segment("seg_02", 5000, 10000, "主角发现异常并说出真相", ("person", "surprise")),
        Segment("seg_03", 10000, 15000, "两人开始争吵", ("person", "conflict")),
    ]
    task = VideoTask(
        task_id="task_01",
        query="发现异常",
        target_segment_ids=frozenset({"seg_02"}),
        answer="主角在第二个片段发现异常",
    )
    return SyntheticVideoEnv(segments, [task])


TRAJECTORIES = {
    "efficient": [
        {"type": "search_transcript", "query": "发现异常"},
        {"type": "inspect_segment", "segment_id": "seg_02"},
        {"type": "answer", "segment_ids": ["seg_02"]},
    ],
    "duplicate": [
        {"type": "search_transcript", "query": "发现异常"},
        {"type": "inspect_segment", "segment_id": "seg_02"},
        {"type": "inspect_segment", "segment_id": "seg_02"},
        {"type": "answer", "segment_ids": ["seg_02"]},
    ],
    "broad_search": [
        {"type": "search_transcript", "query": "主角"},
        {"type": "inspect_segment", "segment_id": "seg_02"},
        {"type": "answer", "segment_ids": ["seg_02"]},
    ],
    "no_evidence": [
        {"type": "answer", "segment_ids": ["seg_02"]},
    ],
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one S0 manual trajectory.")
    parser.add_argument("trajectory", choices=TRAJECTORIES)
    args = parser.parse_args()

    env = build_env()
    observation = env.reset("task_01")
    print("INITIAL OBSERVATION")
    print(json.dumps(observation, ensure_ascii=False, indent=2))

    total_reward = 0.0
    for step_index, action in enumerate(TRAJECTORIES[args.trajectory], start=1):
        result = env.step(action)
        total_reward += result.reward
        print(f"\nSTEP {step_index}")
        print("action:", json.dumps(action, ensure_ascii=False))
        print("reward:", result.reward)
        print("components:", json.dumps(result.info["reward_components"], ensure_ascii=False))
        if "search_metrics" in result.info:
            print("search_metrics:", json.dumps(result.info["search_metrics"], ensure_ascii=False))
        print("state:", json.dumps(result.observation, ensure_ascii=False))
        if result.terminated or result.truncated:
            break

    print("\nEPISODE RETURN:", round(total_reward, 4))


if __name__ == "__main__":
    main()
