from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from videoops_rl import Segment, SyntheticVideoEnv, VideoTask


def build_env() -> SyntheticVideoEnv:
    segments = [
        Segment("seg_01", 0, 5000, "控制室设备正常运行", ("control_room", "normal")),
        Segment("seg_02", 5000, 10000, "主角发现设备冒烟", ("person", "smoke", "alarm")),
        Segment("seg_03", 10000, 15000, "主角立即关闭电源", ("person", "shutdown", "safety")),
    ]
    task = VideoTask(
        task_id="task_02",
        query="主角如何发现并处理设备异常？",
        target_segment_ids=frozenset({"seg_02", "seg_03"}),
        answer="主角先发现设备冒烟，随后关闭电源。",
    )
    return SyntheticVideoEnv(segments, [task])


TRAJECTORIES = {
    "complete": [
        {"type": "search_transcript", "query": "主角"},
        {"type": "inspect_segment", "segment_id": "seg_02"},
        {"type": "inspect_segment", "segment_id": "seg_03"},
        {"type": "answer", "segment_ids": ["seg_02", "seg_03"]},
    ],
    "partial_answer": [
        {"type": "search_transcript", "query": "主角"},
        {"type": "inspect_segment", "segment_id": "seg_02"},
        {"type": "answer", "segment_ids": ["seg_02"]},
    ],
    "missing_evidence": [
        {"type": "search_transcript", "query": "主角"},
        {"type": "inspect_segment", "segment_id": "seg_02"},
        {"type": "answer", "segment_ids": ["seg_02", "seg_03"]},
    ],
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the S0 multi-segment task.")
    parser.add_argument("trajectory", choices=TRAJECTORIES)
    args = parser.parse_args()

    env = build_env()
    observation = env.reset("task_02")
    print("INITIAL OBSERVATION")
    print(json.dumps(observation, ensure_ascii=False, indent=2))

    total_reward = 0.0
    for index, action in enumerate(TRAJECTORIES[args.trajectory], start=1):
        result = env.step(action)
        total_reward += result.reward
        print(f"\nSTEP {index}")
        print("action:", json.dumps(action, ensure_ascii=False))
        print("reward:", round(result.reward, 4))
        print("info:", json.dumps(result.info, ensure_ascii=False))
        if result.terminated or result.truncated:
            break

    print("\nEPISODE RETURN:", round(total_reward, 4))


if __name__ == "__main__":
    main()

