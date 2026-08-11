import json
from pathlib import Path

from videoops_rl.agents import multi_agent_expert
from videoops_rl.dataset_protocol import read_jsonl
from videoops_rl.multivideo_env import MultiVideoHighlightEnv, temporal_iou

ROOT = Path(__file__).resolve().parents[1]


def test_temporal_iou_multisegment():
    assert temporal_iou([{"start_ms": 0, "end_ms": 10}], [{"start_ms": 5, "end_ms": 15}]) == 1 / 3


def test_public_prompt_hides_ground_truth():
    task = read_jsonl(ROOT / "data/registry/formal_tasks_v1.jsonl")[0]
    prompt = json.dumps(MultiVideoHighlightEnv(ROOT, task).public_prompt)
    assert "target_segments" not in prompt and "shot_ids" not in prompt


def test_multimodal_expert_runs_text_and_visual_task():
    tasks = read_jsonl(ROOT / "data/registry/formal_tasks_v1.jsonl")
    for task in (tasks[0], next(item for item in tasks if item["video_id"] == "big_buck_bunny_test")):
        env = MultiVideoHighlightEnv(ROOT, task)
        result = multi_agent_expert(env)
        assert env.state.done and "reward" in result
        assert env.state.inspected_shot_ids

