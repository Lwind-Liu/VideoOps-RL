from pathlib import Path

import numpy as np

from videoops_rl.agents import multi_agent_expert
from videoops_rl.dataset_protocol import read_jsonl
from videoops_rl.qv_env import QVHighlightsEnv, decode_temporal_proposals
from server.train_llm_grpo import VideoOpsGRPOEnvironment


ROOT = Path(__file__).resolve().parents[1]


def test_temporal_decoder_builds_variable_contiguous_proposal():
    scores = np.asarray([0.0, 0.1, 0.8, 0.9, 0.85, 0.1, 0.0], dtype=np.float32)
    proposal = decode_temporal_proposals(scores)[0]
    indices = [int(item.split("_")[1]) for item in proposal.shot_ids]
    assert proposal.anchor_shot_id in proposal.shot_ids
    assert indices == list(range(indices[0], indices[-1] + 1))
    assert proposal.end_ms > proposal.start_ms


def test_qv_prompt_hides_labels_and_expert_is_grounded():
    task = read_jsonl(ROOT / "data/external/qvhighlights/annotations/tasks_val_v1.jsonl")[0]
    env = QVHighlightsEnv(ROOT, task)
    prompt = str(env.public_prompt)
    assert "target_segments" not in prompt
    assert task["evidence"]["shot_ids"][0] not in prompt
    result = multi_agent_expert(env)
    assert result["audit_passed"]
    assert result["evidence_supported"]
    assert result["temporal_iou"] > 0.5
    assert [step["tool"] for step in env.trajectory] == [
        "search_visual", "inspect_keyframe", "expand_context", "request_audit", "submit"
    ]


def test_qv_rejects_unretrieved_inspection():
    task = read_jsonl(ROOT / "data/external/qvhighlights/annotations/tasks_val_v1.jsonl")[0]
    env = QVHighlightsEnv(ROOT, task)
    observation = env.inspect_keyframe("clip_0000")
    assert observation["error"] == "shot_not_in_retrieved_candidates"
    assert env.state.invalid_calls == 1


def test_grpo_external_task_id_makes_group_reset_deterministic():
    wrapper = VideoOpsGRPOEnvironment()
    task_id = wrapper.qv_tasks[0]["task_id"]
    assert wrapper.reset(task_id=task_id) is None
    first_video = wrapper.env.task["video_id"]
    assert wrapper.reset(task_id=task_id) is None
    assert wrapper.env.task["task_id"] == task_id
    assert wrapper.env.task["video_id"] == first_video


def test_grpo_train_mix_matches_configured_ratio():
    records = read_jsonl(ROOT / "data/training/grpo_train_v2.jsonl")
    formal = sum(item["source"] == "formal_multimodal" for item in records)
    assert len(records) == 9018
    assert formal == 1800
    assert abs(formal / len(records) - 0.20) < 0.001
