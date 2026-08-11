import json
from pathlib import Path

from videoops_rl.dataset_protocol import read_jsonl
from videoops_rl.evidence_graph import TemporalEvidenceGraph
from videoops_rl.multivideo_env import MultiVideoHighlightEnv

ROOT = Path(__file__).resolve().parents[1]


def test_multimodal_agreement_increases_fused_belief():
    graph = TemporalEvidenceGraph([{"shot_id": "a", "start_ms": 0, "end_ms": 1000}, {"shot_id": "b", "start_ms": 1000, "end_ms": 2000}])
    graph.update("text", {"a": 0.8, "b": 0.3})
    text_only = graph.nodes["a"].fused_score
    graph.update("image", {"a": 0.7, "b": 0.9})
    assert graph.nodes["a"].fused_score > text_only
    assert graph.nodes["a"].sources == {"text", "image"}


def test_hard_constraint_rejects_unretrieved_inspection():
    task = read_jsonl(ROOT / "data/registry/formal_tasks_v1.jsonl")[0]
    env = MultiVideoHighlightEnv(ROOT, task)
    observation = env.inspect_keyframe("shot_0098")
    assert observation["error"] == "shot_not_in_retrieved_candidates"
    assert env.state.invalid_calls == 1


def test_grounded_correct_submission_gets_decomposed_reward():
    task = read_jsonl(ROOT / "data/registry/formal_tasks_v1.jsonl")[0]
    env = MultiVideoHighlightEnv(ROOT, task)
    shot_id = task["evidence"]["shot_ids"][0]
    env.graph.update("text", {shot_id: 1.0})
    env.graph.update("image", {shot_id: 1.0})
    env.state.searched_modalities.extend(["text", "image"])
    env.inspect_keyframe(shot_id)
    audit = env.request_audit([shot_id])
    result = env.submit([shot_id])
    assert audit["passed"]
    assert result["audit_passed"]
    assert set(result["reward_parts"]) == {"temporal", "shot_set", "semantic_audit", "modality", "process", "tool_cost", "repeat_cost", "invalid_cost"}
    assert result["reward"] > 0.5


def test_grpo_prompt_has_no_hidden_process_or_target_labels():
    task = read_jsonl(ROOT / "data/registry/formal_tasks_v1.jsonl")[0]
    prompt = json.dumps(MultiVideoHighlightEnv(ROOT, task).public_prompt)
    assert "target_segments" not in prompt
    assert "process_reward" not in prompt
    assert task["evidence"]["shot_ids"][0] not in prompt
