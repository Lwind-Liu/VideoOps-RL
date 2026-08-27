import copy
import json
from pathlib import Path

import numpy as np
import pytest

from server.tool_schemas import TOOL_SCHEMAS
from videoops_rl.training_data_quality import (
    validate_feature_matrix,
    validate_grpo_record,
    validate_sft_record,
)


ROOT = Path(__file__).resolve().parents[1]


def read_first(name: str) -> dict:
    path = ROOT / "data/training" / name
    return json.loads(path.read_text(encoding="utf-8").splitlines()[0])


def test_committed_training_records_pass_quality_contract():
    validate_sft_record(read_first("sft_train_v2.jsonl"), TOOL_SCHEMAS)
    validate_grpo_record(read_first("grpo_train_v2.jsonl"))


def test_feature_validator_rejects_non_finite_and_wrong_dimension():
    with pytest.raises(ValueError, match="NaN or Inf"):
        validate_feature_matrix(np.asarray([[float("nan")] * 512], dtype=np.float32), "bad")
    with pytest.raises(ValueError, match="shape"):
        validate_feature_matrix(np.ones((2, 16), dtype=np.float32), "bad")


def test_grpo_validator_rejects_hidden_labels():
    record = read_first("grpo_train_v2.jsonl")
    public = json.loads(record["prompt"][1]["content"])
    public["target_segments"] = [{"start_ms": 0, "end_ms": 1000}]
    record["prompt"][1]["content"] = json.dumps(public)
    with pytest.raises(ValueError, match="non-public"):
        validate_grpo_record(record)


def test_sft_validator_rejects_scorer_feedback():
    record = copy.deepcopy(read_first("sft_train_v2.jsonl"))
    record["messages"].insert(-1, {"role": "tool", "content": [{"type": "text", "text": '{"reward": 1.0}'}]})
    with pytest.raises(ValueError):
        validate_sft_record(record, TOOL_SCHEMAS)
