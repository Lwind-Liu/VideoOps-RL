"""Semantic validation for tasks, CLIP assets and training records."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import jsonschema
import numpy as np


PUBLIC_PROMPT_KEYS = {
    "task_id", "video_id", "query", "video_has_dialogue", "tool_budget",
    "rules", "available_tools", "evidence_type",
}
HIDDEN_KEYS = {"target_segments", "search_terms", "saliency_scores", "teacher_reward", "teacher_success"}
SCORER_KEYS = {"temporal_iou", "shot_f1", "reward", "reward_parts", "success"}


def validate_feature_matrix(matrix: np.ndarray, name: str, dimension: int = 512) -> None:
    values = np.asarray(matrix)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] != dimension:
        raise ValueError(f"{name} must have shape [N, {dimension}], found {values.shape}")
    if not np.issubdtype(values.dtype, np.floating):
        raise ValueError(f"{name} must contain floating-point features, found {values.dtype}")
    if not np.isfinite(values).all():
        raise ValueError(f"{name} contains NaN or Inf")
    if np.any(np.linalg.norm(values.astype(np.float32), axis=1) <= 1e-8):
        raise ValueError(f"{name} contains a zero-norm feature")


def validate_tasks(tasks: list[dict[str, Any]], schema_path: Path, durations: dict[str, int]) -> dict[str, int]:
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    ids = [str(task.get("task_id", "")) for task in tasks]
    duplicates = [task_id for task_id, count in Counter(ids).items() if count > 1]
    if duplicates:
        raise ValueError(f"duplicate task_id values: {duplicates[:10]}")
    for task in tasks:
        jsonschema.validate(task, schema)
        duration = int(task.get("duration_ms", durations.get(task["video_id"], 0)))
        for segment in task["target_segments"]:
            start, end = int(segment["start_ms"]), int(segment["end_ms"])
            if end <= start:
                raise ValueError(f"non-positive target interval in {task['task_id']}: {start}-{end}")
            if duration and end > duration:
                raise ValueError(f"target exceeds video duration in {task['task_id']}: {end}>{duration}")
        if len(task["evidence"]["shot_ids"]) != len(set(task["evidence"]["shot_ids"])):
            raise ValueError(f"duplicate evidence shot IDs in {task['task_id']}")
    return dict(Counter(task["split"] for task in tasks))


def _text(content: Any) -> str | None:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                return block.get("text")
    return None


def _nested_keys(value: Any) -> set[str]:
    if isinstance(value, dict):
        return set(value) | {key for item in value.values() for key in _nested_keys(item)}
    if isinstance(value, list):
        return {key for item in value for key in _nested_keys(item)}
    return set()


def validate_sft_record(record: dict[str, Any], tool_schemas: list[dict[str, Any]]) -> None:
    if not record.get("teacher_success"):
        raise ValueError(f"SFT record is not a successful teacher trajectory: {record.get('task_id')}")
    messages = record.get("messages", [])
    if len(messages) < 3 or [messages[0].get("role"), messages[1].get("role")] != ["system", "user"]:
        raise ValueError(f"invalid SFT prefix in {record.get('task_id')}")
    parameter_schemas = {
        tool["function"]["name"]: tool["function"]["parameters"] for tool in tool_schemas
    }
    index = 2
    calls = 0
    while index < len(messages):
        message = messages[index]
        if message.get("role") != "assistant":
            raise ValueError(f"expected assistant at message {index} in {record.get('task_id')}")
        try:
            call = json.loads(_text(message.get("content")) or "")
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid tool JSON in {record.get('task_id')}") from error
        tool, arguments = call.get("tool"), call.get("arguments")
        if tool not in parameter_schemas or not isinstance(arguments, dict):
            raise ValueError(f"unknown tool or arguments in {record.get('task_id')}: {tool}")
        jsonschema.validate(arguments, parameter_schemas[tool])
        calls += 1
        if tool == "submit":
            if index != len(messages) - 1:
                raise ValueError(f"submit must be the final SFT message in {record.get('task_id')}")
            break
        if index + 1 >= len(messages) or messages[index + 1].get("role") != "tool":
            raise ValueError(f"missing tool response after {tool} in {record.get('task_id')}")
        try:
            json.loads(_text(messages[index + 1].get("content")) or "")
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid tool observation after {tool} in {record.get('task_id')}") from error
        index += 2
    if calls == 0 or json.loads(_text(messages[-1]["content"]) or "{}").get("tool") != "submit":
        raise ValueError(f"SFT trajectory does not end with submit: {record.get('task_id')}")
    if SCORER_KEYS & _nested_keys(messages):
        raise ValueError(f"hidden scorer output leaked into SFT messages: {record.get('task_id')}")


def validate_grpo_record(record: dict[str, Any]) -> None:
    expected = {"task_id", "video_id", "source", "split", "prompt"}
    if set(record) != expected:
        raise ValueError(f"unexpected GRPO fields in {record.get('task_id')}: {sorted(set(record) - expected)}")
    prompt = record["prompt"]
    if len(prompt) != 2 or [message.get("role") for message in prompt] != ["system", "user"]:
        raise ValueError(f"invalid GRPO prompt roles in {record.get('task_id')}")
    public = json.loads(prompt[1]["content"])
    unknown = set(public) - PUBLIC_PROMPT_KEYS
    hidden = _nested_keys(public) & HIDDEN_KEYS
    if unknown or hidden:
        raise ValueError(f"non-public GRPO prompt fields in {record.get('task_id')}: {sorted(unknown | hidden)}")
    if public.get("task_id") != record["task_id"] or public.get("video_id") != record["video_id"]:
        raise ValueError(f"GRPO prompt identity mismatch in {record.get('task_id')}")


def duplication_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(record["task_id"] for record in records)
    return {
        "rows": len(records),
        "unique_tasks": len(counts),
        "duplicate_rows": len(records) - len(counts),
        "max_repeat": max(counts.values(), default=0),
    }
