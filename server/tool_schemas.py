"""Canonical JSON schemas shared by tool-calling SFT and runtime checks."""

from __future__ import annotations

import uuid


def _function(name: str, description: str, properties: dict, required: list[str]) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
                "additionalProperties": False,
            },
        },
    }


TOOL_SCHEMAS = [
    _function(
        "search_transcript",
        "Search subtitle evidence and return grounded temporal candidates.",
        {"query": {"type": "string"}, "top_k": {"type": "integer", "minimum": 1, "maximum": 10}},
        ["query"],
    ),
    _function(
        "search_visual",
        "Search visual evidence and decode adaptive temporal proposals.",
        {"query": {"type": "string"}, "top_k": {"type": "integer", "minimum": 1, "maximum": 10}},
        ["query"],
    ),
    _function(
        "inspect_keyframe",
        "Inspect the image or official visual feature for one returned candidate.",
        {"shot_id": {"type": "string"}},
        ["shot_id"],
    ),
    _function(
        "expand_context",
        "Expand around a grounded shot; radius zero accepts its adaptive proposal.",
        {"shot_id": {"type": "string"}, "radius": {"type": "integer", "minimum": 0, "maximum": 5}},
        ["shot_id"],
    ),
    _function(
        "request_audit",
        "Ask the evidence auditor to validate grounded candidate shots.",
        {"shot_ids": {"type": "array", "items": {"type": "string"}, "minItems": 1}},
        ["shot_ids"],
    ),
    _function(
        "submit",
        "Finish the episode with the selected grounded shots.",
        {"shot_ids": {"type": "array", "items": {"type": "string"}, "minItems": 1}},
        ["shot_ids"],
    ),
]


TOOL_RUNTIME_METADATA = {
    "search_transcript": ("TimelineScout", "subtitle-search", "local-bm25-srt", 1.0),
    "search_visual": ("TimelineScout", "visual-retrieval", "local-openclip-index", 2.0),
    "inspect_keyframe": ("VisionAnalyst", "keyframe-inspection", "local-keyframe-store", 4.0),
    "expand_context": ("TimelineScout", "timeline-context", "local-evidence-graph", 1.5),
    "request_audit": ("EvidenceAuditor", "evidence-audit", "local-rule-auditor", 2.0),
    "submit": ("Coordinator", "highlight-submit", "local-task-verifier", 0.0),
}


def wrap_offline_observation(tool: str, arguments: dict, observation: dict, task_id: str, call_index: int) -> dict:
    """Upgrade legacy teacher observations to the runtime gateway contract."""
    owner, service, backend, cost_units = TOOL_RUNTIME_METADATA[tool]
    episode_id = f"teacher-{task_id}-{uuid.uuid5(uuid.NAMESPACE_URL, task_id).hex[:8]}"
    return {
        "request_id": f"{episode_id}:{call_index:02d}",
        "episode_id": episode_id,
        "task_id": task_id,
        "tool": tool,
        "owner": owner,
        "service": service,
        "backend": backend,
        "status": "error" if observation.get("error") else "ok",
        "error_code": str(observation.get("error", "")).upper() or None,
        "latency_ms": 0.0,
        "cost_units": cost_units,
        "state_before": "teacher-offline",
        "state_after": "teacher-offline",
        "arguments": arguments,
        "observation": observation,
    }
