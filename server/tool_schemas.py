"""Canonical JSON schemas shared by tool-calling SFT and runtime checks."""

from __future__ import annotations


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
