"""Business-equivalent tool gateway with auditable request/response traces.

The public reproduction cannot call former employer services.  This gateway
keeps the service boundary explicit while delegating to deterministic local
backends built from public subtitles, CLIP features and keyframes.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import uuid
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ToolServiceSpec:
    owner: str
    service: str
    backend: str
    cost_units: float


TOOL_SERVICE_SPECS = {
    "search_transcript": ToolServiceSpec("TimelineScout", "subtitle-search", "local-bm25-srt", 1.0),
    "search_visual": ToolServiceSpec("TimelineScout", "visual-retrieval", "local-openclip-index", 2.0),
    "inspect_keyframe": ToolServiceSpec("VisionAnalyst", "keyframe-inspection", "local-keyframe-store", 4.0),
    "expand_context": ToolServiceSpec("TimelineScout", "timeline-context", "local-evidence-graph", 1.5),
    "request_audit": ToolServiceSpec("EvidenceAuditor", "evidence-audit", "local-rule-auditor", 2.0),
    "submit": ToolServiceSpec("Coordinator", "highlight-submit", "local-task-verifier", 0.0),
}


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


class ToolGateway:
    """Dispatch tools through a stable service-like protocol.

    Default training is deterministic.  Latency is observed for diagnostics
    but is not injected into reward; optional fault injection is reserved for
    held-out robustness evaluation so GRPO groups remain comparable.
    """

    def __init__(self, env: Any, trace_dir: Path | None = None, fault_rate: float = 0.0):
        self.env = env
        self.task_id = str(getattr(env, "task", {}).get("task_id", "unknown"))
        self.episode_id = f"{self.task_id}-{uuid.uuid4().hex[:12]}"
        self.call_index = 0
        self.fault_rate = max(0.0, min(1.0, float(fault_rate)))
        self.trace_dir = trace_dir
        if self.trace_dir is not None:
            self.trace_dir.mkdir(parents=True, exist_ok=True)

    def _state_hash(self) -> str:
        payload = _jsonable(getattr(self.env, "state", {}))
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()[:16]

    def _should_fail(self, tool: str) -> bool:
        if self.fault_rate <= 0.0 or tool == "submit":
            return False
        key = f"{self.episode_id}:{self.call_index}:{tool}".encode("utf-8")
        draw = int(hashlib.sha256(key).hexdigest()[:8], 16) / 0xFFFFFFFF
        return draw < self.fault_rate

    def _write_trace(self, row: dict[str, Any]) -> None:
        if self.trace_dir is None:
            return
        rank = os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0"))
        path = self.trace_dir / f"tool_trace_rank{rank}_pid{os.getpid()}.jsonl"
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    def _mark_gateway_failure(self, invalid: bool) -> None:
        state = getattr(self.env, "state", None)
        if state is None:
            return
        if hasattr(state, "tool_calls"):
            state.tool_calls += 1
        if invalid and hasattr(state, "invalid_calls"):
            state.invalid_calls += 1
        if invalid and hasattr(state, "process_reward") and hasattr(self.env, "config"):
            state.process_reward += float(self.env.config.get("process_reward", {}).get("invalid_action", 0.0))

    def invoke(self, tool: str, arguments: dict[str, Any]) -> dict[str, Any]:
        self.call_index += 1
        request_id = f"{self.episode_id}:{self.call_index:02d}"
        spec = TOOL_SERVICE_SPECS.get(tool)
        started = time.perf_counter()
        state_before = self._state_hash()

        if spec is None:
            observation = {"error": "unknown_tool"}
            status, error_code = "error", "UNKNOWN_TOOL"
            self._mark_gateway_failure(invalid=True)
        elif self._should_fail(tool):
            observation = {"error": "simulated_service_unavailable", "retryable": True}
            status, error_code = "error", "SERVICE_UNAVAILABLE"
            self._mark_gateway_failure(invalid=False)
        else:
            try:
                observation = getattr(self.env, tool)(**arguments)
                status = "error" if observation.get("error") else "ok"
                error_code = str(observation.get("error", "")).upper() or None
            except (TypeError, ValueError) as error:
                self._mark_gateway_failure(invalid=True)
                observation = {"error": "invalid_arguments", "detail": str(error)}
                status, error_code = "error", "INVALID_ARGUMENTS"

        latency_ms = round((time.perf_counter() - started) * 1000.0, 3)
        service = spec or ToolServiceSpec("Unknown", "unknown", "none", 0.0)
        backend = "official-qv-clip-feature-proxy" if tool == "inspect_keyframe" and observation.get("feature_only") else service.backend
        response = {
            "request_id": request_id,
            "episode_id": self.episode_id,
            "task_id": self.task_id,
            "tool": tool,
            "owner": service.owner,
            "service": service.service,
            "backend": backend,
            "status": status,
            "error_code": error_code,
            "latency_ms": latency_ms,
            "cost_units": service.cost_units,
            "state_before": state_before,
            "state_after": self._state_hash(),
            "arguments": _jsonable(arguments),
            "observation": _jsonable(observation),
        }
        self._write_trace(response)
        return response
