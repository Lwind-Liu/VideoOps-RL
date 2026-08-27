import json
from pathlib import Path

from videoops_rl.dataset_protocol import read_jsonl
from videoops_rl.multivideo_env import MultiVideoHighlightEnv
from videoops_rl.tool_gateway import ToolGateway

ROOT = Path(__file__).resolve().parents[1]


def test_gateway_wraps_local_backend_as_auditable_service(tmp_path):
    task = read_jsonl(ROOT / "data/registry/formal_tasks_v1.jsonl")[0]
    env = MultiVideoHighlightEnv(ROOT, task)
    gateway = ToolGateway(env, trace_dir=tmp_path)
    response = gateway.invoke("search_transcript", {"query": task["query"], "top_k": 3})
    assert response["tool"] == "search_transcript"
    assert response["owner"] == "TimelineScout"
    assert response["backend"] == "local-bm25-srt"
    assert response["state_before"] != response["state_after"]
    assert response["status"] in {"ok", "error"}
    trace_rows = [json.loads(line) for path in tmp_path.glob("*.jsonl") for line in path.read_text(encoding="utf-8").splitlines()]
    assert trace_rows[0]["request_id"] == response["request_id"]


def test_gateway_rejects_unknown_tool_without_exposing_labels():
    task = read_jsonl(ROOT / "data/registry/formal_tasks_v1.jsonl")[0]
    env = MultiVideoHighlightEnv(ROOT, task)
    response = ToolGateway(env).invoke("read_ground_truth", {})
    assert response["status"] == "error"
    assert response["error_code"] == "UNKNOWN_TOOL"
    assert "target_segments" not in json.dumps(response)
