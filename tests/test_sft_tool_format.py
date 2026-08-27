import json
from pathlib import Path

from server.train_sft import normalize_tool_record


ROOT = Path(__file__).resolve().parents[1]


def test_sft_record_is_converted_to_native_tool_calls():
    record = json.loads((ROOT / "data/training/sft_train_v2.jsonl").open(encoding="utf-8").readline())
    normalized = normalize_tool_record(record)
    calls = [message for message in normalized["messages"] if "tool_calls" in message]
    responses = [message for message in normalized["messages"] if message["role"] == "tool"]
    assert calls
    assert len(calls) == len(responses) + 1
    assert calls[-1]["tool_calls"][0]["function"]["name"] == "submit"
    assert normalized["messages"][-1]["role"] == "assistant"
    assert all(message.get("name") for message in responses)
    assert {tool["function"]["name"] for tool in normalized["tools"]} >= {"search_visual", "submit"}
    text_blocks = [
        block["text"]
        for message in responses
        for block in message["content"]
        if block.get("type") == "text"
    ]
    assert all("request_id" in json.loads(text) and "observation" in json.loads(text) for text in text_blocks)
    serialized = json.dumps(normalized["messages"])
    assert "temporal_iou" not in serialized
    assert '"reward"' not in serialized
