"""Stage 2: mixed formal/QVHighlights tool-using GRPO (server only)."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from videoops_rl.dataset_protocol import read_jsonl
from videoops_rl.multivideo_env import MultiVideoHighlightEnv
from videoops_rl.qv_env import QVHighlightsEnv
from videoops_rl.tool_gateway import ToolGateway

try:
    from server.metric_logging import build_metric_callback
except ModuleNotFoundError:
    from metric_logging import build_metric_callback


class VideoOpsGRPOEnvironment:
    """Stateful environment instantiated independently for each rollout."""

    def __init__(self):
        import yaml
        self.formal_tasks = [item for item in read_jsonl(ROOT / "data/registry/formal_tasks_v1.jsonl") if item["split"] == "train"]
        self.qv_tasks = read_jsonl(ROOT / "data/external/qvhighlights/annotations/tasks_train_v1.jsonl")
        self.tasks_by_id = {item["task_id"]: item for item in self.formal_tasks + self.qv_tasks}
        config = yaml.safe_load((ROOT / "configs/algorithm_v2.yaml").read_text(encoding="utf-8"))
        self.formal_ratio = float(config["training"]["formal_sampling_ratio"])
        self.max_tool_steps = int(config["training"]["max_tool_steps"])
        self.env: MultiVideoHighlightEnv | QVHighlightsEnv | None = None
        self.gateway: ToolGateway | None = None
        self.final_reward = -1.0

    def reset(self, task_id: str | None = None, **_: object) -> str | None:
        # With an external dataset, TRL repeats one row for all G rollouts in a
        # group. Resolving the row's task_id here guarantees a shared initial
        # state and a meaningful group-relative baseline.
        if task_id is not None:
            task = self.tasks_by_id[task_id]
        elif random.random() < self.formal_ratio:
            task = random.choice(self.formal_tasks)
        else:
            task = random.choice(self.qv_tasks)
        if task["video_id"].startswith("qvh:"):
            self.env = QVHighlightsEnv(ROOT, task, self.max_tool_steps)
        else:
            self.env = MultiVideoHighlightEnv(ROOT, task, self.max_tool_steps)
        trace_dir = Path(os.environ.get("VIDEOOPS_TRACE_DIR", ROOT / "outputs/traces/grpo"))
        fault_rate = float(os.environ.get("VIDEOOPS_TOOL_FAULT_RATE", "0"))
        self.gateway = ToolGateway(self.env, trace_dir=trace_dir, fault_rate=fault_rate)
        self.final_reward = -1.0
        return None if task_id is not None else json.dumps(self.env.public_prompt, ensure_ascii=False)

    def _invoke(self, tool: str, arguments: dict) -> dict:
        if self.gateway is None:
            raise RuntimeError("reset must be called before invoking a tool")
        return self.gateway.invoke(tool, arguments)

    def search_transcript(self, query: str, top_k: int = 3) -> str:
        """Search subtitle evidence.

        Args:
            query: Natural-language search query.
            top_k: Maximum number of candidate shots.
        """
        return json.dumps(self._invoke("search_transcript", {"query": query, "top_k": top_k}), ensure_ascii=False)

    def search_visual(self, query: str, top_k: int = 3) -> str:
        """Search visual evidence and decode temporal candidates.

        Args:
            query: Natural-language visual description.
            top_k: Maximum number of candidate shots.
        """
        return json.dumps(self._invoke("search_visual", {"query": query, "top_k": top_k}), ensure_ascii=False)

    def inspect_keyframe(self, shot_id: str) -> list[dict]:
        """Inspect one candidate keyframe.

        Args:
            shot_id: Candidate shot identifier returned by search.
        """
        from PIL import Image
        response = self._invoke("inspect_keyframe", {"shot_id": shot_id})
        path = response["observation"].get("keyframe_path")
        content = [{"type": "text", "text": json.dumps(response, ensure_ascii=False)}]
        if path:
            content.insert(0, {"type": "image", "image": Image.open(ROOT / path).convert("RGB")})
        return content

    def expand_context(self, shot_id: str, radius: int = 1) -> str:
        """Get neighboring shots around a candidate.

        Args:
            shot_id: Center shot identifier.
            radius: Zero accepts an adaptive proposal; positive values request fixed neighbors.
        """
        return json.dumps(self._invoke("expand_context", {"shot_id": shot_id, "radius": radius}), ensure_ascii=False)

    def request_audit(self, shot_ids: list[str]) -> str:
        """Ask the evidence auditor to validate a selection.

        Args:
            shot_ids: Proposed grounded shot identifiers.
        """
        return json.dumps(self._invoke("request_audit", {"shot_ids": shot_ids}), ensure_ascii=False)

    def submit(self, shot_ids: list[str]) -> str:
        """Finish the episode and submit selected shots.

        Args:
            shot_ids: Final grounded shot identifiers.
        """
        response = self._invoke("submit", {"shot_ids": shot_ids})
        result = response["observation"]
        self.final_reward = float(result["reward"])
        return json.dumps(response, ensure_ascii=False)

    def get_reward(self) -> float:
        return self.final_reward if self.env and self.env.state.done else -1.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=str(ROOT / "models/Qwen3-VL-2B-Instruct"))
    parser.add_argument("--sft-checkpoint", default=str(ROOT / "artifacts/sft_qwen3vl2b_merged"))
    parser.add_argument("--output-dir", default=str(ROOT / "artifacts/grpo_qwen3vl2b"))
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--num-generations", type=int, default=4)
    args = parser.parse_args()
    from datasets import load_dataset
    from transformers import AutoProcessor
    from trl import GRPOConfig, GRPOTrainer

    model_path = args.sft_checkpoint if Path(args.sft_checkpoint).exists() else args.model
    config = GRPOConfig(
        output_dir=args.output_dir, max_steps=args.max_steps, learning_rate=1e-6,
        per_device_train_batch_size=1, gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_generations=args.num_generations, bf16=True, gradient_checkpointing=True,
        logging_steps=1, save_steps=50, report_to="none",
        use_vllm=True, vllm_mode="server", vllm_server_host="127.0.0.1", vllm_server_port=8000,
    )
    dataset = load_dataset("json", data_files=str(ROOT / "data/training/grpo_train_v2.jsonl"), split="train")
    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
    metric_root = ROOT / "outputs"
    trainer = GRPOTrainer(
        model=model_path,
        args=config,
        train_dataset=dataset,
        processing_class=processor,
        environment_factory=VideoOpsGRPOEnvironment,
        callbacks=[build_metric_callback("grpo", metric_root)],
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    trainer.save_state()
    processor.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
