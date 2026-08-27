"""Minimal JSONL logging callbacks shared by SFT and GRPO."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path


def build_metric_callback(stage: str, output_dir: Path):
    from transformers import TrainerCallback

    run_id = os.environ.get("VIDEOOPS_RUN_ID", "manual")
    path = output_dir / "metrics" / f"{stage}_{run_id}_metrics.jsonl"

    class JSONLMetricCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if not state.is_world_process_zero or not logs:
                return
            path.parent.mkdir(parents=True, exist_ok=True)
            row = {
                "stage": stage,
                "timestamp": time.time(),
                "step": state.global_step,
                "epoch": state.epoch,
                "metrics": logs,
            }
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    return JSONLMetricCallback()
