#!/usr/bin/env bash
set -euo pipefail
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
mkdir -p outputs outputs/reports
python server/preflight.py --required-gpus 8
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch --config_file configs/accelerate_zero2_8gpu.yaml server/train_sft.py
CUDA_VISIBLE_DEVICES=0 python server/merge_lora.py
CUDA_VISIBLE_DEVICES=6,7 trl vllm-serve --model artifacts/sft_qwen3vl2b_merged --tensor-parallel-size 2 --host 127.0.0.1 --port 8000 > outputs/vllm.log 2>&1 &
VLLM_PID=$!
trap 'kill "$VLLM_PID" 2>/dev/null || true' EXIT
for _ in $(seq 1 120); do
  if curl --fail --silent http://127.0.0.1:8000/v1/models >/dev/null; then break; fi
  if ! kill -0 "$VLLM_PID" 2>/dev/null; then tail -n 100 outputs/vllm.log; exit 1; fi
  sleep 5
done
curl --fail --silent http://127.0.0.1:8000/v1/models >/dev/null || { tail -n 100 outputs/vllm.log; exit 1; }
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 accelerate launch --config_file configs/accelerate_zero2_6gpu.yaml server/train_llm_grpo.py --sft-checkpoint artifacts/sft_qwen3vl2b_merged
kill "$VLLM_PID" 2>/dev/null || true
trap - EXIT
CUDA_VISIBLE_DEVICES=0 python server/evaluate_checkpoint.py --dataset all --split val --max-tasks 300
CUDA_VISIBLE_DEVICES=0 python server/evaluate_checkpoint.py --dataset all --split test --max-tasks 300

