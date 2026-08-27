#!/usr/bin/env bash
set -euo pipefail
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
mkdir -p outputs outputs/reports
SFT_EPOCHS=${SFT_EPOCHS:-3.0}
GRPO_STEPS=${GRPO_STEPS:-200}
EVAL_TASKS=${EVAL_TASKS:-300}
RUN_MODE=${RUN_MODE:-full}
RUN_ID=${VIDEOOPS_RUN_ID:-${RUN_MODE}_$(date -u +%Y%m%dT%H%M%SZ)}
export VIDEOOPS_RUN_ID="$RUN_ID"
export VIDEOOPS_TRACE_DIR="$PWD/outputs/traces/$RUN_ID"
ARTIFACT_ROOT=${ARTIFACT_ROOT:-artifacts}
SFT_DIR="$ARTIFACT_ROOT/sft_qwen3vl2b"
MERGED_DIR="$ARTIFACT_ROOT/sft_qwen3vl2b_merged"
GRPO_DIR="$ARTIFACT_ROOT/grpo_qwen3vl2b"
VLLM_LOG="outputs/vllm_${RUN_MODE}.log"
python server/preflight.py --required-gpus 8
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch --config_file configs/accelerate_zero2_8gpu.yaml \
  server/train_sft.py --output-dir "$SFT_DIR" --epochs "$SFT_EPOCHS" --gradient-accumulation-steps 4
CUDA_VISIBLE_DEVICES=0 python server/merge_lora.py --adapter "$SFT_DIR" --output "$MERGED_DIR"
ln -sfn "$(basename "$VLLM_LOG")" outputs/vllm.log
CUDA_VISIBLE_DEVICES=6,7 trl vllm-serve --model "$MERGED_DIR" --tensor-parallel-size 2 --host 127.0.0.1 --port 8000 > "$VLLM_LOG" 2>&1 &
VLLM_PID=$!
trap 'kill "$VLLM_PID" 2>/dev/null || true' EXIT
for _ in $(seq 1 120); do
  if curl --fail --silent http://127.0.0.1:8000/v1/models >/dev/null; then break; fi
  if ! kill -0 "$VLLM_PID" 2>/dev/null; then tail -n 100 "$VLLM_LOG"; exit 1; fi
  sleep 5
done
curl --fail --silent http://127.0.0.1:8000/v1/models >/dev/null || { tail -n 100 "$VLLM_LOG"; exit 1; }
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 accelerate launch --config_file configs/accelerate_zero2_6gpu.yaml \
  server/train_llm_grpo.py --sft-checkpoint "$MERGED_DIR" --output-dir "$GRPO_DIR" \
  --gradient-accumulation-steps 4 --num-generations 4 --max-steps "$GRPO_STEPS"
kill "$VLLM_PID" 2>/dev/null || true
trap - EXIT
CUDA_VISIBLE_DEVICES=0 python server/evaluate_checkpoint.py --model "$GRPO_DIR" --dataset all --split val --max-tasks "$EVAL_TASKS"
CUDA_VISIBLE_DEVICES=0 python server/evaluate_checkpoint.py --model "$GRPO_DIR" --dataset all --split test --max-tasks "$EVAL_TASKS"
python server/analyze_training_run.py --run-mode "$RUN_MODE" --run-id "$RUN_ID" --group-size 4 --require-traces
