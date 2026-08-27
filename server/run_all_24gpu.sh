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

python server/preflight.py --required-gpus 24

# Stage 1: data parallelism across all 24 H200s. Accumulation 1 keeps the
# effective batch (24) close to the 8-GPU path (8 x accumulation 4 = 32).
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23 \
  accelerate launch --config_file configs/accelerate_zero2_24gpu.yaml \
  server/train_sft.py --output-dir "$SFT_DIR" --epochs "$SFT_EPOCHS" --gradient-accumulation-steps 1

CUDA_VISIBLE_DEVICES=0 python server/merge_lora.py --adapter "$SFT_DIR" --output "$MERGED_DIR"

# Stage 2: the 2B model is small enough that TP=4 is preferable to TP=8;
# twenty GPUs remain available for the policy optimizer.
CUDA_VISIBLE_DEVICES=20,21,22,23 \
  trl vllm-serve --model "$MERGED_DIR" \
  --tensor-parallel-size 4 --gpu-memory-utilization 0.70 \
  --host 127.0.0.1 --port 8000 > "$VLLM_LOG" 2>&1 &
ln -sfn "$(basename "$VLLM_LOG")" outputs/vllm.log
VLLM_PID=$!
trap 'kill "$VLLM_PID" 2>/dev/null || true' EXIT
for _ in $(seq 1 120); do
  if curl --fail --silent http://127.0.0.1:8000/v1/models >/dev/null; then break; fi
  if ! kill -0 "$VLLM_PID" 2>/dev/null; then tail -n 100 "$VLLM_LOG"; exit 1; fi
  sleep 5
done
curl --fail --silent http://127.0.0.1:8000/v1/models >/dev/null || { tail -n 100 "$VLLM_LOG"; exit 1; }

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19 \
  accelerate launch --config_file configs/accelerate_zero2_20gpu.yaml \
  server/train_llm_grpo.py --sft-checkpoint "$MERGED_DIR" --output-dir "$GRPO_DIR" \
  --gradient-accumulation-steps 1 --num-generations 4 --max-steps "$GRPO_STEPS"

kill "$VLLM_PID" 2>/dev/null || true
trap - EXIT

# Stage 3: evaluate val and test concurrently, each split sharded over 12 GPUs.
EVAL_PIDS=()
for SHARD in $(seq 0 11); do
  CUDA_VISIBLE_DEVICES=$SHARD python server/evaluate_checkpoint.py \
    --model "$GRPO_DIR" --dataset all --split val --max-tasks "$EVAL_TASKS" --num-shards 12 --shard-index "$SHARD" \
    > "outputs/eval_${RUN_MODE}_val_shard_${SHARD}.log" 2>&1 &
  EVAL_PIDS+=("$!")
  TEST_GPU=$((SHARD + 12))
  CUDA_VISIBLE_DEVICES=$TEST_GPU python server/evaluate_checkpoint.py \
    --model "$GRPO_DIR" --dataset all --split test --max-tasks "$EVAL_TASKS" --num-shards 12 --shard-index "$SHARD" \
    > "outputs/eval_${RUN_MODE}_test_shard_${SHARD}.log" 2>&1 &
  EVAL_PIDS+=("$!")
done

cleanup_eval() {
  for PID in "${EVAL_PIDS[@]}"; do kill "$PID" 2>/dev/null || true; done
}
trap cleanup_eval EXIT
EVAL_FAILED=0
for PID in "${EVAL_PIDS[@]}"; do
  if ! wait "$PID"; then EVAL_FAILED=1; fi
done
trap - EXIT
if [ "$EVAL_FAILED" -ne 0 ]; then
  echo "At least one evaluation shard failed; inspect outputs/eval_*_shard_*.log." >&2
  exit 1
fi

python server/merge_eval_shards.py --dataset all --split val --num-shards 12
python server/merge_eval_shards.py --dataset all --split test --num-shards 12
python server/analyze_training_run.py --run-mode "$RUN_MODE" --run-id "$RUN_ID" --group-size 4 --require-traces
