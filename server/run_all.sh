#!/usr/bin/env bash
set -euo pipefail

GPU_COUNT=$(python -c 'import torch; print(torch.cuda.device_count())')
if [ "$GPU_COUNT" -ge 24 ]; then
  echo "Detected ${GPU_COUNT} GPUs: selecting the 24-GPU fast path."
  exec bash server/run_all_24gpu.sh
elif [ "$GPU_COUNT" -ge 8 ]; then
  echo "Detected ${GPU_COUNT} GPUs: selecting the 8-GPU compatibility path."
  exec bash server/run_all_8gpu.sh
else
  echo "VideoOps-RL requires at least 8 CUDA GPUs; detected ${GPU_COUNT}." >&2
  exit 1
fi

