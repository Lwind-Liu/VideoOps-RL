#!/usr/bin/env bash
set -euo pipefail

REPOSITORY=${VIDEOOPS_REPOSITORY:-Lwind-Liu/VideoOps-RL}
RELEASE_TAG=${VIDEOOPS_RELEASE_TAG:-offline-v2.0.0}
RUN_MODE=${RUN_MODE:-auto}
INSTALL_DEPS=${INSTALL_DEPS:-1}
PREPARE_ONLY=${PREPARE_ONLY:-0}
WORK_DIR=${VIDEOOPS_WORK_DIR:-$PWD/.videoops-bootstrap}
PACKAGE_NAME=VideoOps-RL-offline-server.zip
PACKAGE_SHA256=cdd99e1467be9f0de4f311afd6dafe522c3a0b16cc8485b741ab622d1ecd4fe1
BASE_URL=${VIDEOOPS_BASE_URL:-"https://github.com/${REPOSITORY}/releases/download/${RELEASE_TAG}"}
ASSET_DIR=${VIDEOOPS_ASSET_DIR:-}
ASSET_SEARCH_ROOTS=${VIDEOOPS_ASSET_SEARCH_ROOTS:-"/root/code /root/input /input /mnt/data /mnt/oss /root/oss /dataset"}
MIN_FREE_GIB=${VIDEOOPS_MIN_FREE_GIB:-60}
export VIDEOOPS_MIN_FREE_GIB="$MIN_FREE_GIB"

mkdir -p "$WORK_DIR"
exec > >(tee -a "$WORK_DIR/bootstrap.log") 2>&1

for REQUIRED_COMMAND in python curl sha256sum awk tee tar; do
  command -v "$REQUIRED_COMMAND" >/dev/null || {
    echo "Missing required command: $REQUIRED_COMMAND" >&2
    exit 2
  }
done

python server/audit_one_click_contract.py

python - <<'PY'
import os
import shutil
import sys

if not ((3, 11) <= sys.version_info[:2] < (3, 13)):
    raise SystemExit(f"Python 3.11 or 3.12 is required; found {sys.version.split()[0]}")
try:
    import torch
except ImportError as error:
    raise SystemExit("The selected image must provide CUDA PyTorch before bootstrap.") from error
gpu_count = torch.cuda.device_count()
if not torch.cuda.is_available() or gpu_count < 8:
    raise SystemExit(f"At least 8 CUDA GPUs are required; detected {gpu_count}.")
free_gib = shutil.disk_usage(".").free / 1024**3
min_free_gib = int(os.environ["VIDEOOPS_MIN_FREE_GIB"])
if free_gib < min_free_gib:
    raise SystemExit(f"At least {min_free_gib} GiB free disk is required; found {free_gib:.2f} GiB.")
print(f"Host gate passed: Python {sys.version.split()[0]}, {gpu_count} GPUs, {free_gib:.2f} GiB free.")
PY

mkdir -p "$WORK_DIR/downloads" "$WORK_DIR/runtime"

asset_dir_valid() {
  local CANDIDATE=$1
  [ -d "$CANDIDATE" ] || return 1
  for PART in 00 01 02; do
    local FILE="${PACKAGE_NAME}.part-${PART}"
    local EXPECTED_SHA
    EXPECTED_SHA=$(awk -v file="$FILE" '$2 == file { print $1 }' release_manifest.sha256)
    [ -n "$EXPECTED_SHA" ] || return 1
    [ -f "$CANDIDATE/$FILE" ] || return 1
    echo "$EXPECTED_SHA  $CANDIDATE/$FILE" | sha256sum --check --status || return 1
  done
}

find_asset_dir() {
  local CANDIDATE
  for CANDIDATE in "$PWD/offline_assets" "$PWD" $ASSET_SEARCH_ROOTS; do
    if asset_dir_valid "$CANDIDATE"; then
      printf '%s\n' "$CANDIDATE"
      return 0
    fi
    if [ -d "$CANDIDATE" ]; then
      local CHILD
      for CHILD in "$CANDIDATE"/*; do
        [ -d "$CHILD" ] || continue
        if asset_dir_valid "$CHILD"; then
          printf '%s\n' "$CHILD"
          return 0
        fi
      done
    fi
  done
  return 1
}

RESOLVED_ASSET_DIR=""
if [ -n "$ASSET_DIR" ]; then
  if asset_dir_valid "$ASSET_DIR"; then
    RESOLVED_ASSET_DIR="$ASSET_DIR"
  else
    echo "VIDEOOPS_ASSET_DIR is set but does not contain valid release parts: $ASSET_DIR" >&2
    exit 3
  fi
else
  RESOLVED_ASSET_DIR=$(find_asset_dir || true)
fi
if [ -n "$RESOLVED_ASSET_DIR" ]; then
  echo "Using local offline assets from $RESOLVED_ASSET_DIR"
fi

for PART in 00 01 02; do
  FILE="${PACKAGE_NAME}.part-${PART}"
  EXPECTED_SHA=$(awk -v file="$FILE" '$2 == file { print $1 }' release_manifest.sha256)
  if [ -z "$EXPECTED_SHA" ]; then
    echo "Missing checksum for $FILE" >&2
    exit 3
  fi

  if [ ! -f "$WORK_DIR/downloads/$FILE" ] || \
     ! echo "$EXPECTED_SHA  $WORK_DIR/downloads/$FILE" | sha256sum --check --status; then
    if [ -n "$RESOLVED_ASSET_DIR" ]; then
      echo "Using uploaded offline asset $RESOLVED_ASSET_DIR/$FILE"
      cp "$RESOLVED_ASSET_DIR/$FILE" "$WORK_DIR/downloads/$FILE"
      continue
    fi
    echo "Downloading $FILE ..."
    if ! curl --fail --location --retry 8 --retry-all-errors \
      --continue-at - --output "$WORK_DIR/downloads/$FILE" "$BASE_URL/$FILE"; then
      echo "Resume failed for $FILE; retrying from byte zero ..."
      rm -f "$WORK_DIR/downloads/$FILE"
      curl --fail --location --retry 8 --retry-all-errors \
        --output "$WORK_DIR/downloads/$FILE" "$BASE_URL/$FILE"
    fi

    if ! echo "$EXPECTED_SHA  $WORK_DIR/downloads/$FILE" | sha256sum --check --status; then
      echo "Checksum mismatch for $FILE; retrying a clean download ..."
      rm -f "$WORK_DIR/downloads/$FILE"
      curl --fail --location --retry 8 --retry-all-errors \
        --output "$WORK_DIR/downloads/$FILE" "$BASE_URL/$FILE"
    fi
  fi
done

cp release_manifest.sha256 "$WORK_DIR/downloads/release_manifest.sha256"
(
  cd "$WORK_DIR/downloads"
  sha256sum --check release_manifest.sha256
)

if [ ! -f "$WORK_DIR/$PACKAGE_NAME" ] || \
   ! echo "$PACKAGE_SHA256  $WORK_DIR/$PACKAGE_NAME" | sha256sum --check --status; then
  cat "$WORK_DIR/downloads/${PACKAGE_NAME}.part-"* > "$WORK_DIR/$PACKAGE_NAME"
fi
echo "$PACKAGE_SHA256  $WORK_DIR/$PACKAGE_NAME" | sha256sum --check

if [ ! -f "$WORK_DIR/runtime/VideoOps-RL/PACKAGE_MANIFEST.json" ]; then
  python -m zipfile -e "$WORK_DIR/$PACKAGE_NAME" "$WORK_DIR/runtime"
fi

PROJECT_DIR="$WORK_DIR/runtime/VideoOps-RL"

# The Release owns large immutable assets. The small Git checkout owns the
# current executable code, so fixes do not require republishing 5.5 GiB.
for PATH_TO_OVERLAY in configs schemas scripts server src tests pyproject.toml README.md; do
  if [ -e "$PATH_TO_OVERLAY" ]; then
    cp -a "$PATH_TO_OVERLAY" "$PROJECT_DIR/"
  fi
done
mkdir -p "$PROJECT_DIR/data"
cp -a data/training "$PROJECT_DIR/data/"

cd "$PROJECT_DIR"

if [ "$INSTALL_DEPS" = "1" ]; then
  python -m pip install --upgrade pip
  python -c 'import torch; assert torch.cuda.is_available(), "The selected image must provide CUDA PyTorch before dependency installation."'
  python -m pip install -r server/requirements-llm-grpo.txt
  python -m pip install --no-build-isolation deepspeed==0.19.5
  python -m pip install --no-deps -e .
fi

if [ "$PREPARE_ONLY" = "1" ]; then
  echo "Assets prepared at $PROJECT_DIR"
  exit 0
fi

mkdir -p outputs

run_pipeline() {
  local MODE=$1
  local MODE_SFT_EPOCHS MODE_GRPO_STEPS MODE_EVAL_TASKS MODE_ARTIFACT_ROOT
  if [ "$MODE" = "smoke" ]; then
    MODE_SFT_EPOCHS=${SMOKE_SFT_EPOCHS:-${SFT_EPOCHS:-0.1}}
    MODE_GRPO_STEPS=${SMOKE_GRPO_STEPS:-${GRPO_STEPS:-20}}
    MODE_EVAL_TASKS=${SMOKE_EVAL_TASKS:-${EVAL_TASKS:-48}}
    MODE_ARTIFACT_ROOT=${SMOKE_ARTIFACT_ROOT:-artifacts/smoke}
  else
    MODE_SFT_EPOCHS=${FULL_SFT_EPOCHS:-${SFT_EPOCHS:-3.0}}
    MODE_GRPO_STEPS=${FULL_GRPO_STEPS:-${GRPO_STEPS:-200}}
    MODE_EVAL_TASKS=${FULL_EVAL_TASKS:-${EVAL_TASKS:-300}}
    MODE_ARTIFACT_ROOT=${FULL_ARTIFACT_ROOT:-artifacts}
  fi
  echo "Starting ${MODE} pipeline."
  RUN_MODE="$MODE" \
  SFT_EPOCHS="$MODE_SFT_EPOCHS" \
  GRPO_STEPS="$MODE_GRPO_STEPS" \
  EVAL_TASKS="$MODE_EVAL_TASKS" \
  ARTIFACT_ROOT="$MODE_ARTIFACT_ROOT" \
    bash server/run_all.sh 2>&1 | tee outputs/run_all.log "outputs/run_all_${MODE}.log"
}

case "$RUN_MODE" in
  auto)
    run_pipeline smoke
    echo "Smoke passed; continuing to full training automatically."
    run_pipeline full
    ;;
  smoke|full)
    run_pipeline "$RUN_MODE"
    ;;
  *)
    echo "RUN_MODE must be 'auto', 'smoke', or 'full'." >&2
    exit 2
    ;;
esac
