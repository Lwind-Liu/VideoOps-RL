#!/usr/bin/env bash
set -euo pipefail

REPOSITORY=${VIDEOOPS_REPOSITORY:-Lwind-Liu/VideoOps-RL}
RELEASE_TAG=${VIDEOOPS_RELEASE_TAG:-offline-v2.0.0}
RUN_MODE=${RUN_MODE:-full}
INSTALL_DEPS=${INSTALL_DEPS:-1}
PREPARE_ONLY=${PREPARE_ONLY:-0}
WORK_DIR=${VIDEOOPS_WORK_DIR:-$PWD/.videoops-bootstrap}
PACKAGE_NAME=VideoOps-RL-offline-server.zip
PACKAGE_SHA256=cdd99e1467be9f0de4f311afd6dafe522c3a0b16cc8485b741ab622d1ecd4fe1
BASE_URL=${VIDEOOPS_BASE_URL:-"https://github.com/${REPOSITORY}/releases/download/${RELEASE_TAG}"}

mkdir -p "$WORK_DIR/downloads" "$WORK_DIR/runtime"

for PART in 00 01 02; do
  FILE="${PACKAGE_NAME}.part-${PART}"
  if [ ! -f "$WORK_DIR/downloads/$FILE" ]; then
    echo "Downloading $FILE ..."
    curl --fail --location --retry 8 --retry-all-errors \
      --continue-at - --output "$WORK_DIR/downloads/$FILE" "$BASE_URL/$FILE"
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
cd "$PROJECT_DIR"

if [ "$INSTALL_DEPS" = "1" ]; then
  python -m pip install --upgrade pip
  python -m pip install -r server/requirements-llm-grpo.txt
fi

if [ "$PREPARE_ONLY" = "1" ]; then
  echo "Assets prepared at $PROJECT_DIR"
  exit 0
fi

if [ "$RUN_MODE" = "smoke" ]; then
  export SFT_EPOCHS=${SFT_EPOCHS:-0.1}
  export GRPO_STEPS=${GRPO_STEPS:-20}
  export EVAL_TASKS=${EVAL_TASKS:-48}
elif [ "$RUN_MODE" != "full" ]; then
  echo "RUN_MODE must be 'smoke' or 'full'." >&2
  exit 2
fi

mkdir -p outputs
exec bash server/run_all.sh 2>&1 | tee outputs/run_all.log
