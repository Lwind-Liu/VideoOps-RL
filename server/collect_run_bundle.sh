#!/usr/bin/env bash
set -uo pipefail

SOURCE_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
RUNTIME_CANDIDATE="$SOURCE_ROOT/.videoops-bootstrap/runtime/VideoOps-RL"
if [ -d "$RUNTIME_CANDIDATE" ]; then
  ROOT="$RUNTIME_CANDIDATE"
else
  ROOT="$SOURCE_ROOT"
fi
cd "$ROOT"

STAMP=$(date -u +%Y%m%dT%H%M%SZ)
HANDOFF_ROOT="$ROOT/outputs/handoff"
BUNDLE_NAME="VideoOps-RL-run-report-${STAMP}"
STAGING="$HANDOFF_ROOT/$BUNDLE_NAME"
ARCHIVE="$HANDOFF_ROOT/${BUNDLE_NAME}.tar.gz"

mkdir -p "$STAGING/logs" "$STAGING/reports" "$STAGING/metrics"

{
  echo "utc_time=$STAMP"
  echo "source_root=$SOURCE_ROOT"
  echo "run_root=$ROOT"
  echo "git_commit=$(git -C "$SOURCE_ROOT" rev-parse HEAD 2>/dev/null || echo unavailable)"
  echo "git_branch=$(git -C "$SOURCE_ROOT" branch --show-current 2>/dev/null || echo unavailable)"
  echo "git_status_begin"
  git -C "$SOURCE_ROOT" status --short 2>/dev/null || true
  echo "git_status_end"
  echo "uname=$(uname -a 2>/dev/null || echo unavailable)"
  echo "python=$(python --version 2>&1 || true)"
  echo "disk_begin"
  df -h "$ROOT" 2>&1 || true
  echo "disk_end"
  echo "nvidia_smi_begin"
  nvidia-smi 2>&1 || true
  echo "nvidia_smi_end"
} > "$STAGING/system_info.txt" 2>&1

python -m pip freeze > "$STAGING/pip_freeze.txt" 2>&1 || true

if [ -f "$SOURCE_ROOT/.videoops-bootstrap/bootstrap.log" ]; then
  cp "$SOURCE_ROOT/.videoops-bootstrap/bootstrap.log" "$STAGING/logs/"
fi

for LOG in outputs/run_all.log outputs/run_all_*.log outputs/vllm.log outputs/vllm_*.log outputs/eval_*.log; do
  if [ -f "$LOG" ]; then
    cp "$LOG" "$STAGING/logs/"
  fi
done

if [ -d outputs/reports ]; then
  find outputs/reports -maxdepth 1 -type f \( -name '*.json' -o -name '*.csv' -o -name '*.txt' \) \
    -exec cp {} "$STAGING/reports/" \;
fi

if [ -d outputs/metrics ]; then
  find outputs/metrics -maxdepth 1 -type f -name '*.jsonl' -exec cp {} "$STAGING/metrics/" \;
fi

if [ -d outputs/traces ]; then
  tar -czf "$STAGING/tool_traces.tar.gz" -C outputs traces
fi

{
  echo -e "bytes\tpath"
  if [ -d artifacts ]; then
    find artifacts -type f -printf '%s\t%p\n' | sort -k2
  fi
} > "$STAGING/artifact_inventory.tsv"

if [ -d artifacts ]; then
  find artifacts -type f -name '*.safetensors' -print0 | sort -z | \
    xargs -0 -r sha256sum > "$STAGING/checkpoint_sha256.txt"
else
  : > "$STAGING/checkpoint_sha256.txt"
fi

if [ -f PACKAGE_MANIFEST.json ]; then
  cp PACKAGE_MANIFEST.json "$STAGING/"
fi

tar -czf "$ARCHIVE" -C "$HANDOFF_ROOT" "$BUNDLE_NAME"
(
  cd "$HANDOFF_ROOT"
  sha256sum "${BUNDLE_NAME}.tar.gz" > "${BUNDLE_NAME}.tar.gz.sha256"
)

echo "Run report: $ARCHIVE"
echo "Checksum:   ${ARCHIVE}.sha256"
echo "Checkpoints are intentionally excluded; return their storage paths separately."
echo "Checkpoint hashes are recorded inside checkpoint_sha256.txt."
