#!/usr/bin/env bash
set -euo pipefail
PYTHON_BIN="/home/rong/miniconda3/envs/aigc_det_baseline/bin/python"

ROOT="${GENIMAGE_ROOT:-data/raw/GenImage}"
mkdir -p "$ROOT"

echo "GenImage preparation helper"
echo "Expected root: $ROOT"
echo "You may place already extracted generator folders here, for example:"
echo "  $ROOT/ADM/train/ai"
echo "  $ROOT/ADM/train/nature"
echo "  $ROOT/ADM/val/ai"
echo "  $ROOT/ADM/val/nature"

if [ "${1:-}" = "--url" ]; then
  if [ -z "${2:-}" ]; then
    echo "Usage: bash scripts/prepare_genimage.sh --url URL"
    exit 1
  fi
  echo "Download URL support is intentionally manual in baseline v1."
  echo "Download from: $2"
  echo "Then extract into $ROOT and rerun inventory generation."
fi

"$PYTHON_BIN" data_pipeline/build_inventory.py --dataset GenImage
