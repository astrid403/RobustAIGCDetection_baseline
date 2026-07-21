#!/usr/bin/env bash
set -euo pipefail
PYTHON_BIN="/home/rong/miniconda3/envs/aigc_det_baseline/bin/python"

ROOT="${WILDFAKE_ROOT:-data/raw/WildFake}"
mkdir -p "$ROOT"

if [ "${1:-}" = "--metadata" ]; then
  if [ -z "${2:-}" ]; then
    echo "Usage: bash scripts/prepare_wildfake.sh --metadata path/to/metadata.csv"
    exit 1
  fi
  "$PYTHON_BIN" data_pipeline/build_inventory.py --dataset WildFake --wildfake-metadata "$2"
else
  echo "Scanning folder-based WildFake layout under $ROOT"
  echo "If labels cannot be inferred from folder names, provide path,label CSV:"
  echo "bash scripts/prepare_wildfake.sh --metadata path/to/metadata.csv"
  "$PYTHON_BIN" data_pipeline/build_inventory.py --dataset WildFake
fi

"$PYTHON_BIN" data_pipeline/make_splits.py --config configs/debug_resnet18.yaml --dataset WildFake
