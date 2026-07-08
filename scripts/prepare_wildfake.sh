#!/usr/bin/env bash
set -euo pipefail

ROOT="${WILDFAKE_ROOT:-data/raw/WildFake}"
mkdir -p "$ROOT"

if [ "${1:-}" = "--metadata" ]; then
  if [ -z "${2:-}" ]; then
    echo "Usage: bash scripts/prepare_wildfake.sh --metadata path/to/metadata.csv"
    exit 1
  fi
  python datasets/build_inventory.py --dataset WildFake --wildfake-metadata "$2"
else
  echo "Scanning folder-based WildFake layout under $ROOT"
  echo "If labels cannot be inferred from folder names, provide path,label CSV:"
  echo "bash scripts/prepare_wildfake.sh --metadata path/to/metadata.csv"
  python datasets/build_inventory.py --dataset WildFake
fi

python datasets/make_splits.py --config configs/debug_resnet18.yaml --dataset WildFake

