#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="/home/rong/miniconda3/envs/aigc_det_baseline/bin/python"
DEFACTIFY_REVISION_VALUE="${DEFACTIFY_REVISION:-787334f7857fa54f29027a7f09c30e895ad486ef}"
HF_DATASETS_CACHE_DIR="${HF_DATASETS_CACHE:-/home/rong/.cache/huggingface/datasets}"

"$PYTHON_BIN" data_pipeline/defactify.py \
  --revision "$DEFACTIFY_REVISION_VALUE" \
  --cache-dir "$HF_DATASETS_CACHE_DIR"
