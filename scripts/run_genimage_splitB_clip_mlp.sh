#!/usr/bin/env bash
set -euo pipefail
PYTHON_BIN="/home/rong/miniconda3/envs/aigc_det_baseline/bin/python"

"$PYTHON_BIN" data_pipeline/build_inventory.py --dataset GenImage
"$PYTHON_BIN" data_pipeline/make_splits.py --config configs/genimage_splitB_clip_mlp.yaml --dataset GenImage
"$PYTHON_BIN" training/train.py --config configs/genimage_splitB_clip_mlp.yaml
"$PYTHON_BIN" evaluation/evaluate.py \
  --config configs/genimage_splitB_clip_mlp.yaml \
  --checkpoint outputs/checkpoints/genimage_splitB_clip_mlp/best_model.pt
