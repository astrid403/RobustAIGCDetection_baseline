#!/usr/bin/env bash
set -euo pipefail
PYTHON_BIN="/home/rong/miniconda3/envs/aigc_det_baseline/bin/python"

"$PYTHON_BIN" data_pipeline/build_inventory.py --dataset CIFAKE
"$PYTHON_BIN" data_pipeline/make_splits.py --config configs/cifake_full_resnet50.yaml --dataset CIFAKE
"$PYTHON_BIN" training/train.py --config configs/cifake_full_resnet50.yaml
"$PYTHON_BIN" evaluation/evaluate.py \
  --config configs/cifake_full_resnet50.yaml \
  --checkpoint outputs/checkpoints/cifake_full_resnet50/best_model.pt
