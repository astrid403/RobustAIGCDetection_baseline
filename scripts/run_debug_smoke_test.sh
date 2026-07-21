#!/usr/bin/env bash
set -euo pipefail
PYTHON_BIN="/home/rong/miniconda3/envs/aigc_det_baseline/bin/python"

echo "Step 1/4: build CIFAKE inventory"
"$PYTHON_BIN" data_pipeline/build_inventory.py --dataset CIFAKE

echo "Step 2/4: generate CIFAKE debug splits"
"$PYTHON_BIN" data_pipeline/make_splits.py --config configs/debug_resnet18.yaml --dataset CIFAKE

echo "Step 3/4: train one debug epoch"
"$PYTHON_BIN" training/train.py --config configs/debug_resnet18.yaml

echo "Step 4/4: evaluate debug checkpoint"
"$PYTHON_BIN" evaluation/evaluate.py \
  --config configs/debug_resnet18.yaml \
  --checkpoint outputs/checkpoints/debug_resnet18/best_model.pt
