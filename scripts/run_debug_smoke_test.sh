#!/usr/bin/env bash
set -euo pipefail

echo "Step 1/4: build CIFAKE inventory"
python datasets/build_inventory.py --dataset CIFAKE

echo "Step 2/4: generate CIFAKE debug splits"
python datasets/make_splits.py --config configs/debug_resnet18.yaml --dataset CIFAKE

echo "Step 3/4: train one debug epoch"
python training/train.py --config configs/debug_resnet18.yaml

echo "Step 4/4: evaluate debug checkpoint"
python evaluation/evaluate.py \
  --config configs/debug_resnet18.yaml \
  --checkpoint outputs/checkpoints/debug_resnet18/best_model.pt

