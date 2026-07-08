#!/usr/bin/env bash
set -euo pipefail

python datasets/build_inventory.py --dataset CIFAKE
python datasets/make_splits.py --config configs/cifake_full_resnet50.yaml --dataset CIFAKE
python training/train.py --config configs/cifake_full_resnet50.yaml
python evaluation/evaluate.py \
  --config configs/cifake_full_resnet50.yaml \
  --checkpoint outputs/checkpoints/cifake_full_resnet50/best_model.pt

