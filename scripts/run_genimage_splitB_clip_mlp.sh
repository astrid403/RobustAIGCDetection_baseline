#!/usr/bin/env bash
set -euo pipefail

python datasets/build_inventory.py --dataset GenImage
python datasets/make_splits.py --config configs/genimage_splitB_clip_mlp.yaml --dataset GenImage
python training/train.py --config configs/genimage_splitB_clip_mlp.yaml
python evaluation/evaluate.py \
  --config configs/genimage_splitB_clip_mlp.yaml \
  --checkpoint outputs/checkpoints/genimage_splitB_clip_mlp/best_model.pt

