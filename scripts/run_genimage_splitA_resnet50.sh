#!/usr/bin/env bash
set -euo pipefail

python datasets/build_inventory.py --dataset GenImage
python datasets/make_splits.py --config configs/genimage_splitA_resnet50.yaml --dataset GenImage
python training/train.py --config configs/genimage_splitA_resnet50.yaml
python evaluation/evaluate.py \
  --config configs/genimage_splitA_resnet50.yaml \
  --checkpoint outputs/checkpoints/genimage_splitA_resnet50/best_model.pt

