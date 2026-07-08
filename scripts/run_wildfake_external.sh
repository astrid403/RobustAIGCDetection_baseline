#!/usr/bin/env bash
set -euo pipefail

python datasets/build_inventory.py --dataset WildFake
python datasets/make_splits.py --config configs/wildfake_external_resnet50.yaml --dataset WildFake

if [ ! -f outputs/checkpoints/genimage_splitB_resnet50/best_model.pt ]; then
  echo "Missing GenImage-trained checkpoint: outputs/checkpoints/genimage_splitB_resnet50/best_model.pt"
  echo "Run bash scripts/run_genimage_splitB_resnet50.sh first."
  exit 1
fi

python evaluation/evaluate.py \
  --config configs/genimage_splitB_resnet50.yaml \
  --checkpoint outputs/checkpoints/genimage_splitB_resnet50/best_model.pt \
  --test-csv outputs/splits/wildfake_baseline_test.csv

