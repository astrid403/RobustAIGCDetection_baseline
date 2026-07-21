#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="/home/rong/miniconda3/envs/aigc_det_baseline/bin/python"
SPLIT_CSV="outputs/splits/defactify_official_test.csv"
BALANCED_SPLIT_CSV="outputs/splits/defactify_balanced_test.csv"
RESNET_CHECKPOINT="outputs/checkpoints/genimage_splitB_available_resnet50/best_model.pt"
CLIP_CHECKPOINT="outputs/checkpoints/genimage_splitB_available_clip_mlp/best_model.pt"

for split_csv in "$SPLIT_CSV" "$BALANCED_SPLIT_CSV"; do
  if [ ! -f "$split_csv" ]; then
    echo "Missing $split_csv. Run scripts/prepare_defactify.sh first."
    exit 1
  fi
done
for checkpoint in "$RESNET_CHECKPOINT" "$CLIP_CHECKPOINT"; do
  if [ ! -f "$checkpoint" ]; then
    echo "Missing checkpoint: $checkpoint"
    exit 1
  fi
done

"$PYTHON_BIN" evaluation/evaluate.py \
  --config configs/defactify_external_available_resnet50.yaml \
  --checkpoint "$RESNET_CHECKPOINT"

"$PYTHON_BIN" evaluation/evaluate.py \
  --config configs/defactify_external_available_clip_mlp.yaml \
  --checkpoint "$CLIP_CHECKPOINT"

"$PYTHON_BIN" evaluation/evaluate.py \
  --config configs/defactify_balanced_available_resnet50.yaml \
  --checkpoint "$RESNET_CHECKPOINT"

"$PYTHON_BIN" evaluation/evaluate.py \
  --config configs/defactify_balanced_available_clip_mlp.yaml \
  --checkpoint "$CLIP_CHECKPOINT"
