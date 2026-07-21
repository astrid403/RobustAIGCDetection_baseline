#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="/home/rong/miniconda3/envs/aigc_det_baseline/bin/python"
SPLIT_CSV="outputs/splits/defactify_smoke_test.csv"
RESNET_CHECKPOINT="outputs/checkpoints/genimage_splitB_available_resnet50/best_model.pt"
CLIP_CHECKPOINT="outputs/checkpoints/genimage_splitB_available_clip_mlp/best_model.pt"

if [ ! -f "$SPLIT_CSV" ]; then
  echo "Missing $SPLIT_CSV. Run scripts/prepare_defactify.sh after pinning DEFACTIFY_REVISION."
  exit 1
fi
for checkpoint in "$RESNET_CHECKPOINT" "$CLIP_CHECKPOINT"; do
  if [ ! -f "$checkpoint" ]; then
    echo "Missing checkpoint: $checkpoint"
    exit 1
  fi
done

"$PYTHON_BIN" evaluation/evaluate.py \
  --config configs/defactify_smoke_resnet50.yaml \
  --checkpoint "$RESNET_CHECKPOINT"

"$PYTHON_BIN" evaluation/evaluate.py \
  --config configs/defactify_smoke_clip_mlp.yaml \
  --checkpoint "$CLIP_CHECKPOINT"

"$PYTHON_BIN" robustness/evaluate_robustness.py \
  --config configs/defactify_smoke_robustness_resnet50.yaml

"$PYTHON_BIN" robustness/evaluate_robustness.py \
  --config configs/defactify_smoke_robustness_clip_mlp.yaml
