#!/usr/bin/env bash
set -euo pipefail
PYTHON_BIN="/home/rong/miniconda3/envs/aigc_det_baseline/bin/python"

"$PYTHON_BIN" robustness/evaluate_robustness.py --config configs/robustness_eval.yaml
