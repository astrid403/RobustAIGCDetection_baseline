#!/usr/bin/env bash
set -euo pipefail

python robustness/evaluate_robustness.py --config configs/robustness_eval.yaml

