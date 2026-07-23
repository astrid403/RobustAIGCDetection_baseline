# Part 02 — Complete Metrics, Tests, and CLIP Linear Probe

## Goal and type

Unify required binary-detection metrics, make threshold configurable with
default 0.5, preserve legacy calls/results, and add B1: frozen final CLIP
embedding plus one linear head. Type: code modification and tiny smoke.
Status: completed.

## Preconditions

- Part 1 commit is current or an ancestor.
- `docs/FINAL_PROTOCOL.md` and `artifacts/baseline_v2_manifest.json` exist.
- Original untracked files are inventoried and remain untouched.

## Allowed and forbidden scope

Allowed: `evaluation/metrics.py`, `evaluation/evaluate.py`,
`evaluation/summarize_predictions.py` only as needed,
`models/clip_mlp_detector.py` only for linear-probe support,
`training/train.py` only for the entry point, B1 configs, and tests.
Run unit tests and a CPU/tiny tensor smoke only.

Do not modify Protocol v2, penultimate extraction, fusion, datasets, formal
outputs, or existing artifacts. Do not train, evaluate Defactify, merge, or
push.

## Tasks and validation

1. Implement accuracy, balanced accuracy, precision, fake/real recall, binary
   and macro F1, AUROC, AUPRC, and a fixed-order 2x2 confusion matrix.
2. Support configurable threshold and retain 0.5 by default.
3. Retain `recall` and `f1` compatibility aliases and B2 loading/calls.
4. Define explicit single-class behavior.
5. Add B1 linear classifier using the same frozen normalized final embedding.
6. Test perfect/wrong predictions, imbalance, recall direction, F1 variants,
   AUPRC, threshold changes, single class, invalid input, B1, and B2.

## Outputs and acceptance

- Metric/evaluation/training/model changes within allowed scope.
- B1 config and complete unit tests.
- All tests pass; B1/B2 tensor smoke passes; no formal output is created or
  overwritten; Protocol v2 has no diff; `git diff --check` passes.

## Stop, repair, and rollback

Stop for metric-label reversal, legacy checkpoint incompatibility, protocol
change, or any formal run. Permit at most two local code/test repair rounds.
Do not expand to penultimate/fusion. Roll back only Part 2 changes if backward
compatibility cannot be preserved and request approval.

## Git and handoff

Commit accepted files with
`feat: add complete detection metrics and CLIP linear probe`; no push/merge.
Handoff records definitions, single-class behavior, compatibility, B1 config,
tests, and known limitations.

Recorded completion: commit
`fdc149acb1e203a66bb30d3ffc21d39be36dd12a`; 18 unit tests and B1/B2 tiny smoke
passed.
