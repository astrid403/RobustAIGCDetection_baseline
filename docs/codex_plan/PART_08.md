# Part 08 — Formal GenImage and Defactify Evaluation

## Goal and type

Evaluate all frozen B2 and final-route seed checkpoints on GenImage unseen and
Defactify official full, derive balanced and per-generator results from the
same full predictions, and preserve provenance. Type: formal evaluation.

## Preconditions

- Six accepted checkpoints from Part 7.
- Evaluation code/metrics and threshold are frozen.
- Defactify full and balanced manifest hashes match Protocol v2.
- Formal output directories do not already exist.

## Allowed and forbidden scope

Run inference on GenImage unseen and Defactify official full for each frozen
checkpoint. Derive Defactify balanced and per-generator tables from each
official-full prediction file without new inference. Add only narrow result
aggregation/provenance fixes with tests if pre-authorized.

Do not tune, retrain, select models, change threshold, redesign after seeing
Defactify, run Defactify train/validation, or omit an unfavorable seed.

## Tasks and validation

1. Verify checkpoint/config/commit/split hashes and label contract.
2. Run one GenImage-unseen inference per model/seed.
3. Run one Defactify-full inference per model/seed.
4. Select the deterministic balanced sample IDs from each corresponding full
   prediction and compute balanced metrics; do not run a second clean inference.
5. Compute full and balanced per-generator metrics and worst-generator values.
6. Save all Protocol v2 metrics, confusion matrices, commands, timings, hashes,
   and row/sample counts.
7. Audit that every seed/model/scope exists exactly once and full-to-balanced
   sample IDs are consistent.

## Outputs and acceptance

- Formal metrics for 2 models × 3 seeds × GenImage unseen/Defactify full plus
  derived balanced/per-generator analyses.
- All commands exit 0; counts/hashes match; prediction labels use the frozen
  threshold; metrics recompute exactly; no tuning occurred.
- Full predictions remain uncommitted; lightweight metric tables/manifests may
  be committed.

## Stop, repair, and rollback

Stop on manifest/hash mismatch, incomplete/corrupt checkpoint, duplicate IDs,
wrong counts, output collision, or metric inconsistency. Permit one
deterministic recomputation after fixing an evaluation-only bug with documented
approval. Defactify performance never triggers retraining.

## Git and handoff

Suggested commit:
`exp: record formal cross-dataset evaluation`.
No push/merge. Handoff contains the complete run matrix, artifact hashes,
seed-level metrics, derived-analysis proof, failures, and inputs for Part 9.
