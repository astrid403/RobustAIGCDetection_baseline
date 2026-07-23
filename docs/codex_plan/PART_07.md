# Part 07 — Frozen Three-seed Formal Training

## Goal and type

Train the frozen B2 baseline and user-approved final route with seeds 42, 43,
and 44 under identical formal policy. Type: formal training.

## Preconditions

- Part 6 user approval recorded in STATUS.
- `Model frozen: yes` and an exact final-config contract.
- Formal configs, output names, manifests, storage, and GPU availability are
  verified; no existing final directory would be overwritten.

## Allowed and forbidden scope

Add/fix formal configs and run registry only; code changes are limited to a
proven correctness bug and require stopping for approval. Run six formal
training jobs (reuse a seed-42 checkpoint only if provenance/config exactly
matches the frozen contract), validation, and checkpoint hashing.

Do not alter architecture, hyperparameters, threshold, augmentation, data,
checkpoint rule, or seeds; access Defactify; overwrite pilots; cherry-pick
epochs; merge; or push.

## Tasks and validation

1. Validate all formal configs against the freeze contract and name each
   `final_..._seed{42,43,44}_v2`.
2. Record commit, config hash, split hashes, environment, GPU, and commands
   before launch.
3. Train B2 and final route for each seed, monitoring only correctness/OOM.
4. Save best/last checkpoints according to the frozen validation rule.
5. Record exit status, validation metrics, runtime, checkpoint hash/size, and
   failure reason. Rerun only invalid jobs, never weak but valid jobs.
6. Audit completeness and cross-seed config equivalence.

## Outputs and acceptance

- Six valid formal training records/checkpoints, or a documented approved reuse.
- Complete validation metrics, logs, config snapshots, hashes, and timings.
- No Defactify evaluation and no architecture change.
- Lightweight registry/configs may be committed; checkpoints/features/large
  logs remain untracked.

## Stop, repair, and rollback

Stop on frozen-contract mismatch, overwrite, leakage, NaNs, corrupted cache,
or systemic failure. One operational retry per invalid job is allowed; a code
bug requires user approval and invalidates affected runs. Do not replace seeds.

## Git and handoff

Suggested commit:
`exp: register frozen three-seed training runs`.
No push/merge. Handoff lists every run ID, seed, config/commit/checkpoint hash,
validation metrics, runtime, status, and the exact checkpoints Part 8 may
evaluate.
