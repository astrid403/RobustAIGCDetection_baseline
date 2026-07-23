# Part 04 — Penultimate-only Implementation and Seed-42 Pilot

## Goal and type

Implement the pre-approved penultimate-only classifier and obtain one isolated
seed-42 pilot on GenImage validation/unseen using the Part 3 feature contract.
Type: focused code change plus pilot experiment.

## Preconditions

- Part 3 accepted; extraction location, shapes, normalization, and cache
  fingerprint are frozen.
- B2 and B1 configs/metrics are available.
- GenImage train/validation/unseen manifests match frozen hashes.

## Allowed and forbidden scope

Modify the CLIP feature classifier/factory, necessary training/evaluation
entry points, penultimate pilot config, and tests. Run tests, tiny smoke,
feature caching for GenImage only, one seed-42 penultimate pilot, and its
GenImage evaluation.

Do not implement fusion; access Defactify; tune repeatedly on unseen; change
splits/augmentation/threshold from the approved policy; or overwrite B1/B2,
smoke, or final outputs.

## Tasks and validation

1. Add a penultimate-only classifier using the established feature dimension
   and the same head family/budget policy planned for fair comparison.
2. Add a seed-42 `pilot_..._v2` config with explicit provenance and output
   isolation.
3. Unit-test construction, forward shape, checkpoint round trip, frozen CLIP,
   feature-mode routing, and B1/B2 compatibility.
4. Run a tiny smoke and stop if cache or shape checks fail.
5. Run exactly one seed-42 training pilot; select checkpoint using GenImage
   validation only.
6. Evaluate once on GenImage unseen and save complete Part 2 metrics, config,
   commands, timings, checkpoint/prediction hashes, and failure notes.

## Outputs and acceptance

- Code/tests/config committed; lightweight pilot metrics/registry may be
  committed, but checkpoint/features/full predictions/logs stay untracked.
- All commands exit 0; unit tests pass; smoke and pilot directories are
  distinct; metrics contain all required fields; artifacts map to commit and
  config; no Defactify access occurred.
- Result validity, not improvement, is the acceptance criterion.

## Stop, repair, and rollback

Stop for data/cache mismatch, NaNs, label reversal, overwrite risk, or repeated
unseen tuning. Allow one implementation repair and one rerun only when the
first run is invalid for a documented bug; never rerun merely for poor
performance. Do not switch route in this Part.

## Git and handoff

Suggested commit:
`feat: add penultimate CLIP pilot`.
No push/merge. Handoff provides config, commit, seed, commands, validation and
unseen metrics, parameter count, artifact paths/hashes, runtime, known issues,
and readiness for fusion implementation.
