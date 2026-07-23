# Cross-Dataset Robustness Research Status

- Research branch: `research/cross-dataset-robustness-v3`
- Frozen base branch: `feature/defactify-external-eval`
- Frozen base commit: `cedc2a18d956948acfed87d575d41d4b93689d1d`
- Current task: `Task 03`
- Last completed task: `Task 02`
- Protocol v3 frozen: `no`
- S1 specification frozen: `no`
- Model v3 frozen: `no`
- Research numbers frozen: `no`
- GenImage unseen accessed by research v3: `no`
- Defactify accessed by research v3: `no`
- Blocking issues: `none`

## Frozen Protocol v2 boundary

Task 01 verified every source and primary-report-artifact SHA256 recorded in
`artifacts/number_freeze_v2_manifest.json`. All hashes matched.

Protocol v2 remains authoritative for the existing Part 11–12 workflow.
Research v3 must not modify, replace, or reinterpret:

- `docs/FINAL_PROTOCOL.md`;
- `docs/FINAL_RESULTS_FREEZE.md`;
- `artifacts/number_freeze_v2_manifest.json`;
- `configs/final_model_contract_v2.yaml`;
- Part 07–10 registries and metric sources;
- the five frozen primary report artifacts;
- existing v2 checkpoints, feature caches, predictions, logs, and run
  directories.

## User-owned untracked boundary

Pre-existing untracked files other than the research plan are user-owned and
must not be modified, staged, moved, or deleted. At Task 01 start, their
sorted-path-list SHA256 was:

`1bae20b07e6476a16f4c0cd74ea087edf54cc8970365101bf8d6d94de3ccb03e`

## Task history

### Task 02 — Audit development data and propose source-aware LOGO

- Status: completed.
- Scope: GenImage train/validation only.
- Input rows: 3,000 train and 1,200 validation.
- Original sample-ID/exact-content/source-group overlaps: all zero.
- Proposed folds: hold out ADM, BigGAN, or Stable Diffusion V1.5.
- Fold validation rows: 400 each, balanced 200/200.
- Fold training rows: 1,999 / 2,000 / 2,000.
- Near-duplicate exclusions: one BigGAN fake row in the ADM holdout fold.
- Determinism: all six manifests and distribution CSV were byte-identical on
  an isolated second run.
- Tests: 43 unittest tests passed.
- Initial environment retries: system Python lacked pandas; the first conda
  test invocation used unavailable pytest. The user approved conversion to
  the repository's unittest style; no dependency was installed.
- Training/inference executed: no.
- GenImage unseen accessed: no.
- Defactify accessed: no.
- Decision: GO for Task 03 protocol design.

### Task 01 — Create isolated research branch and boundary snapshot

- Status: completed.
- Start branch: `feature/defactify-external-eval`.
- Start commit: `cedc2a18d956948acfed87d575d41d4b93689d1d`.
- Research branch created: `research/cross-dataset-robustness-v3`.
- Frozen hashes: all matched the number-freeze manifest.
- Training/inference executed: no.
- GenImage unseen accessed: no.
- Defactify accessed: no.
- Part 11 status modified: no.
- User-owned untracked files modified: no.

## Next boundary

Task 03 may write only the Protocol v3/S1 contract, machine-readable contract
metadata, and contract-validation tests. It must not implement or train a
model, and it ends at an explicit user-approval gate.
