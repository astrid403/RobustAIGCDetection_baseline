# Part 01 — Freeze Baselines and Protocol v2

## Goal and type

Freeze the existing ResNet50 and CLIP-MLP baseline evidence and define the
dataset, threshold, artifact, naming, and Git rules for Protocol v2.
Type: read-only audit plus lightweight documentation. Status: completed.

## Preconditions

- Branch `feature/defactify-external-eval`.
- Known baseline HEAD `66fd5b09e5a86309d6b53c39820cd8bc9448b814`.
- Existing configs, splits, checkpoints, metrics, predictions, and manifests
  are readable; pre-existing untracked files are inventoried and untouched.

## Allowed and forbidden scope

Read the full repository and Git metadata. Add or update only lightweight
protocol and artifact-manifest documents. Run read-only Git, filesystem,
hashing, CSV/YAML/JSON inspection, and tiny validation commands.

Do not modify model/training/evaluation/data logic; train or formally evaluate;
install/download; delete, move, or process untracked files; commit large
artifacts; merge; or push.

## Tasks and validation

1. Verify branch, HEAD, worktree, tracking state, and all baseline assets.
   Stop on a mismatched tracked worktree or missing mandatory artifact.
2. Freeze B0 ResNet50 and B2 CLIP-MLP configs, metrics, checkpoints, splits,
   manifests, environment, and SHA256 values. Recompute hashes and fail on any
   mismatch.
3. Define roles for GenImage train/validation/seen/unseen, Defactify full and
   balanced, CIFAKE, and WildFake.
4. State that Defactify cannot select structure, hyperparameters, augmentation,
   checkpoint, calibration, or threshold.
5. Define `smoke`, `pilot`, and `final` naming/output isolation.
6. Review diff and ensure only approved lightweight files changed.

## Outputs and acceptance

- `docs/FINAL_PROTOCOL.md`.
- `artifacts/baseline_v2_manifest.json`.
- All mandatory artifact paths exist and recorded hashes verify.
- Git diff contains no code, data, model, prediction, cache, or large log.
- `git diff --check` passes.

## Stop, repair, and rollback

Stop immediately for wrong branch/HEAD, dirty tracked files, missing artifacts,
or irreconcilable hash mismatch. At most two documentation-only repair rounds.
Never regenerate an artifact in this Part. Roll back only newly introduced
Part 1 changes, never user files, and request approval if baseline evidence
cannot be frozen.

## Git and handoff

Commit only after acceptance with `docs: freeze baseline and protocol v2`.
Never push or merge. Handoff must include branch, commit, verified hashes,
frozen baselines, protocol decisions, known issues, and readiness for Part 2.

Recorded completion: commit
`7f4aac29fdd69d713eea282aa1d521144cdcba60`.
