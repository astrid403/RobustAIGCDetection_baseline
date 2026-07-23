# Cross-Dataset Robustness Research Status

- Research branch: `research/cross-dataset-robustness-v3`
- Frozen base branch: `feature/defactify-external-eval`
- Frozen base commit: `cedc2a18d956948acfed87d575d41d4b93689d1d`
- Current task: `Awaiting explicit Task 06 approval`
- Last completed task: `Task 05`
- Protocol v3 frozen: `yes`
- S1 specification frozen: `yes`
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

### Task 05 — Implement deterministic semantic-breaking and paired degradations

- Status: completed.
- Implementation: `data_pipeline/research_transforms.py`.
- Tests: `tests/test_research_transforms.py`.
- Semantic breaking: NPR-only 4x4 patch shuffle, driven solely by global seed,
  epoch, sample ID, and transform purpose.
- Paired views: clean/degraded views retain the same label, sample ID, flip,
  and patch permutation.
- Degradations: fixed JPEG quality 70, resize scale 0.5, and Gaussian blur
  radius 1.0.
- Evaluation: clean path is non-random by default.
- RNG/provenance: local generators avoid global Torch RNG drift, and the full
  transform contract is exposed through provenance.
- Tests: 74 unittest tests passed; one CUDA consistency test was skipped
  because CUDA is unavailable to the sandbox.
- Full detector implemented: no.
- Training/inference executed: no.
- GenImage unseen accessed: no.
- Defactify accessed: no.
- Old dataset APIs changed: no.

### Task 04 — Implement and verify the NPR input representation

- Status: completed.
- Implementation: `models/npr_detector.py`.
- Tests: `tests/test_npr_detector.py`.
- Contract: CHW/BCHW float RGB `[0,1]`, exact 224x224 input, signed
  down-right diagonal difference, no padding, `[3,223,223]` output, fixed
  range mapping and ImageNet normalization.
- Analytic tests: constant, horizontal gradient, vertical gradient, and
  checkerboard passed.
- Safety tests: shape, dtype, range, finite, determinism, no in-place input
  mutation, batch/single equivalence, autograd, PIL RGB conversion, and
  normalization order passed.
- Tests: 64 unittest tests passed; one CUDA consistency test was skipped
  because CUDA is unavailable to the sandbox.
- Full detector implemented: no.
- Training/inference executed: no.
- GenImage unseen accessed: no.
- Defactify accessed: no.
- Old detector APIs changed: no.

### Task 03 — Freeze Protocol v3 and the S1 specification

- Status: completed and explicitly approved by the user.
- Protocol: `docs/cross_dataset_plan/PROTOCOL_V3.md`.
- Machine contract: `configs/research_v3/s1_contract.yaml`.
- Candidate manifest:
  `artifacts/research_v3/protocol_v3_candidate_manifest.json`.
- Approval record: `artifacts/research_v3/protocol_v3_approval.json`.
- NPR representation: signed diagonal down-right difference, no padding,
  output `[3,223,223]`.
- NPR expert: ImageNet-pretrained ResNet18, all parameters trainable.
- Main semantic breaking: 4x4 patch shuffle at probability 0.5.
- Main consistency loss: disabled.
- Primary fusion: fixed CLIP/NPR alpha 0.5/0.5.
- Seeds: 42 pilot; 42/43/44 confirmation.
- Tests: 53 unittest tests passed, including 10 contract tests.
- Training/inference executed: no.
- GenImage unseen accessed: no.
- Defactify accessed: no.
- Gate: user approved the exact candidate protocol/contract hashes at commit
  `caf45034542ae35090977347b9685ab132e34754`; Task 04 is authorized.

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

Task 05 is complete. Task 06 is not authorized by the current request and must
not start without explicit user approval. Task 06 is the first task allowed to
assemble the NPR expert and training path and run only its prescribed tiny
smoke; it still may not access GenImage unseen or Defactify.
