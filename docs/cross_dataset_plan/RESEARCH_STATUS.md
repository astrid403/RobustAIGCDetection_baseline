# Cross-Dataset Robustness Research Status

- Research branch: `research/cross-dataset-robustness-v3`
- Frozen base branch: `feature/defactify-external-eval`
- Frozen base commit: `cedc2a18d956948acfed87d575d41d4b93689d1d`
- Current task: `Awaiting explicit approval for F02`
- Last completed task: `Task F01`
- Protocol v3 frozen: `yes`
- S1 specification frozen: `yes`
- Model v3 frozen: `no`
- Research numbers frozen: `no`
- GenImage unseen accessed by research v3: `no`
- Defactify accessed by research v3: `no`
- Blocking issues: `none; F02 requires explicit approval`

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

### Task F01 — S1 failure audit and S2 preregistration

- Status: completed; stopped at the F02 approval gate.
- Evidence scope: read-only Task 08 GenImage development OOF results and
  registries. No image inference or training was run.
- Failure classification: primarily weak, generator-specific NPR local signal
  on held-out Stable Diffusion V1.5; secondarily score/class bias under fixed
  fusion. Degradation sensitivity was not dominant, measurable error
  complementarity existed but was insufficient, and no implementation or
  split-integrity defect was found.
- S2 route: frozen OpenAI CLIP ViT-B/32 blocks `[3,6,9,12]`, one-forward CLS
  extraction, shared `768->128` projection, sample-conditioned shared TIE,
  128-D weighted sum, and a single-logit classifier.
- Objective: BCE plus binary-label SupCon at fixed weight `0.1` and
  temperature `0.07`.
- Budget: batch 32, 2,000 balanced draws per epoch, 10 epochs, 20,000 draws
  and 630 optimizer steps per fold; seed 42 pilot.
- Gate: the frozen single-model Task 08 criteria versus same-fold B2-v3;
  all criteria are mandatory. No alpha, calibration, threshold, block,
  projection, loss-weight, augmentation, or external-data search is allowed.
- Contract: `configs/research_v3/S2_CONTRACT.yaml`.
- Explanation: `docs/cross_dataset_plan/S2_PREREGISTRATION.md`.
- S2 implemented: no.
- Training/inference executed: no.
- GenImage unseen accessed: no.
- Defactify accessed: no.

### Task 08 — Seed-42 three-fold LOGO pilot

- Status: completed with a preregistered **No-Go** decision.
- Runs: reused three successful B2-v3 and three successful NPR-only seed-42
  LOGO runs. All training and tee exit codes were zero; no training was rerun
  during the OOF analysis repair.
- OOF integrity: all six registries, best-checkpoint predictions, hashes,
  ordered sample IDs, folds, and labels passed strict checks. Each condition
  contains 1,200 unique OOF samples, 400 per held-out fold.
- Primary fusion: fixed B2/NPR probability alpha 0.5/0.5; the descriptive
  alpha grid did not replace the primary result.
- Clean fold-mean AUROC: B2 `0.966050`, NPR `0.950383`, fusion `0.962717`.
  Fusion-minus-B2 was `-0.003333`; clean worst-fold delta was `-0.015400`.
- Fold deltas: ADM `+0.002475`, BigGAN `+0.002925`, Stable Diffusion V1.5
  `-0.015400`; two of three folds improved.
- Degradations: fusion-minus-B2 mean fold-AUROC across JPEG/resize/blur was
  `+0.030631`, satisfying only the degradation criterion.
- Complementarity: 102 shared errors, 127 B2-wrong/NPR-right, 103
  NPR-wrong/B2-right, and 230 prediction disagreements among 1,200 clean
  samples. Fusion still trailed the stronger clean expert by `-0.003333`.
- Collapse check: clean fusion real recall was `0.988333`, but fake recall
  was `0.631667`, below the frozen `0.70` minimum.
- Analysis repair: the initial partial clean OOF directory was preserved.
  A regression-tested single-expert summary path was added; successful
  analysis used a new `retry1` directory without rerunning any pilot.
- Decision: No-Go because mean gain, worst-fold safety, stronger-expert
  margin, and fake-recall requirements failed. Tasks 09–13 are
  `skipped_by_gate`; S2 requires separate user approval.
- Evidence: `docs/cross_dataset_plan/S1_SEED42_DECISION.md` and
  `artifacts/research_v3/task08_seed42_manifest.json`.
- GenImage unseen accessed: no.
- Defactify accessed: no.

### Task 08 preflight — Repair pilot correctness infrastructure

- Status: completed without starting Task 08 pilot runs.
- Cache: B2-v3 uses isolated `research_clip_cache_v3` metadata and signatures
  covering encoder, pretrained tag, final feature mode, preprocessing,
  manifest SHA256, fold, split role, and extraction version.
- Outputs: B2/NPR checkpoints, logs, metrics, predictions, and registries are
  isolated under `outputs/research_v3`; any existing run directory hard fails.
- Checkpoint correctness: clean and degraded predictions are generated only
  after reloading the validation-AUROC-selected best checkpoint.
- Inference: both B2 and NPR support clean, JPEG quality 70, resize scale 0.5,
  and Gaussian blur radius 1.0 from the same held-out validation manifest.
- Prediction contract: `sample_id,fold,label,probability`, stable ordering, and
  unique sample IDs suitable for Task 07 strict OOF alignment.
- Provenance: each registry hashes the runtime Git commit, config snapshot,
  train/validation splits, best checkpoint, every degradation prediction, any
  CLIP feature caches, and ordered sample IDs.
- Configs: six seed-42 three-fold pilot configs match the frozen S1 contract
  and Task 02 split hashes.
- Correctness smoke: temporary synthetic images and tiny B2/NPR models
  verified best-checkpoint reload, all four degradations, schemas, hashes, and
  overwrite rejection.
- Tests: 94 unittest tests passed; one CUDA consistency test was skipped
  because CUDA is unavailable.
- Formal LOGO training/pilot executed: no.
- GenImage unseen accessed: no.
- Defactify accessed: no.
- Manifest: `artifacts/research_v3/task08_preflight_manifest.json`.

### Task 07 — Implement OOF fusion, metrics, and provenance correctness

- Status: completed; stopped at the Task 08 approval gate.
- Alignment: CLIP and NPR predictions require identical ordered sample IDs,
  labels, and held-out fold assignments. Duplicate, missing, reordered, or
  misaligned inputs hard fail.
- OOF safety: every sample ID is unique and therefore may occur in only one
  held-out fold; the expected fold set can be enforced explicitly.
- Primary fusion: fixed probability fusion with CLIP/NPR alpha 0.5/0.5 and
  threshold 0.5.
- Alpha grid: the frozen `[0,0.25,0.5,0.75,1]` grid is descriptive only and
  cannot replace the primary alpha. Descriptive ties are ordered by mean AUROC,
  distance to 0.5, then lower alpha.
- Metrics: overall and per-fold AUROC, AUPRC, balanced accuracy, macro-F1,
  real recall, fake recall, fold mean/worst, error overlap, disagreement, and
  complementary-error counts/rates.
- Provenance: registry hashes config, split manifests, checkpoints, both input
  predictions, fused predictions, metrics, and ordered sample IDs.
- Correctness evidence: artificial three-fold OOF data independently
  recomputed fusion, metrics, complementarity, grid ordering, and registry
  hashes.
- Tests: 89 unittest tests passed; one CUDA consistency test was skipped
  because CUDA is unavailable.
- Training/pilot executed: no.
- GenImage unseen accessed: no.
- Defactify accessed: no.

### Task 06 — Assemble and smoke-test the NPR training path

- Status: completed.
- Detector: torchvision ImageNet-pretrained ResNet18 with the standard
  three-channel stem and `Linear(512,1)` classifier; all 11,177,025 parameters
  are trainable.
- Objective/optimizer: BCEWithLogitsLoss and AdamW at the frozen S1 settings;
  consistency remains disabled.
- Sampler: deterministic equal replacement quotas over sorted
  `(generator,label)` groups, keyed by seed and epoch.
- Smoke scope: synthetic local data only, four training draws and four
  validation samples, seed 42, batch size 4, one epoch, workers 0.
- Determinism: two new isolated runs produced tensor-exact state dictionaries,
  identical losses, prediction probabilities, labels, sample order, and
  canonical metrics. Maximum parameter difference was 0.0 and canonical
  state SHA256 was
  `955572d2791efde6dce28f49e8f353d18ebedd3da47e9318d62c640d5be0a2e6`.
- Pipeline audit: both training processes and both tee processes exited 0;
  logs were non-empty; checkpoints reloaded; metrics and provenance were
  complete.
- Tests: 82 unittest tests passed; one CUDA consistency test was skipped
  because CUDA is unavailable.
- Resource check: smoke ran on CPU, so allocated GPU memory was 0 MiB.
- Manifest: `artifacts/research_v3/task06_smoke_manifest.json`.
- Earlier failed/inconclusive smoke records remain preserved and were not
  overwritten or deleted.
- GenImage unseen accessed: no.
- Defactify accessed: no.
- Scientific model selection performed: no.

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

Task 08 preflight correctness repair is complete, but none of its six pilot
runs has started. A new explicit user approval is required before executing
the seed-42 three-fold B2-v3 then NPR-only matrix. The pilots still may not
access GenImage unseen or Defactify.
