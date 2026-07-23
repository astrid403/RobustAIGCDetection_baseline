# Final Experimental Protocol v2

## 1. Frozen baseline snapshot

This protocol freezes the completed baseline at Git commit
`66fd5b09e5a86309d6b53c39820cd8bc9448b814` on branch
`feature/defactify-external-eval`. Exact artifact hashes are recorded in
`artifacts/baseline_v2_manifest.json`.

The binary label contract is unchanged:

- `0`: real
- `1`: fake / AI-generated

The two primary frozen baselines were trained on the same GenImage fallback
split with seed 42:

| Baseline | Model | Train generators | Validation generators | Epochs |
|---|---|---|---|---:|
| B0 | ImageNet-pretrained ResNet50, end-to-end | Stable Diffusion V1.5, ADM, BigGAN | Same three generators, disjoint manifest rows | 5 |
| B2 | Frozen OpenAI CLIP ViT-B/32 final image embedding + two-layer MLP | Stable Diffusion V1.5, ADM, BigGAN | Same three generators, disjoint manifest rows | 10 |

The completed GenImage test uses 4,000 balanced images from held-out
generators GLIDE, Midjourney, VQDM, and Wukong. The completed Defactify
external test uses 45,000 images from a pinned dataset revision.

### Frozen results

| Model | Evaluation | N | Accuracy | Balanced Accuracy | Precision | Fake recall | Binary F1 | AUROC |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| ResNet50 | GenImage unseen generators | 4,000 | 0.905000 | not reported | 0.983294 | 0.824000 | 0.896627 | 0.978279 |
| CLIP-MLP | GenImage unseen generators | 4,000 | 0.933000 | not reported | 0.965091 | 0.898500 | 0.930606 | 0.983852 |
| ResNet50 | Defactify official full | 45,000 | 0.623311 | 0.579213 | 0.868883 | 0.645360 | 0.740624 | 0.647152 |
| CLIP-MLP | Defactify official full | 45,000 | 0.837222 | 0.663667 | 0.885623 | 0.924000 | 0.904405 | 0.807892 |
| ResNet50 | Defactify balanced-from-full | 15,000 | 0.577200 | 0.577200 | 0.568424 | 0.641333 | 0.602681 | 0.644212 |
| CLIP-MLP | Defactify balanced-from-full | 15,000 | 0.662400 | 0.662400 | 0.606973 | 0.921467 | 0.731865 | 0.807510 |

“Not reported” means the frozen result file does not contain that field. It
must not be silently inferred in a frozen-results table.

## 2. Protocol v2 data roles

| Dataset/split | Permitted role | Forbidden use |
|---|---|---|
| GenImage fallback train | Model and classifier training | Final reporting as an independent test set |
| GenImage fallback validation | Model selection, CLIP feature mode, classifier size, dropout, optimizer settings, augmentation policy, validation-only threshold or calibration selection | Reporting as an unseen-generator result |
| GenImage fallback seen test | Optional diagnostic only | Hyperparameter selection after inspection |
| GenImage fallback unseen test | Internal held-out-generator evaluation after a candidate is specified from training/validation evidence | Repeated architecture or hyperparameter tuning |
| Defactify official full test | Final external cross-dataset evaluation of a frozen model and frozen decision protocol | Model structure, CLIP layer, pooling, head width, loss, optimizer, augmentation, epoch, checkpoint, threshold, or calibration selection |
| Defactify deterministic balanced subset | Class-balance diagnostic selected from the corresponding official-full predictions by sample ID | A second independent dataset, a second clean inference result, or any tuning |
| CIFAKE train/test | End-to-end pipeline sanity check only | Main research model selection or an unbiased final result; the current test is also configured as validation |
| WildFake CelebA-HQ/DDIM subset | Milestone 2 legacy evidence only | Current main external benchmark or a full-WildFake claim |

Defactify `train` and `validation` are not used by this project. This preserves
Defactify as an external dataset rather than an adaptation domain.

## 3. Model-selection and external-test discipline

1. Candidate designs must be specified using the literature, code inspection,
   GenImage train, and GenImage validation.
2. Pilot selection may use the GenImage unseen-generator result only at the
   explicit Go/No-Go gate defined by the project plan. Repeated tuning against
   that split is prohibited.
3. The final model, feature mode, pooling, classifier, loss, augmentation,
   optimizer, epoch budget, seed list, threshold, and calibration procedure
   must be frozen before formal Defactify inference.
4. Defactify results may determine how a frozen result is interpreted, but may
   not trigger model redesign or a new tuned run.
5. Every run, including a failed run, must retain its config hash, Git commit,
   seed, command, exit status, and failure reason in a lightweight registry.

## 4. Threshold, metrics, and statistics

The frozen baseline used fake probability threshold `0.5`. Protocol v2 keeps
`0.5` as the default. If a later part adds threshold selection or temperature
calibration, it must be fit exclusively on GenImage validation and then frozen.

Protocol v2 formal reporting must include, when implemented:

- accuracy;
- balanced accuracy;
- precision for fake as the positive class;
- fake recall and real recall;
- binary F1 and macro-F1;
- AUROC and AUPRC;
- confusion matrix.

The current frozen files do not contain all Protocol v2 metrics. They are
historical values, not substitutes for the later formal multi-seed runs.

The primary baseline and frozen final method will use seeds 42, 43, and 44.
Formal summaries must show every seed plus mean and sample standard deviation.
Paired confidence intervals are optional only if the five-day schedule permits.

## 5. Split integrity and known audit boundary

The manifest IDs inspected at baseline freeze have no overlap between:

- CIFAKE train and test;
- GenImage fallback train, validation, and unseen-generator test.

This is an ID/path-level check. It does not prove the absence of cross-dataset
near duplicates or shared source content. Full image SHA256 and perceptual-hash
audits are not part of this freeze and must be reported as incomplete unless a
later approved part performs them.

The Defactify balanced subset intentionally overlaps the official full test by
exactly 15,000 sample IDs. This is by design because its metrics are derived
from the official-full predictions.

## 6. Run naming and output isolation

Every new experiment name must use:

```text
{stage}_{model}_{training_scope}_{evaluation_scope}_seed{seed}_v2
```

Allowed `stage` values:

- `smoke`: functional check on a deliberately tiny scope;
- `pilot`: candidate-development run, never a final reported run;
- `final`: frozen formal run.

Examples:

```text
smoke_clip_penultimate_genimage_debug_seed42_v2
pilot_clip_penultimate_genimage_unseen_seed42_v2
pilot_dual_level_clip_genimage_unseen_seed42_v2
final_clip_mlp_genimage_defactify_seed43_v2
final_dual_level_clip_genimage_defactify_seed44_v2
```

Rules:

- smoke, pilot, and final runs must have distinct experiment names and output
  directories;
- a command must stop rather than overwrite an existing run directory unless
  an explicitly documented resume operation is requested;
- formal results must never reuse a smoke experiment name;
- figures, metrics, predictions, checkpoints, features, and logs must share the
  same run ID.

## 7. Artifact and Git policy

Commit:

- source code;
- tests;
- configurations;
- lightweight metric summaries and final report figures;
- protocol documents;
- artifact manifests containing paths, hashes, and provenance.

Do not commit:

- raw datasets or Hugging Face caches;
- checkpoints;
- feature tensors;
- full predictions;
- error-case image copies;
- large logs or temporary caches.

Large artifacts remain outside Git and are addressed by path plus SHA256. A
later release must provide a download location or explicit reproduction
instructions. No baseline artifact was moved, deleted, or added to Git during
this freeze.

## 8. Reproducibility evidence and limitations

- Defactify revision:
  `787334f7857fa54f29027a7f09c30e895ad486ef`
- Recorded formal hardware: NVIDIA GeForce RTX 5060 Laptop GPU, 8151 MiB
- Recorded formal software: Python 3.10.20, PyTorch 2.11.0+cu128
- The Git environment files constrain major dependencies but are not a fully
  resolved lock file.
- Raw data, checkpoints, feature caches, predictions, and logs are intentionally
  ignored by Git.
- The baseline freeze occurred in a worktree with pre-existing untracked files.
  Those files were neither modified nor included in the freeze commit.
- Remote equality was inferred from the local tracking reference without a
  network fetch.

## 9. Conditions for beginning Part 2

Part 2 may begin only when:

1. this protocol and the artifact manifest are committed;
2. the commit contains no model, training, evaluation, or data-pipeline logic;
3. no large artifact is staged or committed;
4. all recorded mandatory artifact paths exist and their hashes verify;
5. the pre-existing untracked worktree content remains untouched.
