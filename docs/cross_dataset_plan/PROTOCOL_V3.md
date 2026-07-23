# Cross-Dataset Robustness Protocol v3

## 1. Status and isolation

- Status: **candidate contract awaiting explicit user approval**.
- Research branch: `research/cross-dataset-robustness-v3`.
- Frozen base commit: `cedc2a18d956948acfed87d575d41d4b93689d1d`.
- Candidate route: `S1`, semantic-broken NPR expert plus frozen-CLIP semantic
  expert late fusion.
- Label contract: `0=real`, `1=fake`.

Protocol v3 is independent of Protocol v2. It does not alter the existing
model freeze, number freeze, Part 11–12 state, results, or claims. No command
authorized by this document may overwrite a v2 run or artifact.

## 2. Research question

Does a local neighboring-pixel-relation (NPR) expert, trained with a
pre-registered semantic-breaking transform and combined with a frozen-CLIP
semantic expert by fixed late fusion, improve leave-one-generator-out
generalization without sacrificing worst-generator behavior?

The primary development comparison is S1 versus a newly trained B2-v3 control
under identical source-aware LOGO folds. Defactify cannot answer any design or
selection question.

## 3. Data roles

| Data | v3 role | Forbidden use |
|---|---|---|
| GenImage available train | Training within each LOGO fold | Independent test claim |
| GenImage available validation | Held-out-generator LOGO validation | Training within its held-out fold |
| GenImage unseen | One diagnostic after model freeze | Architecture, hyperparameter, checkpoint, fusion, threshold, or route selection |
| Defactify full | One external evaluation after separate approval | Any model selection or tuning |
| Defactify balanced | Derived from the matching full predictions | Second inference or tuning |

Task 02 froze three proposed folds:

| Fold | Training generators | Validation generator | Train rows | Validation rows |
|---|---|---|---:|---:|
| holdout ADM | BigGAN, Stable Diffusion V1.5 | ADM | 1,999 | 400 |
| holdout BigGAN | ADM, Stable Diffusion V1.5 | BigGAN | 2,000 | 400 |
| holdout SD1.5 | ADM, BigGAN | Stable Diffusion V1.5 | 2,000 | 400 |

The holdout-ADM fold excludes one pre-registered perceptual near-duplicate.
No later balancing deletion is allowed. Training batches use the sampler in
Section 7.

## 4. B2-v3 semantic control

- Encoder: OpenAI CLIP ViT-B/32 (`open_clip`, pretrained tag `openai`).
- Encoder parameters: frozen.
- Input/preprocess: the model-provided OpenAI preprocessing at 224 resolution.
- Feature: final normalized 512-D image embedding.
- Head: `Linear(512,512)-ReLU-Dropout(0.2)-Linear(512,1)`.
- Loss: `BCEWithLogitsLoss`.
- Optimizer: AdamW, learning rate `0.001`, weight decay `0.01`.
- Batch size: 32.
- Epochs: 10.
- Checkpoint: maximum fold-validation AUROC; earliest epoch wins exact ties.
- Threshold: fixed 0.5.

B2-v3 is retrained on each v3 fold. Protocol v2 checkpoints are not reused as
fold controls.

## 5. NPR representation contract

The NPR expert receives an RGB image and applies these operations in order:

1. Convert to RGB.
2. Resize directly to `224 x 224` using bicubic interpolation.
3. During training only, apply horizontal flip with probability 0.5.
4. During training only, apply the semantic-breaking transform in Section 6.
5. Convert each channel to float32 in `[0,1]`.
6. Compute a signed diagonal neighboring-pixel difference:

   `npr[c,y,x] = rgb[c,y,x] - rgb[c,y+1,x+1]`

   for `c in {R,G,B}`, `0 <= y < 223`, and `0 <= x < 223`.
7. Do not pad. The output is exactly `[3,223,223]`.
8. Preserve sign and map the fixed mathematical range `[-1,1]` to `[0,1]`
   using `(npr + 1) / 2`.
9. Normalize with ImageNet mean `[0.485,0.456,0.406]` and standard deviation
   `[0.229,0.224,0.225]`.

Values outside the input contract are an error; the implementation may allow
a numerical tolerance of `1e-6` before failing. The transform must not inspect
label, generator, source, path text, or file metadata other than decoded RGB
pixels.

## 6. Semantic-breaking transform

- Grid: fixed `4 x 4`.
- Input is exactly `224 x 224`, so every patch is `56 x 56`.
- Training application probability: 0.5.
- When applied, all 16 patches are permuted by a seed-derived permutation.
- The identity permutation is rejected and redrawn, so “applied” always
  changes spatial layout.
- When not applied, the original image is retained.
- No rotation, rescaling, color change, interpolation, or cross-sample mixing
  occurs during patch permutation.
- Validation/evaluation application probability: 0.
- RNG key: global seed, epoch, and stable sample ID.

No random crop, color jitter, Gaussian noise, MixUp, CutMix, or other
augmentation is allowed in the S1 main configuration.

## 7. NPR expert and optimization

- Backbone: torchvision ImageNet-pretrained ResNet18.
- Input channels: 3; standard first convolution is retained.
- Head replacement: final `Linear(512,1)`.
- Trainable parameters: all ResNet18 parameters.
- Loss: `BCEWithLogitsLoss`.
- Consistency loss in the S1 main model: disabled, weight 0.
- Optimizer: AdamW.
- Learning rate: `0.0001`.
- Weight decay: `0.0001`.
- Batch size: 32; this may not be reduced based on scientific results.
- Epochs: 10.
- Scheduler: none.
- Mixed precision: enabled on CUDA.
- Checkpoint: maximum fold-validation AUROC; earliest epoch wins exact ties.
- Threshold: fixed 0.5.
- Seeds: 42 for pilot/ablation; 42, 43, 44 for confirmation.
- Workers: 0 for the first formal implementation contract.

Training uses a deterministic class/generator-balanced sampler. Each epoch
draws `2,000` samples with replacement, allocating equal quotas to every
available `(generator,label)` group. There are four groups in each training
fold, so each contributes 500 draws. Batch ordering is determined by seed and
epoch. Validation is never sampled or shuffled.

The fixed epoch budget is therefore 10 epochs and 20,000 sampled training
examples per fold. No early stopping changes the executed budget.

## 8. Degradation views and consistency ablation

The S1 main model does not use degradation augmentation or consistency loss.
Task 05 nevertheless implements the deterministic paired-view contract needed
for the pre-registered Task 09 ablation and robustness diagnostics:

- JPEG: quality 70;
- resize: downscale to 0.5 then bicubic upsample to original size;
- Gaussian blur: radius 1.0;
- clean diagnostic: no degradation.

For robustness evaluation, each named degradation is applied with probability
1.0. For the optional Task 09 `with_consistency` ablation only:

- a degraded partner is constructed with probability 0.5;
- degradation type is selected uniformly from JPEG, resize, and blur;
- the same semantic-breaking patch permutation is used for the clean and
  degraded members before NPR calculation;
- prediction consistency is mean squared error between sigmoid probabilities;
- consistency weight is fixed at 0.1;
- BCE remains the primary loss.

This ablation cannot replace the main S1 specification based on external
results. It is run only if Task 08 passes and Task 09 is approved.

## 9. Late fusion

- Semantic probability: `p_clip` from B2-v3.
- Local probability: `p_npr` from the NPR expert.
- Primary S1 score: `0.5 * p_clip + 0.5 * p_npr`.
- Decision threshold: fixed 0.5 on the primary fused score.
- No learned fusion head and no calibration.
- The descriptive alpha grid is exactly `[0.0,0.25,0.5,0.75,1.0]`.
- Alpha-grid results are ablations only; they cannot replace alpha 0.5 as the
  primary S1 score.
- Predictions must be joined one-to-one by sample ID. Missing, duplicate, or
  reordered IDs are hard failures.

Only out-of-fold validation predictions may be used to compute development
fusion metrics.

## 10. Metrics and selection

Primary development metric: mean AUROC over the three LOGO folds.

Safety/tie-break metric: worst-fold AUROC. Other mandatory metrics are AUPRC,
balanced accuracy, macro-F1, real recall, fake recall, per-generator results,
and clean/JPEG/resize/blur results. All comparisons use matched sample IDs.

Task 08 seed-42 Go requires all of:

- S1 minus B2-v3 mean AUROC at least `+0.005`;
- worst-fold AUROC degradation no worse than `-0.002`;
- improvement in at least two of three folds;
- mean degraded AUROC no worse than B2-v3, unless a pre-registered clean
  trade-off is explicitly reported;
- S1 exceeds the stronger single expert by at least `+0.002` AUROC;
- real and fake recall both at least 0.70.

Task 10 model-freeze Go requires all of:

- mean three-seed S1-minus-B2 LOGO AUROC at least `+0.005`;
- paired 95% CI lower bound above zero, or all three seed deltas positive if
  the approved analysis determines the CI sample unit is insufficient;
- mean worst-fold degradation no worse than `-0.002`;
- fixed-alpha and fold ranking are stable across seeds;
- inference time at most 2.5 times B2-v3 and peak allocated GPU memory below
  8,000 MiB;
- complete provenance.

No criterion may be changed after Task 08 begins.

## 11. Naming, outputs, and provenance

- Run name: `{stage}_{model}_{training_scope}_{evaluation_scope}_seed{seed}_v3`.
- Stages: exactly `smoke`, `pilot`, `ablation`, `final`.
- Fold IDs: `holdout_adm`, `holdout_biggan`,
  `holdout_stable_diffusion_v15`.
- Output root: `outputs/research_v3/`.
- Checkpoints, logs, predictions, figures, features, and metrics use distinct
  subdirectories and the same run ID.
- Existing directories are never overwritten.
- Cache schema: `research_clip_cache_v3` for B2-v3; its signature includes
  encoder, pretrained tag, feature mode, preprocessing fingerprint, manifest
  SHA256, fold, split role, and extraction version.
- NPR inputs are computed from decoded images and are not persisted as full
  feature tensors in the first implementation.

Every run registry records branch, commit, command, config snapshot/hash,
input manifest/hash, seed, fold, start/end time, exit status, checkpoint/hash,
prediction/hash, environment, and failure reason. Failed runs remain recorded.

## 12. Forbidden decisions and stopping rules

- Do not access GenImage unseen before Task 10 model freeze approval.
- Do not access Defactify before Task 12's separate immediate approval.
- Do not use Defactify train/validation.
- Do not select architecture, checkpoint, transform, alpha, threshold, or
  calibration from GenImage unseen or Defactify.
- Do not add a backbone, augmentation, loss, data source, seed, or search
  dimension after seeing pilot or external results.
- Do not exclude a failed or unfavorable seed.
- Do not overwrite or modify Protocol v2 assets.
- Stop at every gate and wait for explicit user approval.

## 13. Task 03 approval gate

No implementation is authorized until the user explicitly approves both this
document and `configs/research_v3/s1_contract.yaml`. Approval changes the
contract status from `candidate` to `approved`; it does not authorize Task 06
or any later training by itself.
