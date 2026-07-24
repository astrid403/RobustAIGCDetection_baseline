# Frozen S2 Defactify External Evaluation

The user explicitly authorized the first and only frozen Defactify evaluation.
B2-v3 and S2 used seeds 42/43/44, with each seed defined as the equal-
probability ensemble of its ADM, BigGAN, and SD1.5 LOGO best checkpoints.
Threshold 0.5 was unchanged.

Only Defactify official full was inferred. Each balanced result was derived
from its corresponding full prediction using the frozen balanced manifest
sample IDs; no balanced image inference occurred.

Mean results across seeds:

| Model | Scope | AUROC | AUPRC | Balanced accuracy | Real recall | Fake recall |
|---|---|---:|---:|---:|---:|---:|
| B2-v3 | full | 0.670774 | 0.895484 | 0.631982 | 0.358800 | 0.905164 |
| S2 | full | 0.719800 | 0.914853 | 0.663471 | 0.565333 | 0.761609 |
| B2-v3 | balanced | 0.669366 | 0.635804 | 0.630933 | 0.358800 | 0.903067 |
| S2 | balanced | 0.717464 | 0.688509 | 0.661044 | 0.565333 | 0.756756 |

S2-minus-B2 mean AUROC is `+0.049026` on full and `+0.048098` on balanced.
S2 improves the mean full per-generator AUROC for DALL-E 3, Midjourney 6,
Stable Diffusion 2.1, and Stable Diffusion 3, but is lower on Stable Diffusion
XL (`0.588723` versus `0.616546`). This is a subgroup limitation, not a model
selection signal.

The evaluation and tee processes exited zero with no retry. Six full files
contain 45,000 unique IDs each; six balanced files contain exactly the frozen
15,000 IDs in manifest order. Sixty per-generator rows are complete. All
manifest/checkpoint/prediction/output hashes and independent metric
recomputations passed. No tuning, retraining, checkpoint selection, threshold
change, or selective seed exclusion occurred.
