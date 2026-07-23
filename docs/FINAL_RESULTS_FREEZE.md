# Final Results Freeze

## Scope

These are the only reportable final results for Protocol v2. They use frozen
B2 (CLIP final feature + MLP) and A1 (CLIP penultimate feature + MLP), seeds
42/43/44, and threshold 0.5. Values are mean ± sample standard deviation over
all three seeds. Defactify was never used for model selection or tuning.

## Main results

| Scope | Model | AUROC | AUPRC | Balanced accuracy | Macro-F1 |
|---|---|---:|---:|---:|---:|
| GenImage unseen | B2 | 0.984071 ± 0.000249 | 0.981849 ± 0.000155 | 0.933250 ± 0.001639 | 0.933175 ± 0.001656 |
| GenImage unseen | A1 | 0.988194 ± 0.000129 | 0.987415 ± 0.000137 | 0.937000 ± 0.003192 | 0.936924 ± 0.003219 |
| Defactify full 45k | B2 | 0.809061 ± 0.001318 | 0.948743 ± 0.000549 | 0.663524 ± 0.009654 | 0.677794 ± 0.006662 |
| Defactify full 45k | A1 | 0.796040 ± 0.001584 | 0.945939 ± 0.000541 | 0.662187 ± 0.011423 | 0.672647 ± 0.006021 |
| Defactify balanced 15k | B2 | 0.808732 ± 0.001485 | 0.800036 ± 0.002282 | 0.662000 ± 0.009739 | 0.637702 ± 0.013572 |
| Defactify balanced 15k | A1 | 0.795021 ± 0.001616 | 0.791151 ± 0.001922 | 0.661622 ± 0.011898 | 0.638487 ± 0.017371 |

## Paired statistical evidence

The 95% confidence intervals use 1,000 paired hierarchical bootstrap
repetitions with fixed random seed 20260723, resampling both training seeds and
matched sample IDs.

- GenImage unseen A1−B2 AUROC: +0.004123; CI [0.002493, 0.006033].
- GenImage unseen A1−B2 AUPRC: +0.005566; CI [0.002962, 0.008906].
- Defactify full A1−B2 AUROC: −0.013021; CI [−0.016487, −0.009541].
- Defactify full A1−B2 AUPRC: −0.002804; CI [−0.004176, −0.001467].
- Defactify balanced A1−B2 AUROC: −0.013711; CI [−0.017987, −0.009504].
- Defactify balanced A1−B2 AUPRC: −0.008885; CI [−0.014367, −0.002824].
- Balanced-accuracy intervals cross zero for all three scopes; no supported
  balanced-accuracy superiority claim is permitted.

## Robustness and subgroup evidence

On GenImage unseen, mean A1−B2 AUROC is +0.009764 under JPEG quality 70,
+0.007514 under resize scale 0.5, and +0.008178 under Gaussian blur radius
1.0. A1 has the smaller clean-to-corruption AUROC drop for all three settings.

On Defactify full, A1 improves DALL-E 3 AUROC by +0.042666 but declines on the
other four fake generators. Its largest decline is −0.054460 on Stable
Diffusion XL. A1's worst-generator full AUROC is 0.726247, while B2's is
0.772858 on Stable Diffusion 3.

## Frozen contribution wording

The final story is a **generalization trade-off / controlled negative result**.
Penultimate frozen CLIP features provide statistically supported GenImage
unseen ranking gains and stronger compact in-domain corruption robustness, but
those gains do not transfer to Defactify. The external ranking degradation is
also statistically supported. A1 must not be described as comprehensively or
universally superior to B2.

## Controlled report artifact set

Exactly five primary report artifacts are frozen:

1. `reports/final_assets/method_diagram.svg`
2. `reports/final_assets/main_results_table.csv`
3. `reports/final_assets/core_ablation_table.csv`
4. `reports/final_assets/generalization_robustness.png`
5. `outputs/figures/part09_error_comparison.png`

Other ROC, confusion-matrix, and per-run plots are supplemental and must not
replace this compact set.

## Limitations

- Only three training seeds are available.
- The robustness protocol uses one pre-frozen severity per corruption on
  GenImage unseen; it is not a comprehensive corruption benchmark.
- Defactify contains a strong class imbalance in the official full split;
  both full 45k and sample-ID-derived balanced 15k results must be identified
  explicitly.
- The selected A1 route remains frozen from the pre-Defactify decision, despite
  its external degradation.
- Efficiency timings are environment-specific and should not be generalized
  across hardware.

Corrections after the number-freeze manifest is committed require explicit
user approval and an audit note. No retraining, checkpoint reselection,
threshold adjustment, or selective seed exclusion is permitted.
