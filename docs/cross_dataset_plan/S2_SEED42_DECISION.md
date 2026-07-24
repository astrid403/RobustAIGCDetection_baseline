# S2 Seed-42 LOGO Invalid Execution Record

## Decision

**No Gate decision is valid.** The first F04 execution used FP32 head
training even though `S2_CONTRACT.yaml` freezes CUDA mixed precision as
enabled. Its three jobs and artifacts are retained as an invalid execution
record, but none of the metrics below may drive a Go/No-Go decision. F05 is
not authorized.

## Results

| Condition | Model | Fold-mean AUROC | Worst-fold AUROC | Overall AUROC | AUPRC | Balanced accuracy | Real recall | Fake recall |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| clean | B2-v3 | 0.966050 | 0.933925 | 0.904872 | 0.918140 | 0.809167 | 0.963333 | 0.655000 |
| clean | S2 | 0.988192 | 0.972000 | 0.981224 | 0.980177 | 0.826667 | 0.990000 | 0.663333 |
| JPEG | B2-v3 | 0.937492 | 0.903825 | 0.900803 | 0.906001 | 0.830833 | 0.881667 | 0.780000 |
| JPEG | S2 | 0.979008 | 0.966825 | 0.973261 | 0.970507 | 0.873333 | 0.966667 | 0.780000 |
| resize | B2-v3 | 0.919575 | 0.876225 | 0.853547 | 0.866255 | 0.780833 | 0.898333 | 0.663333 |
| resize | S2 | 0.978142 | 0.958725 | 0.946214 | 0.947289 | 0.818333 | 0.966667 | 0.670000 |
| blur | B2-v3 | 0.921100 | 0.874000 | 0.869564 | 0.879719 | 0.793333 | 0.880000 | 0.706667 |
| blur | S2 | 0.976408 | 0.962500 | 0.946606 | 0.943491 | 0.830833 | 0.953333 | 0.708333 |

Clean S2-minus-B2 fold AUROC deltas were `+0.024225` on ADM, `+0.004125`
on BigGAN, and `+0.038075` on Stable Diffusion V1.5.

## Descriptive checks from the invalid run

| Criterion | Required | Observed | Result |
|---|---:|---:|---|
| Clean fold-mean AUROC delta | >= +0.005 | +0.022142 | Pass |
| Clean worst-fold AUROC delta | >= -0.002 | +0.038075 | Pass |
| Improved clean folds | >= 2/3 | 3/3 | Pass |
| Mean degraded fold-AUROC delta | >= 0 | +0.051797 | Pass |
| Clean real recall | >= 0.70 | 0.990000 | Pass |
| Clean fake recall | >= 0.70 | 0.663333 | **Fail** |

If the execution had been protocol-valid, the failure would have been
concentrated in the frozen threshold behavior of the held-out
SD1.5 fold, whose clean fake recall was `0.09` despite AUROC `0.972`. This is
evidence of severe cross-fold score calibration/class bias. Threshold,
calibration, loss weight, checkpoint, and architecture changes after this
pilot are forbidden, so the strong ranking result cannot be converted into a
Go decision.

## Provenance

All three invalid jobs completed with training and tee exit code zero and no retries.
Best epochs were 1 (ADM), 6 (BigGAN), and 1 (SD1.5). Each registry hashes its
config, splits, two multi-block caches, best checkpoint, four prediction
files, and four block-importance files. Every condition contains 1,200 unique
OOF samples with exact B2/S2 sample, fold, label, and order alignment.

No GenImage unseen or Defactify data was accessed.

The protocol conflict was detected before a result commit. A corrected run
requires explicit user approval, a mixed-precision implementation/config
repair, three new non-overwriting run IDs, and preservation of these outputs.
