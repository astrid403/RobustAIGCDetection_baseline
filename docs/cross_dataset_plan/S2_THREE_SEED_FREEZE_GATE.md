# S2 Three-Seed Confirmation and Freeze Gate

## Valid execution

Seed 42 reuses the valid F04 CUDA AMP runs. Seeds 43 and 44 each ran three
new B2-v3 and three new S2 LOGO jobs. All twelve training and tee processes
exited zero. One cached-weight HEAD request timed out and was retried
internally by `huggingface_hub`; the affected command completed normally and
its full warning remains in the console log. No scientific job was rerun.

All S2 registries report effective CUDA float16 autocast and GradScaler. All
conditions use best-validation-AUROC checkpoints and fixed threshold 0.5.
GenImage unseen and Defactify were not accessed.

## Frozen Gate evidence

- Per-seed clean fold-mean S2-minus-B2 AUROC:
  seed 42 `+0.022829`, seed 43 `+0.027333`, seed 44 `+0.022021`.
- Overall nine-fold mean delta: `+0.024061`.
- Worst generator's three-seed mean delta: `+0.004388`.
- Every seed-level mean delta is positive.
- S2 clean mean real recall: `0.986667`; fake recall: `0.724444`.
- Thirty-repeat CUDA batch-32 benchmark: B2 `0.078730 s`, S2 `0.139795 s`,
  ratio `1.775628`; S2 peak allocated memory `636.904 MiB`.
- Block-importance files are present and hashed in every S2 registry.

All numerically defined Model Freeze Gate criteria pass.

## Approved deployable checkpoint construction

The user approved option 2: for each seed, average fake probabilities from the
three frozen LOGO best checkpoints in fixed ADM, BigGAN, SD1.5 fold order with
equal weights. Apply threshold 0.5 after aggregation. B2-v3 uses the same
rule. No retraining, fold selection, learned weight, or calibration is
allowed. The model is now completely frozen.
