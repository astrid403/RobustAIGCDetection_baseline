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

## Required stop before unseen inference

The contract does not define how one final checkpoint per seed is constructed
from the three LOGO fold checkpoints. Full-development retraining, selecting a
fold checkpoint, and a three-fold ensemble imply different data protocols or
deployed model structures. Choosing among them now would modify an unfrozen
scientific protocol. Model architecture and hyperparameters are frozen, but
the deployable-checkpoint construction is blocked pending an explicit
contract choice. No GenImage unseen inference may start before that choice is
recorded.
