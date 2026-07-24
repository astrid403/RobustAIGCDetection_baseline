# S2 RINE-lite Preregistration

## F01 scope and evidence boundary

This preregistration uses only the frozen Task 08 GenImage development OOF
summary and its six run registries. It does not reread images, run inference,
train a model, or access GenImage unseen or Defactify. S1 predictions,
metrics, checkpoints, registries, and the No-Go decision remain unchanged.

## S1 failure audit

The primary failure is **weak and generator-specific local signal**, not
degradation sensitivity:

- NPR clean fold-mean AUROC was `0.950383`, below B2-v3's `0.966050`.
- On held-out Stable Diffusion V1.5, NPR clean AUROC was `0.897850`, versus
  B2-v3 `0.933925`. Fixed fusion then reduced the B2 worst fold by `0.015400`.
- In contrast, fixed fusion improved the mean fold AUROC across
  JPEG/resize/blur by `0.030631` over B2-v3. The evidence therefore does not
  support JPEG/resize/blur fragility as the dominant failure.

Error complementarity exists but is insufficient:

- among 1,200 clean OOF samples, 127 were B2-wrong/NPR-right and 103 were
  NPR-wrong/B2-right;
- only 102 errors were shared and there were 230 prediction disagreements;
- despite this, fusion trailed the stronger clean expert by `0.003333`.

The secondary failure is **score/class bias under fixed fusion**. Fusion
raised real recall to `0.988333` but reduced fake recall to `0.631667`.
Changing alpha, calibration, or threshold after seeing this result is
forbidden, so S2 is a standalone representation route rather than a repair to
S1 fusion.

There is no evidence of an implementation or split-integrity failure: all six
registries, best-checkpoint predictions, sample IDs, fold assignments,
labels, ordering, and hashes passed Task 08 audit. The failure is therefore
classified as a development-domain scientific No-Go dominated by the
held-out SD1.5 shift.

## Frozen S2 hypothesis

S2 tests whether a compact mixture of widely separated frozen CLIP stages
generalizes across generators better than a single final embedding. It does
not reuse final-plus-penultimate concatenation. One CLIP forward extracts the
class token after completed blocks 3, 6, 9, and 12. These four 768-D tokens
share one `Linear(768,128)-ReLU` projection.

A shared trainable importance estimator, `Linear(128,1)`, assigns
sample-conditioned scores to the four projected tokens. A softmax across
blocks produces importance weights; their weighted sum is L2-normalized and
fed to `Linear(128,1)`. There is no dropout, learned temperature, entropy
penalty, residual final feature, or extra fusion expert.

The loss is BCE plus supervised contrastive loss with fixed weight `0.1` and
temperature `0.07`. SupCon positives share the binary real/fake label; self
pairs are excluded. No generator label enters the objective.

## Frozen training and evaluation budget

- Frozen OpenAI CLIP ViT-B/32 encoder.
- Same three LOGO splits and deterministic generator-label-balanced sampler
  as Protocol v3.
- Seed 42 pilot; seeds 42/43/44 only after a successful approved gate.
- AdamW, learning rate `0.001`, weight decay `0.01`, no scheduler.
- Batch size 32, 2,000 sampled examples per epoch, 10 epochs: exactly 20,000
  draws and 630 optimizer steps per fold.
- Validation-AUROC checkpoint selection with earliest-epoch exact tie-break.
- Fixed threshold 0.5.
- Clean, JPEG quality 70, resize 0.5 bicubic, and blur radius 1.0 diagnostics.

The primary F04 comparator is the existing same-fold B2-v3 control. All
Development Gate criteria are mandatory:

- clean fold-mean AUROC delta at least `+0.005`;
- clean worst-fold AUROC delta at least `-0.002`;
- improvement in at least two of three clean folds;
- mean degraded fold AUROC no lower than B2-v3;
- clean real and fake recall both at least `0.70`.

This is the applicable single-model form of the Task 08 Gate. The S1-only
“fusion over stronger expert” clause is inapplicable because S2 has no expert
fusion; it is not replaced by a weaker tunable criterion.

## Approval and stopping boundary

`configs/research_v3/S2_CONTRACT.yaml` is the machine-readable authority.
F01 freezes its block IDs, projection, TIE, SupCon objective, budget, cache
signature, evaluation conditions, and Gate. Approval of F01 permits only F02,
which may implement and test the multi-block feature contract without
training. F03 or later work requires separate approval.
