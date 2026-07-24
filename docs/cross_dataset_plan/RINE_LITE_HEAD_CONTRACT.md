# RINE-lite Head and SupCon Contract

## Model boundary

`RineLiteDetector` consumes only cached multi-block features with exact shape
`[batch,4,768]`. It does not own or fine-tune CLIP. Empty, non-floating,
non-finite, or incorrectly shaped inputs hard fail.

One shared `Linear(768,128,bias=True)-ReLU` projection is applied to every
block. One shared sample-conditioned `Linear(128,1,bias=True)` importance
scorer produces four logits. A temperature-1 softmax across blocks yields
importance weights, and their weighted 128-D sum is L2-normalized. A single
`Linear(128,1,bias=True)` classifier produces logits.

There is no dropout, residual final embedding, entropy regularizer, learned
temperature, extra expert, or auxiliary head. The model has exactly 98,690
trainable parameters. `forward_with_aux` exposes logits, normalized
embeddings, and per-sample block importance for required provenance.

## Objective

`RineLiteObjective` is exactly:

`BCEWithLogitsLoss + 0.1 * supervised_contrastive_loss`.

SupCon uses the L2-normalized 128-D aggregate, temperature `0.07`, and binary
real/fake labels. Self pairs are excluded. Same-label samples are positives;
generator identity is never used. Anchors without a positive are excluded.
If a batch has no valid anchors, SupCon returns differentiable zero.

Invalid feature shape/type/finiteness, invalid label shape or values, invalid
logit shape, or non-positive temperature hard fail.

## F03 verification

Synthetic tests verify the frozen parameter count and module graph, shared
projection, sample-conditioned importance, softmax sums, normalized
embeddings, forward/backward finiteness, exact state-dict round trip, SupCon
permutation invariance, empty-anchor behavior, and malformed-input failures.

One available-GPU CPU/GPU head comparison passed at `rtol=1e-5, atol=2e-6`.
The first directed CPU test used an overly strict `1e-7` softmax-sum absolute
tolerance and observed one float32 ULP (`1.1920928955078125e-7`); the
test-only tolerance was fixed to `2e-7`, after which the directed suite
passed. Model behavior was not changed.

The approved GPU tiny smoke used seed 42 and one synthetic
`[32,4,768]` batch with balanced binary labels. It completed one
forward/backward and AdamW step at learning rate `0.001`, weight decay
`0.01`. BCE was `0.6934673786`, SupCon was `3.6079678535`, and total loss
was `1.0542641878`. All parameter gradients were finite, and a checkpoint
reload reproduced logits, embeddings, and importance tensor-exactly.

F03 does not register this model in the formal training dispatcher, create
GenImage feature caches, or authorize F04 training.
