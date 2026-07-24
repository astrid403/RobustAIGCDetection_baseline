# Multi-block CLIP Feature Contract

## Scope

Task F02 implements only the frozen S2 encoder feature boundary. It does not
implement projection, TIE, the classifier, supervised contrastive loss, or
training. The API consumes already preprocessed tensors; it does not read any
dataset.

## Extraction

`encode_clip_multiblock_cls` accepts a frozen open_clip ViT-B/32 and an image
batch. It requires the exact ordered block tuple `(3,6,9,12)`, using one-based
completed-transformer-block indexing.

The implementation calls `_embeds` once, traverses all 12 residual blocks
once, and captures the class token immediately after blocks 3, 6, 9, and 12.
The same frozen visual `ln_post` is applied independently to every captured
class token. The visual projection and L2 normalization are not applied.

The output order is always block 3, block 6, block 9, block 12, with shape
`[batch,4,768]`. Features must be finite and are returned under
`torch.no_grad()`. Both batch-first and sequence-first open_clip transformer
layouts are supported.

Wrong, missing, duplicated, or reordered block IDs hard fail. A visual
encoder without the required open_clip API, a transformer with a block count
other than 12, a width other than 768, a non-finite result, or an unexpected
output shape also hard fails.

## Cache identity

S2 uses a new schema, `research_clip_multiblock_cache_v3`, isolated under
`outputs/research_v3/features/multiblock`. Its signature includes:

- encoder and pretrained tag;
- installed `open_clip_torch` version;
- ordered block IDs and their indexing convention;
- extraction version `clip_multiblock_cls_v1`;
- preprocessing fingerprint;
- input manifest SHA256;
- LOGO fold and split role;
- fixed per-sample feature shape `[4,768]`.

Validation compares every required metadata field, not only a supplied
signature. Any block order, open_clip version, preprocessing, manifest,
extraction version, fold, split, schema, signature, tensor-shape, row-count,
or finiteness mismatch hard fails.

## Verification

Synthetic tests verify shape, fixed order, deterministic output, one embed
call and one call per residual block, equivalence to four independently
traversed block references, batch-first/sequence-first equivalence, strict
cache validation, and malformed-input failures.

The available CUDA check compared identical tiny models on CPU and GPU. Its
first `atol=1e-6` check observed maximum absolute difference
`1.1780401791838813e-6`; the test-only cross-device tolerance was fixed at
`rtol=1e-6, atol=2e-6` and the single approved rerun passed. This does not
relax same-device determinism, which remains tensor-exact.

A real cached OpenAI ViT-B/32 CUDA smoke on one deterministic synthetic
`[1,3,224,224]` tensor produced `[1,4,768]`, finite, no-gradient features.
Two consecutive extractions were tensor-exact. No GenImage unseen or
Defactify input was used.
