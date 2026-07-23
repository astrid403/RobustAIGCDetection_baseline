# CLIP Feature Contract v2

## Inspected implementation

- Package: `open_clip_torch` / import package `open_clip`, version `3.3.0`.
- Installed source: `open_clip/transformer.py`.
- Model: OpenAI-pretrained `ViT-B-32`.
- Visual encoder: `VisionTransformer`, 12 residual blocks, token width 768,
  output dimension 512, token pooling, projection shape 768 x 512.
- Preprocess: resize 224 bicubic, center crop 224, RGB/tensor conversion, and
  OpenAI CLIP mean/std normalization.

## Approved modes

`final` is the normal `clip_model.encode_image` result after all 12 residual
blocks, token pooling, post layer normalization, and the 768 x 512 projection.

`penultimate` is the output immediately before residual block 12: image tokens
pass through `_embeds` and residual blocks 1–11, then use the same token
pooling, post layer normalization, and 768 x 512 projection as `final`. This is
implemented explicitly without forward hooks.

Both modes return a finite float tensor of shape `[batch, 512]` for ViT-B-32
and are L2-normalized along the feature dimension. Extraction runs under
`torch.no_grad()` with the encoder in evaluation mode and frozen parameters.

## Cache contract

Feature cache schema version 2 includes:

- manifest path and SHA256;
- dataset ID and revision;
- CLIP model and pretrained tag;
- feature mode;
- feature extraction version;
- preprocessing implementation, image size, interpolation, mean, and std.

The complete sorted metadata is hashed into the cache signature and path.
Changing final/penultimate mode produces a different cache path. Legacy caches
without feature mode and extraction version are rejected with an explicit
regeneration message; they are never silently reused.

## Part 3 validation boundary

Part 3 validates extraction and cache safety only. It does not implement a
penultimate classifier, dual-level fusion, training pilot, or Defactify
evaluation.
