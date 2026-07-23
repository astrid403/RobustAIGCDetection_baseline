from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


class ClipMlpDetector(nn.Module):
    def __init__(self, feature_dim, hidden_dim=512, dropout=0.2, num_outputs=1):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_outputs),
        )

    def forward(self, features):
        return self.classifier(features).squeeze(1)


class ClipLinearDetector(nn.Module):
    """A single linear probe over frozen final CLIP image embeddings."""

    def __init__(self, feature_dim, num_outputs=1):
        super().__init__()
        self.classifier = nn.Linear(feature_dim, num_outputs)

    def forward(self, features):
        return self.classifier(features).squeeze(1)


class DualLevelClipFusionDetector(nn.Module):
    """Minimal normalized concat fusion for final and penultimate CLIP features."""

    def __init__(self, final_dim, penultimate_dim, hidden_dim=256, dropout=0.2, num_outputs=1):
        super().__init__()
        self.final_dim = final_dim
        self.penultimate_dim = penultimate_dim
        self.classifier = nn.Sequential(
            nn.Linear(final_dim + penultimate_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_outputs),
        )

    def forward(self, features):
        expected_dim = self.final_dim + self.penultimate_dim
        if features.ndim != 2 or features.shape[1] != expected_dim:
            raise ValueError(
                f"Expected concatenated dual-level features with shape [batch, {expected_dim}], "
                f"got {tuple(features.shape)}."
            )
        final, penultimate = torch.split(
            features, [self.final_dim, self.penultimate_dim], dim=1
        )
        final = F.normalize(final, dim=1)
        penultimate = F.normalize(penultimate, dim=1)
        return self.classifier(torch.cat([final, penultimate], dim=1)).squeeze(1)


def build_clip_feature_detector(
    model_type,
    feature_dim,
    hidden_dim=512,
    dropout=0.2,
    fusion_hidden_dim=256,
    fusion_feature_dims=None,
):
    if model_type == "clip_linear":
        return ClipLinearDetector(feature_dim)
    if model_type == "clip_mlp":
        return ClipMlpDetector(feature_dim, hidden_dim, dropout)
    if model_type == "clip_fusion":
        if fusion_feature_dims is None:
            if feature_dim % 2:
                raise ValueError("Dual-level fusion requires explicit dimensions or an even feature dimension.")
            fusion_feature_dims = (feature_dim // 2, feature_dim // 2)
        if sum(fusion_feature_dims) != feature_dim:
            raise ValueError("Dual-level feature dimensions do not match the concatenated feature dimension.")
        return DualLevelClipFusionDetector(
            fusion_feature_dims[0],
            fusion_feature_dims[1],
            fusion_hidden_dim,
            dropout,
        )
    raise ValueError(f"Unsupported CLIP feature detector: {model_type}")


def pair_clip_feature_caches(final_cache, penultimate_cache):
    """Validate one-to-one pairing and concatenate final then penultimate rows."""
    final_metadata = final_cache.get("metadata", {})
    penultimate_metadata = penultimate_cache.get("metadata", {})
    if final_metadata.get("feature_mode") != "final":
        raise ValueError("The first fusion cache must contain final CLIP features.")
    if penultimate_metadata.get("feature_mode") != "penultimate":
        raise ValueError("The second fusion cache must contain penultimate CLIP features.")
    if final_metadata.get("cache_signature") == penultimate_metadata.get("cache_signature"):
        raise ValueError("Final and penultimate caches must have distinct cache signatures.")

    final_paths = list(final_cache.get("paths", []))
    penultimate_paths = list(penultimate_cache.get("paths", []))
    if final_paths != penultimate_paths:
        raise ValueError("Final and penultimate feature sample order does not match.")
    final_labels = torch.as_tensor(final_cache["labels"])
    penultimate_labels = torch.as_tensor(penultimate_cache["labels"])
    if final_labels.shape != penultimate_labels.shape or not torch.equal(final_labels, penultimate_labels):
        raise ValueError("Final and penultimate feature labels do not match.")

    final_features = torch.as_tensor(final_cache["features"])
    penultimate_features = torch.as_tensor(penultimate_cache["features"])
    if final_features.ndim != 2 or penultimate_features.ndim != 2:
        raise ValueError("Dual-level feature caches must contain 2D feature tensors.")
    if final_features.shape[0] != len(final_paths) or penultimate_features.shape[0] != len(final_paths):
        raise ValueError("Dual-level feature row counts do not match the paired sample IDs.")
    if not torch.isfinite(final_features).all() or not torch.isfinite(penultimate_features).all():
        raise ValueError("Dual-level feature caches contain non-finite values.")

    return {
        "features": torch.cat([final_features, penultimate_features], dim=1),
        "feature_dims": (final_features.shape[1], penultimate_features.shape[1]),
        "labels": final_labels,
        "paths": final_paths,
        "sample_ids": final_paths,
        "metadata": {
            "feature_modes": ["final", "penultimate"],
            "cache_signatures": [
                final_metadata["cache_signature"],
                penultimate_metadata["cache_signature"],
            ],
        },
    }


def load_open_clip_model(clip_model="ViT-B/32", device="cpu", pretrained="openai"):
    try:
        import open_clip
    except ImportError as exc:
        raise ImportError(
            "open_clip_torch is required for CLIP-MLP. Install requirements.txt first."
        ) from exc
    model_name = clip_model.replace("/", "-")
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    model.eval().to(device)
    for param in model.parameters():
        param.requires_grad = False
    return model, preprocess


def _penultimate_visual_embedding(visual, images):
    """Pool/project the tokens immediately before the final ViT residual block."""
    required = ("_embeds", "_pool", "transformer", "proj")
    if not all(hasattr(visual, name) for name in required):
        raise TypeError("Penultimate mode requires an open_clip VisionTransformer visual encoder.")
    transformer = visual.transformer
    blocks = transformer.resblocks
    if len(blocks) < 2:
        raise ValueError("Penultimate mode requires at least two transformer residual blocks.")

    tokens = visual._embeds(images)
    if not transformer.batch_first:
        tokens = tokens.transpose(0, 1).contiguous()
    for block in list(blocks)[:-1]:
        tokens = block(tokens, attn_mask=None)
    if not transformer.batch_first:
        tokens = tokens.transpose(0, 1)
    pooled, _ = visual._pool(tokens)
    if visual.proj is not None:
        pooled = pooled @ visual.proj
    return pooled


@torch.no_grad()
def encode_clip_image_features(clip_model, images, feature_mode="final"):
    """Return normalized final or penultimate CLIP image embeddings."""
    if feature_mode == "final":
        features = clip_model.encode_image(images)
    elif feature_mode == "penultimate":
        features = _penultimate_visual_embedding(clip_model.visual, images)
    else:
        raise ValueError(f"Unsupported CLIP feature mode: {feature_mode}")
    if features.ndim != 2 or features.shape[0] != images.shape[0]:
        raise ValueError(
            f"Expected 2D CLIP features with batch size {images.shape[0]}, "
            f"got {tuple(features.shape)}."
        )
    if not torch.isfinite(features).all():
        raise ValueError("CLIP feature extraction produced non-finite values.")
    return features / features.norm(dim=-1, keepdim=True).clamp_min(1e-6)


@torch.no_grad()
def extract_clip_features(clip_model, loader, device, feature_mode="final"):
    features, labels, paths = [], [], []
    for images, batch_labels, batch_paths in loader:
        images = images.to(device)
        feats = encode_clip_image_features(clip_model, images, feature_mode=feature_mode)
        features.append(feats.cpu())
        labels.append(torch.as_tensor(batch_labels, dtype=torch.float32))
        paths.extend(batch_paths)
    return torch.cat(features), torch.cat(labels), paths


def save_feature_cache(path, features, labels, paths, metadata=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "features": features,
            "labels": labels,
            "paths": paths,
            "sample_ids": paths,
            "metadata": metadata or {},
        },
        path,
    )


def load_feature_cache(path):
    return torch.load(path, map_location="cpu")
