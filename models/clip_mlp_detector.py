from pathlib import Path

import torch
import torch.nn as nn


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


def build_clip_feature_detector(model_type, feature_dim, hidden_dim=512, dropout=0.2):
    if model_type == "clip_linear":
        return ClipLinearDetector(feature_dim)
    if model_type == "clip_mlp":
        return ClipMlpDetector(feature_dim, hidden_dim, dropout)
    raise ValueError(f"Unsupported CLIP feature detector: {model_type}")


def load_open_clip_model(clip_model="ViT-B/32", device="cpu"):
    try:
        import open_clip
    except ImportError as exc:
        raise ImportError(
            "open_clip_torch is required for CLIP-MLP. Install requirements.txt first."
        ) from exc
    model_name = clip_model.replace("/", "-")
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained="openai")
    model.eval().to(device)
    for param in model.parameters():
        param.requires_grad = False
    return model, preprocess


@torch.no_grad()
def extract_clip_features(clip_model, loader, device):
    features, labels, paths = [], [], []
    for images, batch_labels, batch_paths in loader:
        images = images.to(device)
        feats = clip_model.encode_image(images)
        feats = feats / feats.norm(dim=-1, keepdim=True).clamp_min(1e-6)
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
