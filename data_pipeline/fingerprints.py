import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path

import pandas as pd


FEATURE_EXTRACTION_VERSION = 2
OPENAI_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
OPENAI_CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def _open_clip_version():
    try:
        return version("open_clip_torch")
    except PackageNotFoundError:
        return "unknown"


def manifest_fingerprint(csv_path):
    path = Path(csv_path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def feature_cache_metadata(csv_path, config):
    frame = pd.read_csv(csv_path, nrows=1)
    first = frame.iloc[0] if not frame.empty else {}
    feature_mode = config.get("clip_feature_mode", "final")
    if feature_mode not in {"final", "penultimate"}:
        raise ValueError(f"Unsupported CLIP feature mode for cache metadata: {feature_mode}")
    metadata = {
        "manifest_path": str(Path(csv_path)),
        "manifest_fingerprint": manifest_fingerprint(csv_path),
        "dataset_id": str(first.get("dataset_id", first.get("dataset", "unknown"))),
        "dataset_revision": str(first.get("dataset_revision", "")),
        "model_name": config.get("clip_model", "ViT-B-32"),
        "pretrained": config.get("pretrained", "openai"),
        "feature_mode": feature_mode,
        "feature_extraction_version": FEATURE_EXTRACTION_VERSION,
        "open_clip_version": _open_clip_version(),
        "preprocess_signature": {
            "implementation": "open_clip:create_model_and_transforms",
            "image_size": config.get("image_size", 224),
            "interpolation": config.get("clip_interpolation", "bicubic"),
            "mean": config.get("clip_mean", OPENAI_CLIP_MEAN),
            "std": config.get("clip_std", OPENAI_CLIP_STD),
        },
    }
    signature = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode("utf-8")).hexdigest()
    metadata["cache_signature"] = signature
    return metadata


def feature_cache_path(experiment_name, split_name, metadata):
    short = metadata["cache_signature"][:16]
    return Path("outputs/features") / experiment_name / f"{split_name}_{short}_clip_features.pt"


def validate_feature_cache(cache, expected):
    actual = cache.get("metadata")
    required = {"cache_signature", "feature_mode", "feature_extraction_version"}
    if not actual or not required.issubset(actual):
        raise ValueError(
            "Legacy CLIP feature cache lacks the v2 feature-mode contract; "
            "regenerate it instead of reusing it."
        )
    if actual.get("cache_signature") != expected.get("cache_signature"):
        raise ValueError(
            "CLIP feature cache metadata does not match the requested "
            "dataset/model/preprocess/feature-mode signature."
        )
