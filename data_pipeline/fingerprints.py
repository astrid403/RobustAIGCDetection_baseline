import hashlib
import json
from pathlib import Path

import pandas as pd


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
    metadata = {
        "manifest_path": str(Path(csv_path)),
        "manifest_fingerprint": manifest_fingerprint(csv_path),
        "dataset_id": str(first.get("dataset_id", first.get("dataset", "unknown"))),
        "dataset_revision": str(first.get("dataset_revision", "")),
        "model_name": config.get("clip_model", "ViT-B-32"),
        "pretrained": config.get("pretrained", "openai"),
        "preprocess_signature": "open_clip:create_model_and_transforms",
    }
    signature = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode("utf-8")).hexdigest()
    metadata["cache_signature"] = signature
    return metadata


def feature_cache_path(experiment_name, split_name, metadata):
    short = metadata["cache_signature"][:16]
    return Path("outputs/features") / experiment_name / f"{split_name}_{short}_clip_features.pt"


def validate_feature_cache(cache, expected):
    actual = cache.get("metadata")
    if not actual or actual.get("cache_signature") != expected.get("cache_signature"):
        raise ValueError("CLIP feature cache metadata does not match the requested dataset/model signature.")
