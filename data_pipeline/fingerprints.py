import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path

import pandas as pd
import torch


FEATURE_EXTRACTION_VERSION = 2
RESEARCH_CLIP_CACHE_SCHEMA = "research_clip_cache_v3"
RESEARCH_CLIP_EXTRACTION_VERSION = 3
RESEARCH_MULTIBLOCK_CACHE_SCHEMA = "research_clip_multiblock_cache_v3"
RESEARCH_MULTIBLOCK_EXTRACTION_VERSION = "clip_multiblock_cls_v1"
RESEARCH_MULTIBLOCK_IDS = (3, 6, 9, 12)
RESEARCH_MULTIBLOCK_SHAPE = (4, 768)
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


def research_clip_cache_metadata(csv_path, config, fold, split_role):
    """Build the isolated Protocol-v3 cache signature."""
    if not fold or not split_role:
        raise ValueError("Research cache metadata requires fold and split_role")
    base = feature_cache_metadata(csv_path, config)
    metadata = {
        "schema": RESEARCH_CLIP_CACHE_SCHEMA,
        "encoder": base["model_name"],
        "pretrained": base["pretrained"],
        "feature_mode": base["feature_mode"],
        "preprocess_fingerprint": hashlib.sha256(
            json.dumps(base["preprocess_signature"], sort_keys=True).encode("utf-8")
        ).hexdigest(),
        "manifest_path": base["manifest_path"],
        "manifest_sha256": base["manifest_fingerprint"],
        "fold": str(fold),
        "split_role": str(split_role),
        "extraction_version": RESEARCH_CLIP_EXTRACTION_VERSION,
        "open_clip_version": base["open_clip_version"],
    }
    metadata["cache_signature"] = hashlib.sha256(
        json.dumps(metadata, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return metadata


def research_clip_cache_path(metadata):
    if metadata.get("schema") != RESEARCH_CLIP_CACHE_SCHEMA:
        raise ValueError("Expected research_clip_cache_v3 metadata")
    return (
        Path("outputs/research_v3/features")
        / metadata["fold"]
        / f"{metadata['split_role']}_{metadata['cache_signature'][:16]}_clip_features.pt"
    )


def validate_research_clip_cache(cache, expected):
    actual = cache.get("metadata")
    required = {
        "schema",
        "encoder",
        "pretrained",
        "feature_mode",
        "preprocess_fingerprint",
        "manifest_sha256",
        "fold",
        "split_role",
        "extraction_version",
        "cache_signature",
    }
    if not actual or not required.issubset(actual):
        raise ValueError("CLIP cache lacks the research_clip_cache_v3 contract")
    if actual["schema"] != RESEARCH_CLIP_CACHE_SCHEMA:
        raise ValueError("CLIP cache schema is not research_clip_cache_v3")
    if actual["cache_signature"] != expected["cache_signature"]:
        raise ValueError("Research CLIP cache signature mismatch")


def research_multiblock_clip_cache_metadata(
    csv_path,
    config,
    fold,
    split_role,
):
    """Build the closed S2 multi-block cache signature."""
    if not fold or not split_role:
        raise ValueError("S2 multi-block cache metadata requires fold and split_role")
    block_ids = tuple(config.get("clip_block_ids", RESEARCH_MULTIBLOCK_IDS))
    if block_ids != RESEARCH_MULTIBLOCK_IDS:
        raise ValueError(
            "S2 cache metadata requires frozen ordered block IDs "
            f"{RESEARCH_MULTIBLOCK_IDS}, got {block_ids}"
        )
    preprocess_signature = {
        "implementation": "open_clip:create_model_and_transforms",
        "image_size": config.get("image_size", 224),
        "interpolation": config.get("clip_interpolation", "bicubic"),
        "mean": config.get("clip_mean", OPENAI_CLIP_MEAN),
        "std": config.get("clip_std", OPENAI_CLIP_STD),
    }
    metadata = {
        "schema": RESEARCH_MULTIBLOCK_CACHE_SCHEMA,
        "encoder": config.get("clip_model", "ViT-B-32"),
        "pretrained": config.get("pretrained", "openai"),
        "open_clip_version": _open_clip_version(),
        "block_ids": list(block_ids),
        "block_indexing": "one_based_completed_transformer_blocks",
        "extraction_version": RESEARCH_MULTIBLOCK_EXTRACTION_VERSION,
        "preprocess_fingerprint": hashlib.sha256(
            json.dumps(preprocess_signature, sort_keys=True).encode("utf-8")
        ).hexdigest(),
        "manifest_path": str(Path(csv_path)),
        "manifest_sha256": manifest_fingerprint(csv_path),
        "fold": str(fold),
        "split_role": str(split_role),
        "feature_shape": list(RESEARCH_MULTIBLOCK_SHAPE),
    }
    metadata["cache_signature"] = hashlib.sha256(
        json.dumps(metadata, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return metadata


def research_multiblock_clip_cache_path(metadata):
    if metadata.get("schema") != RESEARCH_MULTIBLOCK_CACHE_SCHEMA:
        raise ValueError("Expected research_clip_multiblock_cache_v3 metadata")
    return (
        Path("outputs/research_v3/features")
        / "multiblock"
        / metadata["fold"]
        / f"{metadata['split_role']}_{metadata['cache_signature'][:16]}_multiblock_cls.pt"
    )


def validate_research_multiblock_clip_cache(cache, expected):
    actual = cache.get("metadata")
    required = {
        "schema",
        "encoder",
        "pretrained",
        "open_clip_version",
        "block_ids",
        "block_indexing",
        "extraction_version",
        "preprocess_fingerprint",
        "manifest_sha256",
        "fold",
        "split_role",
        "feature_shape",
        "cache_signature",
    }
    if not actual or not required.issubset(actual):
        raise ValueError(
            "CLIP cache lacks the research_clip_multiblock_cache_v3 contract"
        )
    if actual["schema"] != RESEARCH_MULTIBLOCK_CACHE_SCHEMA:
        raise ValueError("CLIP cache schema is not research_clip_multiblock_cache_v3")
    mismatched = sorted(
        key for key in required if actual.get(key) != expected.get(key)
    )
    if mismatched:
        raise ValueError(
            "S2 multi-block CLIP cache metadata mismatch: "
            + ", ".join(mismatched)
        )

    features = cache.get("features")
    labels = cache.get("labels")
    paths = cache.get("paths")
    if features is None or labels is None or paths is None:
        raise ValueError("S2 multi-block CLIP cache is incomplete")
    feature_tensor = torch.as_tensor(features)
    label_tensor = torch.as_tensor(labels)
    expected_feature_shape = (
        len(paths),
        *RESEARCH_MULTIBLOCK_SHAPE,
    )
    if tuple(feature_tensor.shape) != expected_feature_shape:
        raise ValueError(
            f"S2 multi-block cache feature shape mismatch: expected "
            f"{expected_feature_shape}, got {tuple(feature_tensor.shape)}"
        )
    if label_tensor.ndim != 1 or len(label_tensor) != len(paths):
        raise ValueError("S2 multi-block cache label/sample count mismatch")
    if not torch.isfinite(feature_tensor).all():
        raise ValueError("S2 multi-block cache contains non-finite features")
