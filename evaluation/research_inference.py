"""Best-checkpoint inference and provenance for Protocol-v3 pilots."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset

from data_pipeline.csv_image_dataset import CsvImageDataset
from data_pipeline.research_transforms import ResearchNprViewBuilder, apply_degradation
from models.clip_mlp_detector import (
    encode_clip_image_features,
    encode_clip_multiblock_cls,
)


DEGRADATIONS = ("clean", "jpeg", "resize", "blur")
PREDICTION_COLUMNS = ("sample_id", "fold", "label", "probability")


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def create_run_directories(output_root, run_id):
    paths = {
        name: Path(output_root) / name / run_id
        for name in ("checkpoints", "logs", "metrics", "predictions", "registries")
    }
    existing = [str(path) for path in paths.values() if path.exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite research run directories: {existing}")
    for path in paths.values():
        path.mkdir(parents=True)
    return paths


class DegradedImageDataset(Dataset):
    def __init__(self, csv_path, degradation, transform=None, npr=False):
        if degradation not in DEGRADATIONS:
            raise ValueError(f"Unknown degradation: {degradation}")
        self.base = CsvImageDataset(csv_path)
        self.degradation = degradation
        self.transform = transform
        self.npr = bool(npr)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        image, label, sample_id = self.base[index]
        image = apply_degradation(image, self.degradation)
        if self.npr:
            tensor = ResearchNprViewBuilder(
                global_seed=0,
                epoch=0,
                training=False,
                forced_degradation=None,
            )(image, label, sample_id).clean
        else:
            if self.transform is None:
                raise ValueError("CLIP inference requires its model-provided preprocess")
            tensor = self.transform(image)
        return tensor, label, sample_id


@torch.no_grad()
def predict_npr(model, csv_path, fold, degradation, device, batch_size=32):
    dataset = DegradedImageDataset(csv_path, degradation, npr=True)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    model.eval()
    rows = []
    for images, labels, sample_ids in loader:
        probabilities = torch.sigmoid(model(images.to(device)).view(-1)).cpu().tolist()
        rows.extend(
            {
                "sample_id": str(sample_id),
                "fold": str(fold),
                "label": int(label),
                "probability": float(probability),
            }
            for sample_id, label, probability in zip(sample_ids, labels, probabilities)
        )
    return pd.DataFrame(rows, columns=PREDICTION_COLUMNS)


@torch.no_grad()
def predict_b2(
    encoder,
    head,
    preprocess,
    csv_path,
    fold,
    degradation,
    device,
    batch_size=32,
):
    dataset = DegradedImageDataset(csv_path, degradation, transform=preprocess)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    encoder.eval()
    head.eval()
    rows = []
    for images, labels, sample_ids in loader:
        features = encode_clip_image_features(encoder, images.to(device), "final")
        probabilities = torch.sigmoid(head(features).view(-1)).cpu().tolist()
        rows.extend(
            {
                "sample_id": str(sample_id),
                "fold": str(fold),
                "label": int(label),
                "probability": float(probability),
            }
            for sample_id, label, probability in zip(sample_ids, labels, probabilities)
        )
    return pd.DataFrame(rows, columns=PREDICTION_COLUMNS)


@torch.no_grad()
def predict_rine_lite(
    encoder,
    head,
    preprocess,
    csv_path,
    fold,
    degradation,
    device,
    batch_size=32,
):
    dataset = DegradedImageDataset(csv_path, degradation, transform=preprocess)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    encoder.eval()
    head.eval()
    rows, importance_rows = [], []
    for images, labels, sample_ids in loader:
        features = encode_clip_multiblock_cls(encoder, images.to(device))
        output = head.forward_with_aux(features)
        probabilities = torch.sigmoid(output["logits"]).cpu().tolist()
        importance = output["importance"].cpu().tolist()
        for sample_id, label, probability, weights in zip(
            sample_ids, labels, probabilities, importance
        ):
            rows.append(
                {
                    "sample_id": str(sample_id),
                    "fold": str(fold),
                    "label": int(label),
                    "probability": float(probability),
                }
            )
            importance_rows.append(
                {
                    "sample_id": str(sample_id),
                    "fold": str(fold),
                    "block_3": float(weights[0]),
                    "block_6": float(weights[1]),
                    "block_9": float(weights[2]),
                    "block_12": float(weights[3]),
                }
            )
    return (
        pd.DataFrame(rows, columns=PREDICTION_COLUMNS),
        pd.DataFrame(importance_rows),
    )


def save_degradation_predictions(predictions, prediction_dir):
    records = {}
    reference = None
    for degradation in DEGRADATIONS:
        frame = predictions[degradation]
        if tuple(frame.columns) != PREDICTION_COLUMNS:
            raise ValueError("Prediction schema must be sample_id, fold, label, probability")
        identity = frame[["sample_id", "fold", "label"]].reset_index(drop=True)
        if identity["sample_id"].duplicated().any():
            raise ValueError("Prediction sample IDs must be unique within a held-out fold")
        if reference is None:
            reference = identity
        elif not identity.equals(reference):
            raise ValueError("Degradation predictions have misaligned sample order")
        path = Path(prediction_dir) / f"{degradation}.csv"
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite prediction: {path}")
        frame.to_csv(path, index=False)
        records[degradation] = {"path": str(path), "sha256": sha256_file(path)}
    sample_order_hash = hashlib.sha256(
        "\n".join(reference["sample_id"]).encode("utf-8")
    ).hexdigest()
    return records, sample_order_hash


def write_run_registry(
    registry_path,
    *,
    branch,
    commit,
    run_id,
    fold,
    config_path,
    split_paths,
    checkpoint_path,
    prediction_records,
    sample_order_sha256,
    cache_records=None,
    importance_records=None,
):
    registry_path = Path(registry_path)
    if registry_path.exists():
        raise FileExistsError(f"Refusing to overwrite registry: {registry_path}")
    registry = {
        "schema": "research_logo_run_registry_v3",
        "branch": branch,
        "commit": commit,
        "run_id": run_id,
        "fold": fold,
        "config": {"path": str(config_path), "sha256": sha256_file(config_path)},
        "splits": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in sorted(split_paths.items())
        },
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": sha256_file(checkpoint_path),
        },
        "predictions": prediction_records,
        "sample_order_sha256": sample_order_sha256,
        "feature_caches": cache_records or {},
        "block_importance": importance_records or {},
        "checkpoint_selection": {
            "split": "held_out_fold_validation",
            "metric": "auroc",
            "mode": "max",
            "tie_break": "earliest_epoch",
        },
        "genimage_unseen_accessed": False,
        "defactify_accessed": False,
    }
    registry_path.write_text(
        json.dumps(registry, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return registry


def load_best_checkpoint(checkpoint_path, model, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return checkpoint
