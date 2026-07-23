import argparse
from pathlib import Path
import random
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_pipeline.dataset_factory import create_image_dataset
from data_pipeline.fingerprints import feature_cache_metadata, feature_cache_path, validate_feature_cache
from evaluation.plots import plot_training_curves
from models.clip_mlp_detector import (
    build_clip_feature_detector,
    extract_clip_features,
    load_feature_cache,
    load_open_clip_model,
    save_feature_cache,
)
from models.resnet_detector import build_resnet_detector
from training.engine import evaluate_feature_loader, evaluate_loader, train_feature_epoch, train_one_epoch
from training.losses import get_loss


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resnet_transforms(image_size, train):
    if train:
        aug = [
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
        ]
    else:
        aug = [transforms.Resize((image_size, image_size))]
    return transforms.Compose(
        aug
        + [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def make_loader(csv_path, transform, batch_size, shuffle, workers, max_samples_per_class=None):
    dataset = create_image_dataset(csv_path, transform=transform, max_samples_per_class=max_samples_per_class)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=workers, pin_memory=torch.cuda.is_available())


def save_checkpoint(path, model, optimizer, epoch, best_score, config):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict() if optimizer else None,
            "epoch": epoch,
            "best_score": best_score,
            "config": config,
        },
        path,
    )


def train_resnet(config, device):
    image_size = config.get("image_size", 224)
    batch_size = config.get("batch_size", 16)
    workers = config.get("num_workers", 2)
    train_loader = make_loader(
        config["train_csv"],
        resnet_transforms(image_size, train=True),
        batch_size,
        True,
        workers,
        config.get("max_samples_per_class"),
    )
    val_loader = make_loader(
        config["val_csv"],
        resnet_transforms(image_size, train=False),
        batch_size,
        False,
        workers,
        config.get("max_samples_per_class"),
    )
    model = build_resnet_detector(
        config.get("model_name", "resnet50"),
        config.get("pretrained", True),
        config.get("freeze_backbone", False),
    ).to(device)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=config.get("learning_rate", 1e-4))
    loss_fn = get_loss()
    scaler = torch.cuda.amp.GradScaler() if config.get("mixed_precision", True) and device.type == "cuda" else None
    return run_epoch_loop(config, model, optimizer, loss_fn, train_loader, val_loader, device, scaler)


def cache_or_extract(config, split_name, csv_path, preprocess, clip_model, device):
    exp = config["experiment_name"]
    metadata = feature_cache_metadata(csv_path, config)
    cache_path = feature_cache_path(exp, split_name, metadata)
    if config.get("cache_clip_features", True) and cache_path.exists():
        cache = load_feature_cache(cache_path)
        validate_feature_cache(cache, metadata)
        return cache
    loader = make_loader(csv_path, preprocess, config.get("batch_size", 32), False, config.get("num_workers", 2))
    features, labels, paths = extract_clip_features(
        clip_model, loader, device, feature_mode=config.get("clip_feature_mode", "final")
    )
    if config.get("cache_clip_features", True):
        save_feature_cache(cache_path, features, labels, paths, metadata)
    return {"features": features, "labels": labels, "paths": paths, "sample_ids": paths, "metadata": metadata}


def train_clip_classifier(config, device):
    clip_model, preprocess = load_open_clip_model(
        config.get("clip_model", "ViT-B/32"), device, config.get("pretrained", "openai")
    )
    train_cache = cache_or_extract(config, "train", config["train_csv"], preprocess, clip_model, device)
    val_cache = cache_or_extract(config, "val", config["val_csv"], preprocess, clip_model, device)
    train_ds = TensorDataset(train_cache["features"], train_cache["labels"].float())
    val_ds = TensorDataset(val_cache["features"], val_cache["labels"].float())
    train_loader = DataLoader(train_ds, batch_size=config.get("batch_size", 32), shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config.get("batch_size", 32), shuffle=False)
    feature_dim = train_cache["features"].shape[1]
    model = build_clip_feature_detector(
        config.get("model_type", "clip_mlp"),
        feature_dim,
        config.get("mlp_hidden_dim", 512),
        config.get("dropout", 0.2),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.get("learning_rate", 1e-3))
    loss_fn = get_loss()
    return run_epoch_loop(config, model, optimizer, loss_fn, train_loader, val_loader, device, scaler=None, feature_mode=True)


def train_clip_mlp(config, device):
    """Backward-compatible entry point for existing callers."""
    return train_clip_classifier(config, device)


def run_epoch_loop(config, model, optimizer, loss_fn, train_loader, val_loader, device, scaler=None, feature_mode=False):
    exp = config["experiment_name"]
    ckpt_dir = Path("outputs/checkpoints") / exp
    log_dir = Path("outputs/logs") / exp
    fig_dir = Path("outputs/figures") / exp
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    with open(ckpt_dir / "config_used.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)

    start_epoch = 1
    best_score = -1.0
    if config.get("resume_from"):
        checkpoint = torch.load(config["resume_from"], map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        if checkpoint.get("optimizer_state"):
            optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_epoch = checkpoint.get("epoch", 0) + 1
        best_score = checkpoint.get("best_score", -1.0)

    rows = []
    for epoch in range(start_epoch, config.get("epochs", 1) + 1):
        if feature_mode:
            train_loss = train_feature_epoch(model, train_loader, optimizer, loss_fn, device)
            val = evaluate_feature_loader(model, val_loader, loss_fn, device)
        else:
            train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device, scaler)
            val = evaluate_loader(model, val_loader, loss_fn, device)
        score = val["auroc"] if not np.isnan(val["auroc"]) else val["f1"]
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val["loss"],
            "val_accuracy": val["accuracy"],
            "val_precision": val["precision"],
            "val_recall": val["recall"],
            "val_f1": val["f1"],
            "val_auroc": val["auroc"],
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        rows.append(row)
        pd.DataFrame(rows).to_csv(log_dir / "train_log.csv", index=False)
        save_checkpoint(ckpt_dir / "last_model.pt", model, optimizer, epoch, best_score, config)
        if score > best_score:
            best_score = score
            save_checkpoint(ckpt_dir / "best_model.pt", model, optimizer, epoch, best_score, config)
        print(f"Epoch {epoch}: train_loss={train_loss:.4f} val_f1={val['f1']:.4f} val_auroc={val['auroc']:.4f}")

    log_df = pd.read_csv(log_dir / "train_log.csv")
    plot_training_curves(log_df, fig_dir)


def main():
    parser = argparse.ArgumentParser(description="Train a detector from a YAML config.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    seed_everything(config.get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if config.get("model_type", "resnet") in {"clip_mlp", "clip_linear"}:
        train_clip_classifier(config, device)
    else:
        train_resnet(config, device)


if __name__ == "__main__":
    main()
