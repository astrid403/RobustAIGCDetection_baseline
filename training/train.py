import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
from datetime import datetime, timezone

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
from data_pipeline.csv_image_dataset import CsvImageDataset
from data_pipeline.research_transforms import ResearchNprViewBuilder
from data_pipeline.fingerprints import (
    feature_cache_metadata,
    feature_cache_path,
    research_clip_cache_metadata,
    research_clip_cache_path,
    research_multiblock_clip_cache_metadata,
    research_multiblock_clip_cache_path,
    validate_feature_cache,
    validate_research_clip_cache,
    validate_research_multiblock_clip_cache,
)
from evaluation.metrics import binary_metrics
from evaluation.research_inference import (
    DEGRADATIONS,
    create_run_directories,
    load_best_checkpoint,
    predict_b2,
    predict_npr,
    predict_rine_lite,
    save_degradation_predictions,
    sha256_file,
    write_run_registry,
)
from evaluation.plots import plot_training_curves
from models.clip_mlp_detector import (
    build_clip_feature_detector,
    extract_clip_features,
    extract_clip_multiblock_features,
    load_feature_cache,
    load_open_clip_model,
    pair_clip_feature_caches,
    save_feature_cache,
)
from models.resnet_detector import build_resnet_detector
from models.npr_detector import build_npr_detector
from models.rine_lite_detector import build_rine_lite_detector
from training.engine import (
    evaluate_feature_loader,
    evaluate_loader,
    train_feature_epoch,
    train_npr_epoch,
    train_one_epoch,
)
from training.group_samplers import (
    DeterministicGroupBalancedSampler,
    validate_protocol_groups,
)
from training.losses import get_loss
from training.objectives import build_npr_objective
from training.contrastive_losses import RineLiteObjective


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def seed_everything(seed, deterministic=False):
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if deterministic:
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        torch.set_num_threads(1)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(bool(deterministic))


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


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


def load_training_checkpoint(path, model, optimizer, device):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    if optimizer is not None and checkpoint.get("optimizer_state"):
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    return checkpoint


class NprCsvDataset(torch.utils.data.Dataset):
    def __init__(self, csv_path, seed, training):
        self.base = CsvImageDataset(csv_path)
        self.seed = int(seed)
        self.training = bool(training)
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        image, label, sample_id = self.base[index]
        view = ResearchNprViewBuilder(
            global_seed=self.seed, epoch=self.epoch, training=self.training
        )(image, label, sample_id)
        return view.clean, view.label, view.sample_id


def _research_output_paths(config):
    root = Path(config.get("output_root", "outputs/research_v3"))
    exp = config["experiment_name"]
    return {
        "checkpoint": root / "checkpoints" / exp,
        "log": root / "logs" / exp,
        "metric": root / "metrics" / exp,
        "prediction": root / "predictions" / exp,
        "registry": root / "registries" / exp,
    }


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_value(*args):
    return subprocess.check_output(
        ["git", *args], cwd=PROJECT_ROOT, text=True
    ).strip()


def _research_identity(config):
    branch = _git_value("branch", "--show-current")
    if branch != config["branch"]:
        raise RuntimeError(
            f"Research config requires branch {config['branch']}, got {branch}"
        )
    return branch, _git_value("rev-parse", "HEAD")


def train_npr(config, device):
    if config.get("model_type") != "npr_resnet18":
        raise ValueError("NPR training requires model_type=npr_resnet18")
    if config.get("consistency_weight", 0.0) != 0.0:
        raise ValueError("Task 06 S1 main smoke must keep consistency disabled")
    if config.get("epochs", 1) > 1 and config.get("stage") == "smoke":
        raise ValueError("Task 06 smoke is limited to one epoch")

    train_dataset = NprCsvDataset(config["train_csv"], config["seed"], training=True)
    val_dataset = NprCsvDataset(config["val_csv"], config["seed"], training=False)
    frame = train_dataset.base.df
    validate_protocol_groups(frame["generator"], frame["label"])
    sampler = DeterministicGroupBalancedSampler(
        frame["generator"],
        frame["label"],
        config.get("samples_per_group_per_epoch", 500),
        config["seed"],
    )
    train_generator = torch.Generator(device="cpu").manual_seed(config["seed"])
    validation_generator = torch.Generator(device="cpu").manual_seed(config["seed"])
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.get("batch_size", 32),
        sampler=sampler,
        num_workers=config.get("num_workers", 0),
        generator=train_generator,
        worker_init_fn=seed_worker,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.get("batch_size", 32),
        shuffle=False,
        num_workers=config.get("num_workers", 0),
        generator=validation_generator,
        worker_init_fn=seed_worker,
    )
    model = build_npr_detector(config.get("pretrained", True)).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.get("learning_rate", 1e-4),
        weight_decay=config.get("weight_decay", 1e-4),
    )
    loss_fn = build_npr_objective(config.get("consistency_weight", 0.0))
    scaler = (
        torch.cuda.amp.GradScaler()
        if config.get("mixed_precision", True) and device.type == "cuda"
        else None
    )
    paths = create_run_directories(
        config.get("output_root", "outputs/research_v3"), config["experiment_name"]
    )
    with open(paths["checkpoints"] / "config_used.yaml", "w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)

    start_epoch = 1
    best_score = -1.0
    if config.get("resume_from"):
        checkpoint = load_training_checkpoint(
            config["resume_from"], model, optimizer, device
        )
        start_epoch = checkpoint["epoch"] + 1
        best_score = checkpoint["best_score"]

    rows = []
    run_started = datetime.now(timezone.utc).isoformat()
    for epoch in range(start_epoch, config.get("epochs", 1) + 1):
        train_dataset.set_epoch(epoch)
        sampler.set_epoch(epoch)
        train_loss = train_npr_epoch(
            model, train_loader, optimizer, loss_fn, device, scaler
        )
        val = evaluate_loader(model, val_loader, loss_fn, device)
        score = val["auroc"] if not np.isnan(val["auroc"]) else val["macro_f1"]
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            **{f"val_{key}": value for key, value in val.items()
               if key not in {"labels", "probs", "paths", "confusion_matrix"}},
        }
        if not all(np.isfinite(value) for value in row.values()):
            raise FloatingPointError("NPR smoke metrics contain NaN or Inf")
        rows.append(row)
        pd.DataFrame(rows).to_csv(paths["logs"] / "train_log.csv", index=False)
        save_checkpoint(
            paths["checkpoints"] / "last_model.pt",
            model,
            optimizer,
            epoch,
            best_score,
            config,
        )
        if score > best_score:
            best_score = score
            save_checkpoint(
                paths["checkpoints"] / "best_model.pt",
                model,
                optimizer,
                epoch,
                best_score,
                config,
            )
        pd.DataFrame([row]).to_csv(paths["metrics"] / "validation_metrics.csv", index=False)
        print(
            f"Epoch {epoch}: train_loss={train_loss:.4f} "
            f"val_macro_f1={val['macro_f1']:.4f} val_auroc={val['auroc']:.4f}"
        )
    fold = config.get("fold", "synthetic_smoke_only")
    checkpoint_path = paths["checkpoints"] / "best_model.pt"
    load_best_checkpoint(checkpoint_path, model, device)
    predictions = {
        degradation: predict_npr(
            model,
            config["val_csv"],
            fold,
            degradation,
            device,
            config.get("batch_size", 32),
        )
        for degradation in DEGRADATIONS
    }
    prediction_records, sample_order_hash = save_degradation_predictions(
        predictions, paths["predictions"]
    )
    metric_rows = []
    for degradation, frame in predictions.items():
        metrics = binary_metrics(frame["label"], frame["probability"])
        metric_rows.append(
            {
                "degradation": degradation,
                **{
                    key: value
                    for key, value in metrics.items()
                    if key != "confusion_matrix"
                },
            }
        )
    pd.DataFrame(metric_rows).to_csv(
        paths["metrics"] / "degradation_metrics.csv", index=False
    )
    branch, commit = _research_identity(config)
    write_run_registry(
        paths["registries"] / "run_registry.json",
        branch=branch,
        commit=commit,
        run_id=config["experiment_name"],
        fold=fold,
        config_path=paths["checkpoints"] / "config_used.yaml",
        split_paths={"train": config["train_csv"], "validation": config["val_csv"]},
        checkpoint_path=checkpoint_path,
        prediction_records=prediction_records,
        sample_order_sha256=sample_order_hash,
    )
    return rows


def _research_cache_or_extract(
    config, split_role, csv_path, preprocess, clip_model, device
):
    metadata = research_clip_cache_metadata(
        csv_path, config, config["fold"], split_role
    )
    path = research_clip_cache_path(metadata)
    if path.exists():
        cache = load_feature_cache(path)
        validate_research_clip_cache(cache, metadata)
    else:
        loader = make_loader(
            csv_path,
            preprocess,
            config.get("batch_size", 32),
            False,
            config.get("num_workers", 0),
        )
        features, labels, sample_ids = extract_clip_features(
            clip_model, loader, device, feature_mode="final"
        )
        save_feature_cache(path, features, labels, sample_ids, metadata)
        cache = {
            "features": features,
            "labels": labels,
            "paths": sample_ids,
            "sample_ids": sample_ids,
            "metadata": metadata,
        }
    return cache, path


def train_research_b2(config, device):
    if config.get("model_type") != "clip_mlp" or config.get("clip_feature_mode") != "final":
        raise ValueError("B2-v3 requires clip_mlp with final CLIP features")
    paths = create_run_directories(
        config.get("output_root", "outputs/research_v3"), config["experiment_name"]
    )
    config_path = paths["checkpoints"] / "config_used.yaml"
    with open(config_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
    encoder, preprocess = load_open_clip_model(
        config.get("clip_model", "ViT-B-32"),
        device,
        config.get("pretrained", "openai"),
    )
    train_cache, train_cache_path = _research_cache_or_extract(
        config, "train", config["train_csv"], preprocess, encoder, device
    )
    val_cache, val_cache_path = _research_cache_or_extract(
        config, "validation", config["val_csv"], preprocess, encoder, device
    )
    train_dataset = TensorDataset(
        train_cache["features"], train_cache["labels"].float()
    )
    val_dataset = TensorDataset(val_cache["features"], val_cache["labels"].float())
    generator = torch.Generator(device="cpu").manual_seed(config["seed"])
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=0
    )
    model = build_clip_feature_detector(
        "clip_mlp",
        train_cache["features"].shape[1],
        config["mlp_hidden_dim"],
        config["dropout"],
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )
    loss_fn = get_loss()
    rows, best_score = [], -1.0
    for epoch in range(1, config["epochs"] + 1):
        train_loss = train_feature_epoch(
            model, train_loader, optimizer, loss_fn, device
        )
        validation = evaluate_feature_loader(model, val_loader, loss_fn, device)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            **{
                f"val_{key}": value
                for key, value in validation.items()
                if key not in {"labels", "probs", "confusion_matrix"}
            },
        }
        rows.append(row)
        pd.DataFrame(rows).to_csv(paths["logs"] / "train_log.csv", index=False)
        score = validation["auroc"]
        save_checkpoint(
            paths["checkpoints"] / "last_model.pt",
            model,
            optimizer,
            epoch,
            best_score,
            config,
        )
        if score > best_score:
            best_score = score
            save_checkpoint(
                paths["checkpoints"] / "best_model.pt",
                model,
                optimizer,
                epoch,
                best_score,
                config,
            )
    checkpoint_path = paths["checkpoints"] / "best_model.pt"
    load_best_checkpoint(checkpoint_path, model, device)
    predictions = {
        degradation: predict_b2(
            encoder,
            model,
            preprocess,
            config["val_csv"],
            config["fold"],
            degradation,
            device,
            config["batch_size"],
        )
        for degradation in DEGRADATIONS
    }
    prediction_records, sample_order_hash = save_degradation_predictions(
        predictions, paths["predictions"]
    )
    metric_rows = []
    for degradation, frame in predictions.items():
        metrics = binary_metrics(frame["label"], frame["probability"])
        metric_rows.append(
            {
                "degradation": degradation,
                **{
                    key: value
                    for key, value in metrics.items()
                    if key != "confusion_matrix"
                },
            }
        )
    pd.DataFrame(metric_rows).to_csv(
        paths["metrics"] / "degradation_metrics.csv", index=False
    )
    cache_records = {
        "train": {"path": str(train_cache_path), "sha256": sha256_file(train_cache_path)},
        "validation": {
            "path": str(val_cache_path),
            "sha256": sha256_file(val_cache_path),
        },
    }
    branch, commit = _research_identity(config)
    write_run_registry(
        paths["registries"] / "run_registry.json",
        branch=branch,
        commit=commit,
        run_id=config["experiment_name"],
        fold=config["fold"],
        config_path=config_path,
        split_paths={"train": config["train_csv"], "validation": config["val_csv"]},
        checkpoint_path=checkpoint_path,
        prediction_records=prediction_records,
        sample_order_sha256=sample_order_hash,
        cache_records=cache_records,
    )
    return rows


def _multiblock_cache_or_extract(
    config, split_role, csv_path, preprocess, clip_model, device
):
    metadata = research_multiblock_clip_cache_metadata(
        csv_path, config, config["fold"], split_role
    )
    path = research_multiblock_clip_cache_path(metadata)
    if path.exists():
        cache = load_feature_cache(path)
        validate_research_multiblock_clip_cache(cache, metadata)
    else:
        loader = make_loader(
            csv_path, preprocess, config["batch_size"], False, 0
        )
        features, labels, sample_ids = extract_clip_multiblock_features(
            clip_model, loader, device
        )
        save_feature_cache(path, features, labels, sample_ids, metadata)
        cache = {
            "features": features,
            "labels": labels,
            "paths": sample_ids,
            "sample_ids": sample_ids,
            "metadata": metadata,
        }
    return cache, path


def _rine_lite_epoch(model, loader, objective, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss, labels_all, probabilities_all, importance_all = 0.0, [], [], []
    for features, labels in loader:
        features = features.to(device)
        labels = labels.to(device).long()
        if training:
            optimizer.zero_grad(set_to_none=True)
        output = model.forward_with_aux(features)
        losses = objective(output["logits"], output["embedding"], labels)
        if training:
            losses["total"].backward()
            optimizer.step()
        total_loss += float(losses["total"].detach()) * len(labels)
        labels_all.extend(labels.cpu().tolist())
        probabilities_all.extend(torch.sigmoid(output["logits"]).detach().cpu().tolist())
        importance_all.append(output["importance"].detach().cpu())
    metrics = binary_metrics(labels_all, probabilities_all)
    return {
        "loss": total_loss / len(labels_all),
        **{key: value for key, value in metrics.items() if key != "confusion_matrix"},
        "importance_mean": torch.cat(importance_all).mean(dim=0).tolist(),
    }


def _validate_s2_pilot_config(config):
    required = {
        "model_type": "rine_lite",
        "clip_block_ids": [3, 6, 9, 12],
        "batch_size": 32,
        "epochs": 10,
        "samples_per_group_per_epoch": 500,
        "learning_rate": 0.001,
        "weight_decay": 0.01,
        "supcon_weight": 0.1,
        "supcon_temperature": 0.07,
        "seed": 42,
        "num_workers": 0,
    }
    mismatches = {
        key: (config.get(key), value)
        for key, value in required.items()
        if config.get(key) != value
    }
    if mismatches:
        raise ValueError(f"S2 pilot config violates frozen contract: {mismatches}")


def train_research_rine_lite(config, device):
    _validate_s2_pilot_config(config)
    paths = create_run_directories(
        config.get("output_root", "outputs/research_v3"), config["experiment_name"]
    )
    config_path = paths["checkpoints"] / "config_used.yaml"
    with open(config_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
    encoder, preprocess = load_open_clip_model(
        config["clip_model"], device, config["pretrained"]
    )
    train_cache, train_cache_path = _multiblock_cache_or_extract(
        config, "train", config["train_csv"], preprocess, encoder, device
    )
    val_cache, val_cache_path = _multiblock_cache_or_extract(
        config, "validation", config["val_csv"], preprocess, encoder, device
    )
    train_frame = pd.read_csv(config["train_csv"])
    validate_protocol_groups(train_frame["generator"], train_frame["label"])
    sampler = DeterministicGroupBalancedSampler(
        train_frame["generator"],
        train_frame["label"],
        config["samples_per_group_per_epoch"],
        config["seed"],
    )
    train_loader = DataLoader(
        TensorDataset(train_cache["features"], train_cache["labels"].long()),
        batch_size=32,
        sampler=sampler,
        num_workers=0,
    )
    val_loader = DataLoader(
        TensorDataset(val_cache["features"], val_cache["labels"].long()),
        batch_size=32,
        shuffle=False,
        num_workers=0,
    )
    model = build_rine_lite_detector().to(device)
    objective = RineLiteObjective()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=0.001, weight_decay=0.01
    )
    rows, best_score = [], -1.0
    for epoch in range(1, 11):
        sampler.set_epoch(epoch)
        train_result = _rine_lite_epoch(
            model, train_loader, objective, device, optimizer
        )
        with torch.no_grad():
            validation = _rine_lite_epoch(
                model, val_loader, objective, device
            )
        row = {
            "epoch": epoch,
            "train_loss": train_result["loss"],
            **{
                f"val_{key}": value
                for key, value in validation.items()
                if key != "importance_mean"
            },
            **{
                f"importance_block_{block}": value
                for block, value in zip(
                    (3, 6, 9, 12), validation["importance_mean"]
                )
            },
        }
        if not all(np.isfinite(value) for value in row.values()):
            raise FloatingPointError("S2 training metrics contain NaN or Inf")
        rows.append(row)
        pd.DataFrame(rows).to_csv(paths["logs"] / "train_log.csv", index=False)
        score = validation["auroc"]
        save_checkpoint(
            paths["checkpoints"] / "last_model.pt",
            model, optimizer, epoch, best_score, config
        )
        if score > best_score:
            best_score = score
            save_checkpoint(
                paths["checkpoints"] / "best_model.pt",
                model, optimizer, epoch, best_score, config
            )
        print(
            f"Epoch {epoch}: train_loss={train_result['loss']:.4f} "
            f"val_macro_f1={validation['macro_f1']:.4f} "
            f"val_auroc={validation['auroc']:.4f}"
        )

    checkpoint_path = paths["checkpoints"] / "best_model.pt"
    load_best_checkpoint(checkpoint_path, model, device)
    predictions, importance_frames = {}, {}
    for degradation in DEGRADATIONS:
        predictions[degradation], importance_frames[degradation] = predict_rine_lite(
            encoder, model, preprocess, config["val_csv"], config["fold"],
            degradation, device, 32
        )
    prediction_records, sample_order_hash = save_degradation_predictions(
        predictions, paths["predictions"]
    )
    importance_records = {}
    for degradation, frame in importance_frames.items():
        path = paths["predictions"] / f"{degradation}_block_importance.csv"
        frame.to_csv(path, index=False)
        importance_records[degradation] = {
            "path": str(path), "sha256": sha256_file(path)
        }
    metric_rows = []
    for degradation, frame in predictions.items():
        metrics = binary_metrics(frame["label"], frame["probability"])
        metric_rows.append({
            "degradation": degradation,
            **{key: value for key, value in metrics.items() if key != "confusion_matrix"},
        })
    pd.DataFrame(metric_rows).to_csv(
        paths["metrics"] / "degradation_metrics.csv", index=False
    )
    cache_records = {
        "train": {"path": str(train_cache_path), "sha256": sha256_file(train_cache_path)},
        "validation": {"path": str(val_cache_path), "sha256": sha256_file(val_cache_path)},
    }
    branch, commit = _research_identity(config)
    write_run_registry(
        paths["registries"] / "run_registry.json",
        branch=branch, commit=commit, run_id=config["experiment_name"],
        fold=config["fold"], config_path=config_path,
        split_paths={"train": config["train_csv"], "validation": config["val_csv"]},
        checkpoint_path=checkpoint_path, prediction_records=prediction_records,
        sample_order_sha256=sample_order_hash, cache_records=cache_records,
        importance_records=importance_records,
    )
    return rows


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


def cache_or_extract(config, split_name, csv_path, preprocess, clip_model, device, feature_mode=None):
    exp = config["experiment_name"]
    mode = feature_mode or config.get("clip_feature_mode", "final")
    mode_config = dict(config)
    mode_config["clip_feature_mode"] = mode
    metadata = feature_cache_metadata(csv_path, mode_config)
    cache_path = feature_cache_path(exp, split_name, metadata)
    if config.get("cache_clip_features", True) and cache_path.exists():
        cache = load_feature_cache(cache_path)
        validate_feature_cache(cache, metadata)
        return cache
    loader = make_loader(csv_path, preprocess, config.get("batch_size", 32), False, config.get("num_workers", 2))
    features, labels, paths = extract_clip_features(
        clip_model, loader, device, feature_mode=mode
    )
    if config.get("cache_clip_features", True):
        save_feature_cache(cache_path, features, labels, paths, metadata)
    return {"features": features, "labels": labels, "paths": paths, "sample_ids": paths, "metadata": metadata}


def dual_level_cache_or_extract(config, split_name, csv_path, preprocess, clip_model, device):
    modes = config.get("clip_feature_modes", ["final", "penultimate"])
    if modes != ["final", "penultimate"]:
        raise ValueError("Dual-level fusion requires clip_feature_modes: [final, penultimate].")
    final = cache_or_extract(
        config, split_name, csv_path, preprocess, clip_model, device, feature_mode="final"
    )
    penultimate = cache_or_extract(
        config, split_name, csv_path, preprocess, clip_model, device, feature_mode="penultimate"
    )
    return pair_clip_feature_caches(final, penultimate)


def train_clip_classifier(config, device):
    clip_model, preprocess = load_open_clip_model(
        config.get("clip_model", "ViT-B/32"), device, config.get("pretrained", "openai")
    )
    if config.get("model_type") == "clip_fusion":
        train_cache = dual_level_cache_or_extract(
            config, "train", config["train_csv"], preprocess, clip_model, device
        )
        val_cache = dual_level_cache_or_extract(
            config, "val", config["val_csv"], preprocess, clip_model, device
        )
    else:
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
        config.get("fusion_hidden_dim", 256),
        train_cache.get("feature_dims"),
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
            "val_balanced_accuracy": val["balanced_accuracy"],
            "val_precision": val["precision"],
            "val_fake_recall": val["fake_recall"],
            "val_real_recall": val["real_recall"],
            "val_binary_f1": val["binary_f1"],
            "val_macro_f1": val["macro_f1"],
            "val_recall": val["recall"],
            "val_f1": val["f1"],
            "val_auroc": val["auroc"],
            "val_auprc": val["auprc"],
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
    seed_everything(
        config.get("seed", 42),
        deterministic=config.get("stage") == "smoke",
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if config.get("research_v3") and config.get("model_type") == "rine_lite":
        train_research_rine_lite(config, device)
    elif config.get("research_v3") and config.get("model_type") == "clip_mlp":
        train_research_b2(config, device)
    elif config.get("model_type") == "npr_resnet18":
        train_npr(config, device)
    elif config.get("model_type", "resnet") in {"clip_mlp", "clip_linear", "clip_fusion"}:
        train_clip_classifier(config, device)
    else:
        train_resnet(config, device)


if __name__ == "__main__":
    main()
