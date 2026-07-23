"""Compact frozen-feature robustness evaluation for Part 09."""

import argparse
from pathlib import Path
import sys
import time

import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_pipeline.dataset_factory import create_image_dataset
from evaluation.metrics import binary_metrics
from models.clip_mlp_detector import (
    build_clip_feature_detector,
    extract_clip_features,
    load_open_clip_model,
)
from robustness.transforms import robustness_pil_ops


METRICS = (
    "accuracy", "balanced_accuracy", "precision", "real_recall", "fake_recall",
    "binary_f1", "macro_f1", "auroc", "auprc",
)


def synchronize():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


@torch.no_grad()
def classify(features, labels, checkpoint, config, device):
    model = build_clip_feature_detector(
        "clip_mlp", features.shape[1], config.get("mlp_hidden_dim", 512),
        config.get("dropout", 0.2),
    ).to(device)
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state["model_state"])
    model.eval()
    loader = DataLoader(
        TensorDataset(features, labels.float()),
        batch_size=config["batch_size"], shuffle=False,
    )
    probs = []
    synchronize()
    start = time.perf_counter()
    for batch_features, _ in loader:
        logits = model(batch_features.to(device)).view(-1)
        probs.extend(torch.sigmoid(logits).cpu().tolist())
    synchronize()
    return probs, time.perf_counter() - start, model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--clean-summary", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    clean = pd.read_csv(args.clean_summary)
    clean = clean[clean["scope"] == "genimage_unseen"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clip_model, preprocess = load_open_clip_model("ViT-B-32", device, "openai")
    clip_total = sum(p.numel() for p in clip_model.parameters())
    rows = []

    for role, spec in config["models"].items():
        for seed in (42, 43, 44):
            source = clean[(clean.role == role) & (clean.seed == seed)].iloc[0]
            row = {
                "role": role, "seed": seed, "transform": "clean",
                "feature_mode": spec["feature_mode"], "threshold": 0.5,
                "feature_seconds": float("nan"), "head_seconds": float("nan"),
                "images_per_second": float("nan"),
                "peak_gpu_memory_mib": float("nan"),
                "total_parameters": None, "trainable_parameters": None,
            }
            row.update({metric: source[metric] for metric in METRICS})
            rows.append(row)

        for transform_name in config["transforms"]:
            transform = transforms.Compose(
                robustness_pil_ops(
                    transform_name, config["jpeg_quality"],
                    config["resize_scale"], config["blur_radius"],
                ) + [preprocess]
            )
            dataset = create_image_dataset(config["test_csv"], transform=transform)
            loader = DataLoader(
                dataset, batch_size=config["batch_size"], shuffle=False,
                num_workers=config["num_workers"],
            )
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            synchronize()
            start = time.perf_counter()
            features, labels, paths = extract_clip_features(
                clip_model, loader, device, feature_mode=spec["feature_mode"]
            )
            synchronize()
            feature_seconds = time.perf_counter() - start
            peak_mib = (
                torch.cuda.max_memory_allocated() / 1024**2
                if torch.cuda.is_available() else float("nan")
            )
            if len(paths) != 4000 or len(set(paths)) != 4000:
                raise ValueError("Robustness manifest must produce 4,000 unique samples.")
            for seed in (42, 43, 44):
                checkpoint = spec["checkpoints"][seed]
                probs, head_seconds, model = classify(
                    features, labels, checkpoint, config, device
                )
                metrics = binary_metrics(labels.numpy(), probs, threshold=0.5)
                head_total = sum(p.numel() for p in model.parameters())
                row = {
                    "role": role, "seed": seed, "transform": transform_name,
                    "feature_mode": spec["feature_mode"], "threshold": 0.5,
                    "feature_seconds": feature_seconds,
                    "head_seconds": head_seconds,
                    "images_per_second": len(paths) / (feature_seconds + head_seconds),
                    "peak_gpu_memory_mib": peak_mib,
                    "total_parameters": clip_total + head_total,
                    "trainable_parameters": head_total,
                }
                row.update({metric: metrics[metric] for metric in METRICS})
                rows.append(row)

    frame = pd.DataFrame(rows)
    for metric in ("auroc", "auprc", "balanced_accuracy"):
        clean_values = frame[frame["transform"] == "clean"].set_index(["role", "seed"])[metric]
        frame[f"{metric}_delta_from_clean"] = frame.apply(
            lambda row: row[metric] - clean_values.loc[(row.role, row.seed)], axis=1
        )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
