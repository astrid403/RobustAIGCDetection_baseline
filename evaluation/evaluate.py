import argparse
from pathlib import Path
import sys

import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.csv_image_dataset import CsvImageDataset
from evaluation.error_analysis import save_error_cases
from evaluation.metrics import binary_metrics
from evaluation.plots import plot_confusion_matrix, plot_roc
from models.clip_mlp_detector import ClipMlpDetector, extract_clip_features, load_feature_cache, load_open_clip_model, save_feature_cache
from models.resnet_detector import build_resnet_detector
from training.losses import get_loss
from training.train import resnet_transforms


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@torch.no_grad()
def predict_resnet(config, checkpoint, csv_path, device):
    dataset = CsvImageDataset(csv_path, transform=resnet_transforms(config.get("image_size", 224), train=False))
    loader = DataLoader(dataset, batch_size=config.get("batch_size", 16), shuffle=False, num_workers=config.get("num_workers", 2))
    model = build_resnet_detector(config.get("model_name", "resnet50"), pretrained=False, freeze_backbone=False).to(device)
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state["model_state"])
    model.eval()
    labels, probs, paths = [], [], []
    for images, batch_labels, batch_paths in loader:
        logits = model(images.to(device)).view(-1)
        probs.extend(torch.sigmoid(logits).cpu().numpy().tolist())
        labels.extend([int(x) for x in batch_labels])
        paths.extend(batch_paths)
    return labels, probs, paths


@torch.no_grad()
def predict_clip_mlp(config, checkpoint, csv_path, device):
    exp = config["experiment_name"]
    cache_path = Path("outputs/features") / exp / "test_clip_features.pt"
    if config.get("cache_clip_features", True) and cache_path.exists():
        cache = load_feature_cache(cache_path)
    else:
        clip_model, preprocess = load_open_clip_model(config.get("clip_model", "ViT-B/32"), device)
        dataset = CsvImageDataset(csv_path, transform=preprocess)
        loader = DataLoader(dataset, batch_size=config.get("batch_size", 32), shuffle=False, num_workers=config.get("num_workers", 2))
        features, labels, paths = extract_clip_features(clip_model, loader, device)
        cache = {"features": features, "labels": labels, "paths": paths}
        if config.get("cache_clip_features", True):
            save_feature_cache(cache_path, features, labels, paths)
    feature_dim = cache["features"].shape[1]
    model = ClipMlpDetector(feature_dim, config.get("mlp_hidden_dim", 512), config.get("dropout", 0.2)).to(device)
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state["model_state"])
    model.eval()
    ds = TensorDataset(cache["features"], cache["labels"].float())
    loader = DataLoader(ds, batch_size=config.get("batch_size", 32), shuffle=False)
    labels, probs = [], []
    for features, batch_labels in loader:
        logits = model(features.to(device)).view(-1)
        probs.extend(torch.sigmoid(logits).cpu().numpy().tolist())
        labels.extend(batch_labels.numpy().astype(int).tolist())
    return labels, probs, cache["paths"]


def save_outputs(config, csv_path, labels, probs, paths):
    exp = config["experiment_name"]
    split_df = pd.read_csv(csv_path)
    meta = split_df.set_index("path", drop=False)
    preds = []
    for path, label, prob in zip(paths, labels, probs):
        pred = int(prob >= 0.5)
        row = meta.loc[path] if path in meta.index else {}
        preds.append(
            {
                "path": path,
                "label": int(label),
                "pred_label": pred,
                "fake_prob": prob,
                "dataset": row.get("dataset", config.get("dataset", "unknown")),
                "generator": row.get("generator", "unknown"),
                "source": row.get("source", "unknown"),
                "is_correct": pred == int(label),
            }
        )
    pred_df = pd.DataFrame(preds)
    metrics = binary_metrics(labels, probs)
    Path("outputs/metrics").mkdir(parents=True, exist_ok=True)
    Path("outputs/predictions").mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(Path("outputs/predictions") / f"{exp}_predictions.csv", index=False)
    metric_row = {
        "experiment_name": exp,
        "model": config.get("model_type", "resnet"),
        "train_dataset": config.get("dataset", "unknown"),
        "test_dataset": pred_df["dataset"].iloc[0] if not pred_df.empty else "unknown",
        "test_split": csv_path,
        "train_generators": ",".join(config.get("train_generators", [])),
        "test_generators": ",".join(sorted(pred_df["generator"].dropna().unique())) if not pred_df.empty else "",
        "num_real": int((pred_df["label"] == 0).sum()),
        "num_fake": int((pred_df["label"] == 1).sum()),
        "accuracy": metrics["accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "auroc": metrics["auroc"],
    }
    pd.DataFrame([metric_row]).to_csv(Path("outputs/metrics") / f"{exp}_metrics.csv", index=False)
    summary_path = Path("outputs/metrics/summary_all_experiments.csv")
    summary_row = {
        "experiment_name": exp,
        "model": config.get("model_type", "resnet"),
        "train_setting": config.get("train_csv", ""),
        "test_setting": csv_path,
        "accuracy": metrics["accuracy"],
        "f1": metrics["f1"],
        "auroc": metrics["auroc"],
        "notes": "",
    }
    old = pd.read_csv(summary_path) if summary_path.exists() else pd.DataFrame()
    pd.concat([old, pd.DataFrame([summary_row])], ignore_index=True).to_csv(summary_path, index=False)
    fig_dir = Path("outputs/figures") / exp
    plot_confusion_matrix(metrics["confusion_matrix"], fig_dir / "confusion_matrix.png")
    plot_roc(labels, probs, fig_dir / "roc_curve.png")
    save_error_cases(pred_df, exp)
    print(pd.DataFrame([metric_row]).to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained detector.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--test-csv", default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    csv_path = args.test_csv or config.get("test_csv") or config.get("val_csv")
    if not csv_path:
        raise ValueError("No test CSV provided. Add test_csv to config or pass --test-csv.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if config.get("model_type") == "clip_mlp":
        labels, probs, paths = predict_clip_mlp(config, args.checkpoint, csv_path, device)
    else:
        labels, probs, paths = predict_resnet(config, args.checkpoint, csv_path, device)
    save_outputs(config, csv_path, labels, probs, paths)


if __name__ == "__main__":
    main()

