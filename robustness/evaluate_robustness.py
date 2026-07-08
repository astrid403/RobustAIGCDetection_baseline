import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.csv_image_dataset import CsvImageDataset
from evaluation.metrics import binary_metrics
from models.clip_mlp_detector import ClipMlpDetector, extract_clip_features, load_open_clip_model
from models.resnet_detector import build_resnet_detector
from robustness.transforms import robustness_pil_ops, robustness_transform


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@torch.no_grad()
def predict_resnet(config, transform_name, model, csv_path, device):
    transform = robustness_transform(
        transform_name,
        config.get("image_size", 224),
        config.get("jpeg_quality", 70),
        config.get("resize_scale", 0.5),
        config.get("blur_radius", 1.0),
    )
    dataset = CsvImageDataset(csv_path, transform=transform, max_samples_per_class=config.get("max_samples_per_class"))
    loader = DataLoader(dataset, batch_size=config.get("batch_size", 16), shuffle=False, num_workers=config.get("num_workers", 2))
    labels, probs = [], []
    for images, batch_labels, _ in loader:
        logits = model(images.to(device)).view(-1)
        probs.extend(torch.sigmoid(logits).cpu().numpy().tolist())
        labels.extend([int(x) for x in batch_labels])
    return binary_metrics(labels, probs)


@torch.no_grad()
def predict_clip_mlp(config, transform_name, clip_model, mlp_model, preprocess, csv_path, device):
    transform = transforms.Compose(
        robustness_pil_ops(
            transform_name,
            config.get("jpeg_quality", 70),
            config.get("resize_scale", 0.5),
            config.get("blur_radius", 1.0),
        )
        + [preprocess]
    )
    dataset = CsvImageDataset(csv_path, transform=transform, max_samples_per_class=config.get("max_samples_per_class"))
    loader = DataLoader(dataset, batch_size=config.get("batch_size", 32), shuffle=False, num_workers=config.get("num_workers", 2))
    features, labels_tensor, _ = extract_clip_features(clip_model, loader, device)
    ds = torch.utils.data.TensorDataset(features, labels_tensor.float())
    feature_loader = DataLoader(ds, batch_size=config.get("batch_size", 32), shuffle=False)
    labels, probs = [], []
    for batch_features, batch_labels in feature_loader:
        logits = mlp_model(batch_features.to(device)).view(-1)
        probs.extend(torch.sigmoid(logits).cpu().numpy().tolist())
        labels.extend(batch_labels.numpy().astype(int).tolist())
    return binary_metrics(labels, probs)


def plot_bar(df, out_path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(df["transform"], df["processed_auroc"])
    ax.set_ylabel("AUROC")
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Evaluate post-processing robustness for a ResNet checkpoint.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    checkpoint = config.get("checkpoint")
    if not checkpoint:
        raise ValueError("robustness_eval.yaml must set checkpoint.")
    csv_path = config.get("test_csv")
    if not csv_path:
        raise ValueError("robustness_eval.yaml must set test_csv.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_type = config.get("model_type", "resnet")
    if model_type == "resnet":
        model = build_resnet_detector(config.get("model_name", "resnet50"), pretrained=False).to(device)
        state = torch.load(checkpoint, map_location=device)
        model.load_state_dict(state["model_state"])
        model.eval()

        def run_predict(transform_name):
            return predict_resnet(config, transform_name, model, csv_path, device)

        model_name = config.get("model_name", "resnet50")
    elif model_type == "clip_mlp":
        clip_model, preprocess = load_open_clip_model(config.get("clip_model", "ViT-B-32"), device)
        state = torch.load(checkpoint, map_location=device)
        feature_dim = state["model_state"]["classifier.0.weight"].shape[1]
        model = ClipMlpDetector(feature_dim, config.get("mlp_hidden_dim", 512), config.get("dropout", 0.2)).to(device)
        model.load_state_dict(state["model_state"])
        model.eval()

        def run_predict(transform_name):
            return predict_clip_mlp(config, transform_name, clip_model, model, preprocess, csv_path, device)

        model_name = config.get("clip_model", "ViT-B-32")
    else:
        raise ValueError(f"Unsupported model_type for robustness: {model_type}")

    clean = run_predict("clean")
    rows = []
    for name in config.get("transforms", ["clean", "jpeg", "resize", "blur"]):
        current = clean if name == "clean" else run_predict(name)
        rows.append(
            {
                "experiment_name": config["experiment_name"],
                "model": model_name,
                "test_set": csv_path,
                "transform": name,
                "clean_auroc": clean["auroc"],
                "processed_auroc": current["auroc"],
                "robustness_drop": clean["auroc"] - current["auroc"],
            }
        )
    df = pd.DataFrame(rows)
    Path("outputs/metrics").mkdir(parents=True, exist_ok=True)
    out_csv = Path("outputs/metrics") / f"{config['experiment_name']}_robustness.csv"
    df.to_csv(out_csv, index=False)
    plot_bar(df, Path("outputs/figures") / config["experiment_name"] / "robustness_bar_chart.png")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
