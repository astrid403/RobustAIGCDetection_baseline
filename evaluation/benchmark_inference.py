"""Calibrate safe inference batch sizes and throughput on a deterministic manifest."""

import argparse
import json
from pathlib import Path
import sys
import time

import torch
from torch.utils.data import DataLoader
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_pipeline.dataset_factory import create_image_dataset
from models.clip_mlp_detector import ClipMlpDetector, load_open_clip_model
from models.resnet_detector import build_resnet_detector
from training.train import resnet_transforms


def synchronize():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def benchmark(config, checkpoint, csv_path, batch_size):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for performance calibration.")
    device = torch.device("cuda")
    model_type = config["model_type"]
    if model_type == "resnet":
        transform = resnet_transforms(config.get("image_size", 224), train=False)
        model = build_resnet_detector(config.get("model_name", "resnet50"), pretrained=False).to(device)
        state = torch.load(checkpoint, map_location=device)
        model.load_state_dict(state["model_state"])
        encode = lambda images: model(images).view(-1)
    else:
        clip_model, transform = load_open_clip_model(config.get("clip_model", "ViT-B-32"), device)
        state = torch.load(checkpoint, map_location=device)
        feature_dim = state["model_state"]["classifier.0.weight"].shape[1]
        model = ClipMlpDetector(feature_dim, config.get("mlp_hidden_dim", 512), config.get("dropout", 0.2)).to(device)
        model.load_state_dict(state["model_state"])
        encode = lambda images: model(clip_model.encode_image(images)).view(-1)
    model.eval()
    dataset = create_image_dataset(csv_path, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=config.get("num_workers", 2), pin_memory=True)
    load_start = time.perf_counter()
    first = next(iter(loader))
    data_first_batch_seconds = time.perf_counter() - load_start
    del first
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    count = 0
    start = time.perf_counter()
    with torch.no_grad():
        for images, _, _ in loader:
            images = images.to(device, non_blocking=True)
            encode(images)
            count += images.size(0)
    synchronize()
    elapsed = time.perf_counter() - start
    return {
        "model_type": model_type,
        "batch_size": batch_size,
        "num_samples": count,
        "elapsed_seconds": elapsed,
        "images_per_second": count / elapsed,
        "first_batch_load_seconds": data_first_batch_seconds,
        "peak_gpu_memory_mib": torch.cuda.max_memory_allocated() / 1024**2,
        "estimated_full_test_seconds": 45000 / (count / elapsed),
        "estimated_balanced_one_pass_seconds": 15000 / (count / elapsed),
        "estimated_balanced_robustness_seconds": 4 * 15000 / (count / elapsed),
        "device": torch.cuda.get_device_name(0),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--batch-sizes", type=int, nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    rows = []
    for batch_size in args.batch_sizes:
        try:
            row = benchmark(config, args.checkpoint, args.csv, batch_size)
            row["status"] = "ok"
            rows.append(row)
            print(json.dumps(row, ensure_ascii=False))
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            rows.append({"model_type": config["model_type"], "batch_size": batch_size, "status": "oom"})
            print(json.dumps(rows[-1]))
            break
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
