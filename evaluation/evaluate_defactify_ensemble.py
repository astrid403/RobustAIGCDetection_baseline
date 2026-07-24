import argparse
import hashlib
import json
from pathlib import Path
import sys

import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_pipeline.dataset_factory import create_image_dataset
from evaluation.evaluate_genimage_unseen_ensemble import (
    SEEDS, load_heads, mean_probabilities, sha256_file,
)
from evaluation.metrics import binary_metrics
from models.clip_mlp_detector import (
    encode_clip_image_features, encode_clip_multiblock_cls, load_open_clip_model,
)


def derive_balanced(full, balanced_manifest):
    ids = pd.read_csv(balanced_manifest, usecols=["sample_id"])["sample_id"].tolist()
    if len(ids) != len(set(ids)):
        raise ValueError("Balanced manifest contains duplicate sample IDs")
    if full["sample_id"].duplicated().any():
        raise ValueError("Full predictions contain duplicate sample IDs")
    indexed = full.set_index("sample_id", drop=False)
    missing = set(ids) - set(indexed.index)
    if missing:
        raise ValueError(f"Full predictions miss {len(missing)} balanced IDs")
    return indexed.loc[ids].reset_index(drop=True)


def metric_record(model, seed, scope, frame):
    metrics = binary_metrics(frame["label"], frame["fake_prob"])
    return {"model": model, "seed": seed, "scope": scope, "samples": len(frame),
            **{key: value for key, value in metrics.items()
               if key != "confusion_matrix"}}


def generator_records(model, seed, scope, frame):
    real = frame[frame["label_b"] == 0]
    rows = []
    for label_b in range(1, 6):
        fake = frame[frame["label_b"] == label_b]
        if real.empty or fake.empty:
            raise ValueError(f"Missing Defactify group label_b={label_b}")
        subset = pd.concat([real, fake], ignore_index=True)
        row = metric_record(model, seed, scope, subset)
        row.update({"label_b": label_b, "generator": str(fake.generator.iloc[0])})
        rows.append(row)
    return rows


@torch.no_grad()
def evaluate(model_name, encoder, preprocess, full_manifest, balanced_manifest,
             output, device):
    heads, checkpoints = load_heads(model_name, device)
    dataset = create_image_dataset(full_manifest, transform=preprocess)
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)
    rows = {seed: [] for seed in SEEDS}
    for images, labels, sample_ids in loader:
        images = images.to(device)
        amp = model_name == "s2"
        with torch.autocast("cuda", dtype=torch.float16,
                            enabled=amp and device.type == "cuda"):
            features = (
                encode_clip_image_features(encoder, images, "final")
                if model_name == "b2" else encode_clip_multiblock_cls(encoder, images)
            )
            for seed in SEEDS:
                member = [torch.sigmoid(head(features).view(-1)).cpu()
                          for head in heads[seed]]
                probs = mean_probabilities(member).tolist()
                rows[seed].extend(zip(sample_ids, labels.tolist(), probs))
    metadata = pd.read_csv(full_manifest).set_index("sample_id")
    predictions, metric_rows, generator_rows = {}, [], []
    for seed in SEEDS:
        full = pd.DataFrame(rows[seed],
                            columns=["sample_id", "label", "fake_prob"])
        full["pred_label"] = (full.fake_prob >= 0.5).astype(int)
        full = full.join(metadata[["label_b", "generator"]], on="sample_id")
        if len(full) != 45000 or full.sample_id.duplicated().any():
            raise ValueError("Defactify full prediction integrity failure")
        balanced = derive_balanced(full, balanced_manifest)
        if len(balanced) != 15000:
            raise ValueError("Defactify balanced row count mismatch")
        for scope, frame in (("full", full), ("balanced", balanced)):
            path = output / f"{model_name}_seed{seed}_{scope}.csv"
            if path.exists():
                raise FileExistsError(path)
            frame.to_csv(path, index=False)
            predictions[f"{model_name}_seed{seed}_{scope}"] = {
                "path": str(path), "sha256": sha256_file(path)
            }
            metric_rows.append(metric_record(model_name, seed, scope, frame))
            generator_rows.extend(
                generator_records(model_name, seed, scope, frame)
            )
    return predictions, metric_rows, generator_rows, checkpoints


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full-manifest", required=True)
    parser.add_argument("--balanced-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError(f"Refusing overwrite: {output}")
    output.mkdir(parents=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder, preprocess = load_open_clip_model("ViT-B-32", device, "openai")
    predictions, metrics, generators, checkpoints = {}, [], [], {}
    for model in ("b2", "s2"):
        p, m, g, c = evaluate(
            model, encoder, preprocess, args.full_manifest,
            args.balanced_manifest, output, device
        )
        predictions.update(p); metrics.extend(m); generators.extend(g)
        checkpoints[model] = c
    metrics_path = output / "metrics.csv"
    generators_path = output / "per_generator.csv"
    pd.DataFrame(metrics).to_csv(metrics_path, index=False)
    pd.DataFrame(generators).to_csv(generators_path, index=False)
    registry = {
        "schema": "defactify_three_fold_ensemble_v3",
        "full_manifest": {"path": args.full_manifest,
                          "sha256": sha256_file(args.full_manifest)},
        "balanced_manifest": {"path": args.balanced_manifest,
                              "sha256": sha256_file(args.balanced_manifest)},
        "inference_scope": "full_only",
        "balanced_derivation": "sample_id_subset_from_full_no_second_inference",
        "threshold": 0.5, "fold_weights": [1/3, 1/3, 1/3],
        "checkpoints": checkpoints, "predictions": predictions,
        "metrics": {"path": str(metrics_path), "sha256": sha256_file(metrics_path)},
        "per_generator": {"path": str(generators_path),
                          "sha256": sha256_file(generators_path)},
    }
    (output / "registry.json").write_text(
        json.dumps(registry, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
