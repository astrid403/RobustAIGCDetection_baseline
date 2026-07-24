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

from evaluation.metrics import binary_metrics
from evaluation.research_inference import DEGRADATIONS, DegradedImageDataset
from models.clip_mlp_detector import (
    build_clip_feature_detector,
    encode_clip_image_features,
    encode_clip_multiblock_cls,
    load_open_clip_model,
)
from models.rine_lite_detector import build_rine_lite_detector


FOLDS = ("adm", "biggan", "stable_diffusion_v15")
SEEDS = (42, 43, 44)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_path(model, seed, fold):
    if model == "b2":
        run = (
            f"pilot_b2_holdout_{fold}_seed42_v3" if seed == 42
            else f"confirm_b2_holdout_{fold}_seed{seed}_v3"
        )
    else:
        run = (
            f"pilot_s2_holdout_{fold}_seed42_amp_retry1_v3" if seed == 42
            else f"confirm_s2_holdout_{fold}_seed{seed}_amp_v3"
        )
    return Path("outputs/research_v3/checkpoints") / run / "best_model.pt"


def mean_probabilities(member_probabilities):
    stacked = torch.stack(member_probabilities)
    if stacked.shape[0] != 3:
        raise ValueError("Frozen ensemble requires exactly three fold members")
    return stacked.double().mean(dim=0)


def load_heads(model_name, device):
    heads, records = {}, {}
    for seed in SEEDS:
        heads[seed], records[seed] = [], []
        for fold in FOLDS:
            path = checkpoint_path(model_name, seed, fold)
            head = (
                build_clip_feature_detector("clip_mlp", 512, 512, 0.2)
                if model_name == "b2" else build_rine_lite_detector()
            ).to(device)
            payload = torch.load(path, map_location=device)
            head.load_state_dict(payload["model_state"])
            head.eval()
            heads[seed].append(head)
            records[seed].append({"fold": fold, "path": str(path),
                                  "sha256": sha256_file(path)})
    return heads, records


@torch.no_grad()
def evaluate_model(model_name, encoder, preprocess, manifest, output, device):
    heads, checkpoint_records = load_heads(model_name, device)
    result_records, metric_rows = {}, []
    for condition in DEGRADATIONS:
        loader = DataLoader(
            DegradedImageDataset(manifest, condition, transform=preprocess),
            batch_size=32, shuffle=False, num_workers=0,
        )
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
                    member = [
                        torch.sigmoid(
                            head(features).view(-1) if model_name == "b2"
                            else head(features).view(-1)
                        ).cpu()
                        for head in heads[seed]
                    ]
                    probs = mean_probabilities(member).tolist()
                    rows[seed].extend(zip(sample_ids, labels.tolist(), probs))
        for seed in SEEDS:
            frame = pd.DataFrame(rows[seed], columns=["sample_id", "label", "probability"])
            if frame.sample_id.duplicated().any():
                raise ValueError("Duplicate unseen sample ID")
            path = output / f"{model_name}_seed{seed}_{condition}.csv"
            if path.exists():
                raise FileExistsError(path)
            frame.to_csv(path, index=False)
            metrics = binary_metrics(frame.label, frame.probability)
            metric_rows.append({"model": model_name, "seed": seed,
                                "condition": condition,
                                **{k: v for k, v in metrics.items()
                                   if k != "confusion_matrix"}})
            result_records[f"{model_name}_seed{seed}_{condition}"] = {
                "path": str(path), "sha256": sha256_file(path)
            }
    return result_records, metric_rows, checkpoint_records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError(f"Refusing overwrite: {output}")
    output.mkdir(parents=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder, preprocess = load_open_clip_model("ViT-B-32", device, "openai")
    all_records, all_metrics, checkpoints = {}, [], {}
    for model in ("b2", "s2"):
        records, metrics, ckpts = evaluate_model(
            model, encoder, preprocess, args.manifest, output, device
        )
        all_records.update(records); all_metrics.extend(metrics); checkpoints[model] = ckpts
    metrics_path = output / "metrics.csv"
    pd.DataFrame(all_metrics).to_csv(metrics_path, index=False)
    registry = {
        "schema": "genimage_unseen_three_fold_ensemble_v3",
        "manifest": {"path": args.manifest, "sha256": sha256_file(args.manifest)},
        "fold_order": list(FOLDS), "fold_weights": [1/3, 1/3, 1/3],
        "threshold": 0.5, "checkpoints": checkpoints, "predictions": all_records,
        "metrics": {"path": str(metrics_path), "sha256": sha256_file(metrics_path)},
        "defactify_accessed": False,
    }
    (output / "registry.json").write_text(json.dumps(registry, indent=2, sort_keys=True)+"\n")


if __name__ == "__main__":
    main()
