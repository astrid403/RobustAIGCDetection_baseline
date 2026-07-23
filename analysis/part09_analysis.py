"""Aggregate Part 09 statistics, paired bootstrap, efficiency, and errors."""

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.metrics import binary_metrics


METRICS = (
    "accuracy", "balanced_accuracy", "precision", "real_recall", "fake_recall",
    "binary_f1", "macro_f1", "auroc", "auprc",
)
BOOTSTRAP_METRICS = ("auroc", "auprc", "balanced_accuracy")


def prediction_path(role, seed, scope):
    role = role.lower()
    if scope == "genimage_unseen":
        return Path(f"outputs/predictions/final_{role}_genimage_unseen_seed{seed}_v2_predictions.csv")
    suffix = "_balanced" if scope == "defactify_balanced" else ""
    return Path(f"outputs/predictions/final_{role}_defactify_full_seed{seed}_v2{suffix}_predictions.csv")


def aligned_pair(seed, scope):
    b2 = pd.read_csv(prediction_path("B2", seed, scope)).sort_values("sample_id").reset_index(drop=True)
    a1 = pd.read_csv(prediction_path("A1", seed, scope)).sort_values("sample_id").reset_index(drop=True)
    if not b2.sample_id.equals(a1.sample_id) or not b2.label.equals(a1.label):
        raise ValueError(f"Unaligned B2/A1 predictions for {scope}, seed {seed}.")
    return b2, a1


def hierarchical_bootstrap(scope, repeats=1000, random_seed=20260723):
    pairs = [aligned_pair(seed, scope) for seed in (42, 43, 44)]
    labels = pairs[0][0].label.to_numpy()
    if not all(np.array_equal(labels, pair[0].label.to_numpy()) for pair in pairs):
        raise ValueError(f"Labels differ across seeds for {scope}.")
    b2_probs = np.stack([pair[0].fake_prob.to_numpy() for pair in pairs])
    a1_probs = np.stack([pair[1].fake_prob.to_numpy() for pair in pairs])
    rng = np.random.default_rng(random_seed)
    draws = {metric: [] for metric in BOOTSTRAP_METRICS}
    for _ in range(repeats):
        seed_idx = rng.integers(0, 3, 3)
        sample_idx = rng.integers(0, len(labels), len(labels))
        y = labels[sample_idx]
        for metric in BOOTSTRAP_METRICS:
            differences = []
            for idx in seed_idx:
                b2 = binary_metrics(y, b2_probs[idx, sample_idx], threshold=0.5)[metric]
                a1 = binary_metrics(y, a1_probs[idx, sample_idx], threshold=0.5)[metric]
                differences.append(a1 - b2)
            draws[metric].append(float(np.mean(differences)))
    rows = []
    for metric, values in draws.items():
        values = np.asarray(values)
        rows.append({
            "scope": scope,
            "metric": metric,
            "a1_minus_b2_mean": float(values.mean()),
            "ci95_low": float(np.quantile(values, 0.025)),
            "ci95_high": float(np.quantile(values, 0.975)),
            "supported_positive": bool(np.quantile(values, 0.025) > 0),
            "supported_negative": bool(np.quantile(values, 0.975) < 0),
            "bootstrap_repeats": repeats,
            "bootstrap_seed": random_seed,
            "method": "paired hierarchical bootstrap over seeds and sample_ids",
        })
    return rows


def aggregate_metrics(summary):
    rows = []
    for (role, scope), group in summary.groupby(["role", "scope"], sort=False):
        for metric in METRICS:
            values = group[metric].astype(float).tolist()
            rows.append({
                "role": role, "scope": scope, "metric": metric,
                "mean": statistics.mean(values),
                "sample_sd": statistics.stdev(values),
            })
    result = pd.DataFrame(rows)
    pivot = result.pivot(index=["scope", "metric"], columns="role", values="mean").reset_index()
    pivot["a1_minus_b2"] = pivot["A1"] - pivot["B2"]
    return result, pivot


def error_analysis():
    count_rows, manifest_rows = [], []
    for scope in ("genimage_unseen", "defactify_full", "defactify_balanced"):
        for seed in (42, 43, 44):
            b2, a1 = aligned_pair(seed, scope)
            categories = {
                "both_wrong": (~b2.is_correct.astype(bool)) & (~a1.is_correct.astype(bool)),
                "b2_wrong_a1_correct": (~b2.is_correct.astype(bool)) & a1.is_correct.astype(bool),
                "b2_correct_a1_wrong": b2.is_correct.astype(bool) & (~a1.is_correct.astype(bool)),
                "both_correct": b2.is_correct.astype(bool) & a1.is_correct.astype(bool),
            }
            for category, mask in categories.items():
                selected = b2[mask]
                count_rows.append({
                    "scope": scope, "seed": seed, "category": category,
                    "count": len(selected),
                    "real_count": int((selected.label == 0).sum()),
                    "fake_count": int((selected.label == 1).sum()),
                })
                sample = selected.sort_values("sample_id").head(5)
                for _, row in sample.iterrows():
                    idx = b2.index[b2.sample_id == row.sample_id][0]
                    manifest_rows.append({
                        "scope": scope, "seed": seed, "category": category,
                        "sample_id": row.sample_id, "path": row.path,
                        "label": int(row.label), "generator": row.generator,
                        "b2_probability": float(b2.loc[idx, "fake_prob"]),
                        "a1_probability": float(a1.loc[idx, "fake_prob"]),
                        "b2_prediction": int(b2.loc[idx, "pred_label"]),
                        "a1_prediction": int(a1.loc[idx, "pred_label"]),
                    })
            for role, frame in (("B2", b2), ("A1", a1)):
                for error_type, mask in (
                    ("false_positive", (frame.label == 0) & (frame.pred_label == 1)),
                    ("false_negative", (frame.label == 1) & (frame.pred_label == 0)),
                ):
                    count_rows.append({
                        "scope": scope, "seed": seed,
                        "category": f"{role.lower()}_{error_type}",
                        "count": int(mask.sum()),
                        "real_count": int(((frame.label == 0) & mask).sum()),
                        "fake_count": int(((frame.label == 1) & mask).sum()),
                    })
    return pd.DataFrame(count_rows), pd.DataFrame(manifest_rows)


def efficiency():
    registry = json.loads(Path("artifacts/part07_formal_training_registry.json").read_text())
    rows = []
    for run in registry["runs"]:
        role, seed = run["role"], run["seed"]
        checkpoint = Path(run["best_checkpoint_path"])
        feature_dir = Path(
            f"outputs/features/final_clip_{'mlp' if role == 'B2' else 'penultimate'}_genimage_validation_seed{seed}_v2"
        )
        cache_bytes = sum(path.stat().st_size for path in feature_dir.glob("*.pt"))
        rows.append({
            "role": role, "seed": seed,
            "total_parameters": 151540482,
            "trainable_parameters": 263169,
            "checkpoint_bytes": checkpoint.stat().st_size,
            "training_cache_bytes": cache_bytes,
            "training_wall_seconds": run["wall_time_seconds"],
        })
    return pd.DataFrame(rows)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", required=True)
    parser.add_argument("--robustness", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bootstrap-repeats", type=int, default=1000)
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(args.summary)
    mean_sd, delta = aggregate_metrics(summary)
    bootstrap = pd.DataFrame(sum(
        [hierarchical_bootstrap(scope, args.bootstrap_repeats) for scope in
         ("genimage_unseen", "defactify_full", "defactify_balanced")], []
    ))
    error_counts, error_manifest = error_analysis()
    efficiency_table = efficiency()
    robustness = pd.read_csv(args.robustness)
    robust_summary = robustness.groupby(["role", "transform"], sort=False).agg(
        auroc_mean=("auroc", "mean"),
        auroc_sample_sd=("auroc", "std"),
        auprc_mean=("auprc", "mean"),
        balanced_accuracy_mean=("balanced_accuracy", "mean"),
        auroc_delta_mean=("auroc_delta_from_clean", "mean"),
    ).reset_index()

    files = {
        "mean_sd": output / "part09_mean_sd.csv",
        "delta": output / "part09_a1_minus_b2.csv",
        "bootstrap": output / "part09_paired_bootstrap_ci.csv",
        "robustness": output / "part09_robustness_summary.csv",
        "efficiency": output / "part09_efficiency.csv",
        "error_counts": output / "part09_error_counts.csv",
        "error_manifest": output / "part09_error_case_manifest.csv",
    }
    mean_sd.to_csv(files["mean_sd"], index=False)
    delta.to_csv(files["delta"], index=False)
    bootstrap.to_csv(files["bootstrap"], index=False)
    robust_summary.to_csv(files["robustness"], index=False)
    efficiency_table.to_csv(files["efficiency"], index=False)
    error_counts.to_csv(files["error_counts"], index=False)
    error_manifest.to_csv(files["error_manifest"], index=False)
    registry = {
        "part": "Part 09",
        "status": "complete",
        "threshold": 0.5,
        "seeds": [42, 43, 44],
        "bootstrap": {
            "repeats": args.bootstrap_repeats,
            "seed": 20260723,
            "method": "paired hierarchical bootstrap over seeds and sample_ids",
        },
        "robustness": {
            "scope": "GenImage unseen",
            "transforms": {"jpeg_quality": 70, "resize_scale": 0.5, "blur_radius": 1.0},
        },
        "outputs": {name: {"path": str(path), "sha256": sha256(path)} for name, path in files.items()},
        "image_copies_committed": False,
    }
    (output / "part09_analysis_registry.json").write_text(
        json.dumps(registry, indent=2) + "\n"
    )
    print(bootstrap.to_string(index=False))
    print(robust_summary.to_string(index=False))


if __name__ == "__main__":
    main()
