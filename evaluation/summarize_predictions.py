"""Recompute full, balanced, and per-generator metrics from saved predictions."""

import argparse
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.metrics import binary_metrics
from evaluation.plots import plot_confusion_matrix


METRIC_NAMES = (
    "accuracy", "balanced_accuracy", "precision", "fake_recall", "real_recall",
    "binary_f1", "macro_f1", "auroc", "auprc", "recall", "f1",
)


def metric_row(frame, experiment_name, scope, threshold=0.5):
    metrics = binary_metrics(frame["label"], frame["fake_prob"], threshold=threshold)
    row = {
        "experiment_name": experiment_name,
        "scope": scope,
        "num_samples": len(frame),
        "num_real": int((frame["label"] == 0).sum()),
        "num_fake": int((frame["label"] == 1).sum()),
        "threshold": threshold,
    }
    row.update({name: metrics[name] for name in METRIC_NAMES})
    row["confusion_matrix"] = metrics["confusion_matrix"].tolist()
    return row, metrics["confusion_matrix"]


def generator_rows(frame, experiment_name, scope, threshold=0.5):
    if "label_b" not in frame.columns:
        raise ValueError("Predictions must contain label_b for generator analysis.")
    real = frame[frame["label_b"] == 0]
    rows = []
    for label_b in range(1, 6):
        fake = frame[frame["label_b"] == label_b]
        if real.empty or fake.empty:
            raise ValueError(f"Missing real rows or Label_B={label_b} rows for generator analysis.")
        subset = pd.concat([real, fake], ignore_index=True)
        row, _ = metric_row(subset, experiment_name, scope, threshold=threshold)
        row.update({
            "label_b": label_b,
            "generator": str(fake["generator"].iloc[0]),
            "mean_fake_probability": float(fake["fake_prob"].mean()),
            "false_negatives": int((fake["fake_prob"] < threshold).sum()),
            "false_negative_rate": float((fake["fake_prob"] < threshold).mean()),
        })
        rows.append(row)
    return pd.DataFrame(rows)


def select_manifest(predictions, manifest):
    sample_ids = pd.read_csv(manifest, usecols=["sample_id"])
    if sample_ids["sample_id"].duplicated().any():
        raise ValueError("Balanced manifest contains duplicate sample_id values.")
    selected = predictions[predictions["sample_id"].isin(set(sample_ids["sample_id"]))].copy()
    if len(selected) != len(sample_ids):
        missing = set(sample_ids["sample_id"]) - set(selected["sample_id"])
        raise ValueError(f"Predictions are missing {len(missing)} balanced sample IDs.")
    return selected


def assert_matches(metric_row_value, original_metrics_path, tolerance=1e-10):
    original = pd.read_csv(original_metrics_path).iloc[0]
    for name in METRIC_NAMES:
        if name not in original:
            continue
        if abs(float(metric_row_value[name]) - float(original[name])) > tolerance:
            raise ValueError(f"Recomputed {name} does not match original metrics.")


def summarize(predictions_path, original_metrics_path, balanced_manifest, experiment_name, out_dir, figure_dir, threshold=0.5):
    predictions = pd.read_csv(predictions_path)
    required = {"sample_id", "label", "pred_label", "fake_prob", "label_b", "generator"}
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"Predictions are missing required columns: {sorted(missing)}")
    if predictions["sample_id"].duplicated().any():
        raise ValueError("Full predictions contain duplicate sample_id values.")
    full_row, full_cm = metric_row(predictions, experiment_name, "official_full", threshold=threshold)
    assert_matches(full_row, original_metrics_path)
    balanced = select_manifest(predictions, balanced_manifest)
    balanced_row, balanced_cm = metric_row(
        balanced, experiment_name, "balanced_from_full_predictions", threshold=threshold
    )
    out_dir = Path(out_dir)
    figure_dir = Path(figure_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([full_row]).to_csv(out_dir / f"{experiment_name}_full_recomputed.csv", index=False)
    pd.DataFrame([balanced_row]).to_csv(out_dir / f"{experiment_name}_balanced_metrics.csv", index=False)
    generator_rows(predictions, experiment_name, "official_full", threshold=threshold).to_csv(
        out_dir / f"{experiment_name}_full_by_generator.csv", index=False
    )
    generator_rows(balanced, experiment_name, "balanced_from_full_predictions", threshold=threshold).to_csv(
        out_dir / f"{experiment_name}_balanced_by_generator.csv", index=False
    )
    balanced.to_csv(Path(predictions_path).with_name(f"{experiment_name}_balanced_predictions.csv"), index=False)
    plot_confusion_matrix(full_cm, figure_dir / "full_recomputed_confusion_matrix.png")
    plot_confusion_matrix(balanced_cm, figure_dir / "balanced_confusion_matrix.png")
    return full_row, balanced_row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--original-metrics", required=True)
    parser.add_argument("--balanced-manifest", required=True)
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--out-dir", default="outputs/metrics")
    parser.add_argument("--figure-dir", required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()
    full, balanced = summarize(
        args.predictions, args.original_metrics, args.balanced_manifest,
        args.experiment_name, args.out_dir, args.figure_dir, args.threshold,
    )
    print(pd.DataFrame([full, balanced]).to_string(index=False))


if __name__ == "__main__":
    main()
