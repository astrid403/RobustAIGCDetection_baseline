"""Build and audit the compact Part 10 report artifacts from frozen summaries."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REQUIRED_METRICS = (
    "accuracy", "balanced_accuracy", "precision", "real_recall", "fake_recall",
    "binary_f1", "macro_f1", "auroc", "auprc",
)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_registry(path):
    registry = json.loads(Path(path).read_text())
    for record in registry.get("outputs", {}).values():
        if sha256(record["path"]) != record["sha256"]:
            raise ValueError(f"Registry hash mismatch: {record['path']}")
    return registry


def build_aggregate(mean_sd_path, delta_path):
    mean_sd = pd.read_csv(mean_sd_path)
    delta = pd.read_csv(delta_path)
    if len(mean_sd) != 54:
        raise ValueError("Expected 2 roles x 3 scopes x 9 metrics.")
    if set(mean_sd.role) != {"B2", "A1"}:
        raise ValueError("Final aggregate requires exactly B2 and A1.")
    if set(mean_sd.metric) != set(REQUIRED_METRICS):
        raise ValueError("Final aggregate metric set is incomplete.")
    wide = mean_sd.pivot(index=["scope", "metric"], columns="role", values=["mean", "sample_sd"])
    rows = []
    for scope in ("genimage_unseen", "defactify_full", "defactify_balanced"):
        for metric in REQUIRED_METRICS:
            expected_delta = float(
                delta[(delta.scope == scope) & (delta.metric == metric)].a1_minus_b2.iloc[0]
            )
            b2_mean = float(wide.loc[(scope, metric), ("mean", "B2")])
            a1_mean = float(wide.loc[(scope, metric), ("mean", "A1")])
            if not np.isclose(a1_mean - b2_mean, expected_delta, atol=1e-14):
                raise ValueError(f"Delta mismatch for {scope}/{metric}.")
            rows.append({
                "scope": scope,
                "metric": metric,
                "b2_mean": b2_mean,
                "b2_sample_sd": float(wide.loc[(scope, metric), ("sample_sd", "B2")]),
                "a1_mean": a1_mean,
                "a1_sample_sd": float(wide.loc[(scope, metric), ("sample_sd", "A1")]),
                "a1_minus_b2": expected_delta,
                "seeds": "42,43,44",
                "threshold": 0.5,
            })
    return pd.DataFrame(rows)


def main_results(aggregate, bootstrap):
    rows = []
    display_metrics = ("auroc", "auprc", "balanced_accuracy", "macro_f1")
    for scope in ("genimage_unseen", "defactify_full", "defactify_balanced"):
        for role in ("B2", "A1"):
            row = {"scope": scope, "model": role, "seeds": "42,43,44"}
            for metric in display_metrics:
                source = aggregate[(aggregate.scope == scope) & (aggregate.metric == metric)].iloc[0]
                row[f"{metric}_mean"] = source[f"{role.lower()}_mean"]
                row[f"{metric}_sample_sd"] = source[f"{role.lower()}_sample_sd"]
            rows.append(row)
    result = pd.DataFrame(rows)
    if len(bootstrap) != 9:
        raise ValueError("Bootstrap table must contain 3 scopes x 3 metrics.")
    return result


def core_ablation(path):
    source = pd.read_csv(path)
    selected = source[source.scope == "unseen"].copy()
    order = ["B1_linear", "B2_final_mlp", "A1_penultimate_mlp", "P_dual_level_fusion"]
    selected = selected.set_index("model").loc[order].reset_index()
    return selected[[
        "model", "feature_modes", "seed", "trainable_parameters", "cache_bytes",
        "balanced_accuracy", "macro_f1", "auroc", "auprc",
    ]]


def plot_generalization(aggregate, robustness, output):
    clean = aggregate[aggregate.metric == "auroc"].set_index("scope")
    labels = ["GenImage unseen", "Defactify full", "Defactify balanced", "JPEG 70", "Resize 0.5", "Blur 1.0"]
    deltas = [
        clean.loc["genimage_unseen", "a1_minus_b2"],
        clean.loc["defactify_full", "a1_minus_b2"],
        clean.loc["defactify_balanced", "a1_minus_b2"],
    ]
    for transform in ("jpeg", "resize", "blur"):
        values = robustness[robustness["transform"] == transform].set_index("role")
        deltas.append(values.loc["A1", "auroc_mean"] - values.loc["B2", "auroc_mean"])
    colors = ["#2b8cbe" if value >= 0 else "#d95f0e" for value in deltas]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(labels, deltas, color=colors)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylim(min(deltas) - 0.003, max(deltas) + 0.002)
    ax.set_ylabel("AUROC difference (A1 - B2)")
    ax.set_title("Penultimate features: internal robustness gain, external trade-off")
    ax.tick_params(axis="x", rotation=20)
    for index, value in enumerate(deltas):
        ax.text(index, value + (0.0005 if value >= 0 else -0.0008), f"{value:+.4f}",
                ha="center", va="bottom" if value >= 0 else "top", fontsize=8)
    fig.tight_layout()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="reports/final_assets")
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    verify_registry("outputs/metrics/part09_analysis_registry.json")
    part08 = json.loads(Path("artifacts/part08_formal_evaluation_registry.json").read_text())
    if len(part08["runs"]) != 6:
        raise ValueError("Part 08 registry must contain six model/seed pairs.")

    aggregate = build_aggregate(
        "outputs/metrics/part09_mean_sd.csv",
        "outputs/metrics/part09_a1_minus_b2.csv",
    )
    bootstrap = pd.read_csv("outputs/metrics/part09_paired_bootstrap_ci.csv")
    results = main_results(aggregate, bootstrap)
    ablation = core_ablation("outputs/metrics/part06_seed42_validation_unseen_comparison.csv")
    robustness = pd.read_csv("outputs/metrics/part09_robustness_summary.csv")

    aggregate.to_csv("outputs/metrics/final_aggregate_results.csv", index=False)
    results.to_csv(output / "main_results_table.csv", index=False)
    ablation.to_csv(output / "core_ablation_table.csv", index=False)
    plot_generalization(
        aggregate, robustness, output / "generalization_robustness.png"
    )
    print("Part 10 aggregates and report artifacts generated and cross-checked.")


if __name__ == "__main__":
    main()
