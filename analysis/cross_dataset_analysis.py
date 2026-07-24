"""Metrics and descriptive analysis for Protocol-v3 OOF predictions."""

from __future__ import annotations

import numpy as np
import pandas as pd

from evaluation.metrics import binary_metrics
from models.late_fusion import DESCRIPTIVE_ALPHA_GRID, fuse_aligned_probabilities


METRIC_KEYS = (
    "auroc",
    "auprc",
    "balanced_accuracy",
    "macro_f1",
    "real_recall",
    "fake_recall",
)


def _metric_record(labels, probabilities, threshold=0.5):
    metrics = binary_metrics(labels, probabilities, threshold=threshold)
    return {key: float(metrics[key]) for key in METRIC_KEYS}


def error_complementarity(aligned: pd.DataFrame, threshold: float = 0.5) -> dict:
    labels = aligned["label"].to_numpy(dtype=int)
    clip_error = (aligned["clip_probability"].to_numpy() >= threshold).astype(int) != labels
    npr_error = (aligned["npr_probability"].to_numpy() >= threshold).astype(int) != labels
    both_error = clip_error & npr_error
    either_error = clip_error | npr_error
    count = len(labels)
    return {
        "sample_count": count,
        "clip_error_count": int(clip_error.sum()),
        "npr_error_count": int(npr_error.sum()),
        "both_error_count": int(both_error.sum()),
        "either_error_count": int(either_error.sum()),
        "clip_wrong_npr_right_count": int((clip_error & ~npr_error).sum()),
        "npr_wrong_clip_right_count": int((npr_error & ~clip_error).sum()),
        "both_correct_count": int((~clip_error & ~npr_error).sum()),
        "prediction_disagreement_count": int(
            (
                (aligned["clip_probability"].to_numpy() >= threshold)
                != (aligned["npr_probability"].to_numpy() >= threshold)
            ).sum()
        ),
        "error_jaccard": (
            float(both_error.sum() / either_error.sum()) if either_error.any() else 0.0
        ),
        "complementary_error_rate": float((clip_error ^ npr_error).sum() / count),
    }


def summarize_oof(fused: pd.DataFrame, threshold: float = 0.5) -> dict:
    folds = {}
    for fold, group in fused.groupby("fold", sort=True):
        folds[str(fold)] = {
            "sample_count": int(len(group)),
            **_metric_record(group["label"], group["fused_probability"], threshold),
        }
    if not folds:
        raise ValueError("OOF predictions must contain at least one fold")
    aggregate = {}
    for key in METRIC_KEYS:
        values = [record[key] for record in folds.values()]
        aggregate[f"fold_mean_{key}"] = float(np.mean(values))
        aggregate[f"fold_worst_{key}"] = float(np.min(values))
    return {
        "overall": _metric_record(
            fused["label"], fused["fused_probability"], threshold
        ),
        "folds": folds,
        "fold_aggregate": aggregate,
        "complementarity": error_complementarity(fused, threshold),
    }


def descriptive_alpha_grid(aligned: pd.DataFrame, threshold: float = 0.5) -> list[dict]:
    """Evaluate the frozen grid; results are descriptive and never select primary alpha."""
    rows = []
    for alpha in DESCRIPTIVE_ALPHA_GRID:
        fused = fuse_aligned_probabilities(aligned, alpha)
        summary = summarize_oof(fused, threshold)
        rows.append(
            {
                "alpha_clip": alpha,
                "role": "descriptive_ablation_only",
                **summary["fold_aggregate"],
            }
        )
    return rows


def rank_descriptive_alphas(rows: list[dict]) -> list[dict]:
    """Stable descriptive ordering: AUROC desc, closest to 0.5, then lower alpha.

    This ordering cannot replace the Protocol-v3 primary alpha of 0.5.
    """
    return sorted(
        rows,
        key=lambda row: (
            -row["fold_mean_auroc"],
            abs(row["alpha_clip"] - 0.5),
            row["alpha_clip"],
        ),
    )
