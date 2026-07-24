"""Protocol-v3 OOF probability fusion with strict sample alignment."""

from __future__ import annotations

import numpy as np
import pandas as pd


PRIMARY_ALPHA_CLIP = 0.5
DESCRIPTIVE_ALPHA_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)
REQUIRED_COLUMNS = ("sample_id", "fold", "label", "probability")


def validate_oof_predictions(frame: pd.DataFrame, expert: str) -> pd.DataFrame:
    missing = set(REQUIRED_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"{expert} predictions missing columns: {sorted(missing)}")
    checked = frame.loc[:, REQUIRED_COLUMNS].copy()
    checked["sample_id"] = checked["sample_id"].astype(str)
    checked["fold"] = checked["fold"].astype(str)
    if checked["sample_id"].duplicated().any():
        duplicates = checked.loc[checked["sample_id"].duplicated(False), "sample_id"].tolist()
        raise ValueError(f"{expert} sample IDs appear in multiple held-out folds: {duplicates}")
    labels = checked["label"].to_numpy()
    probabilities = checked["probability"].to_numpy(dtype=float)
    if not np.isin(labels, [0, 1]).all():
        raise ValueError(f"{expert} labels must be 0 or 1")
    if not np.isfinite(probabilities).all() or not (
        (probabilities >= 0.0) & (probabilities <= 1.0)
    ).all():
        raise ValueError(f"{expert} probabilities must be finite and in [0,1]")
    if checked["fold"].eq("").any():
        raise ValueError(f"{expert} held-out fold must be non-empty")
    return checked


def align_oof_predictions(
    clip: pd.DataFrame,
    npr: pd.DataFrame,
    expected_folds: tuple[str, ...] | None = None,
) -> pd.DataFrame:
    """Require exact one-to-one, same-order alignment; never silently join/reorder."""
    clip_checked = validate_oof_predictions(clip, "CLIP")
    npr_checked = validate_oof_predictions(npr, "NPR")
    clip_ids = clip_checked["sample_id"].tolist()
    npr_ids = npr_checked["sample_id"].tolist()
    if clip_ids != npr_ids:
        if set(clip_ids) != set(npr_ids):
            missing_npr = sorted(set(clip_ids) - set(npr_ids))
            missing_clip = sorted(set(npr_ids) - set(clip_ids))
            raise ValueError(
                f"OOF sample ID sets differ; missing NPR={missing_npr}, "
                f"missing CLIP={missing_clip}"
            )
        raise ValueError("OOF sample IDs are reordered; exact input order is required")
    if not clip_checked["fold"].equals(npr_checked["fold"]):
        raise ValueError("OOF held-out fold assignment is misaligned")
    if not clip_checked["label"].equals(npr_checked["label"]):
        raise ValueError("OOF labels are misaligned")
    if expected_folds is not None:
        actual = set(clip_checked["fold"])
        expected = set(expected_folds)
        if actual != expected:
            raise ValueError(
                f"OOF fold set mismatch; expected={sorted(expected)}, actual={sorted(actual)}"
            )
    return pd.DataFrame(
        {
            "sample_id": clip_checked["sample_id"],
            "fold": clip_checked["fold"],
            "label": clip_checked["label"].astype(int),
            "clip_probability": clip_checked["probability"].astype(float),
            "npr_probability": npr_checked["probability"].astype(float),
        }
    )


def fuse_aligned_probabilities(
    aligned: pd.DataFrame, alpha_clip: float = PRIMARY_ALPHA_CLIP
) -> pd.DataFrame:
    if not 0.0 <= alpha_clip <= 1.0:
        raise ValueError("alpha_clip must be in [0,1]")
    fused = aligned.copy()
    fused["fused_probability"] = (
        alpha_clip * fused["clip_probability"]
        + (1.0 - alpha_clip) * fused["npr_probability"]
    )
    return fused
