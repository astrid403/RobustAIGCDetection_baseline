"""Strict OOF fusion evaluation and provenance writer for research v3."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from analysis.cross_dataset_analysis import (
    descriptive_alpha_grid,
    rank_descriptive_alphas,
    summarize_oof,
)
from models.late_fusion import (
    DESCRIPTIVE_ALPHA_GRID,
    PRIMARY_ALPHA_CLIP,
    align_oof_predictions,
    fuse_aligned_probabilities,
)


def sha256_file(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def evaluate_oof_frames(
    clip: pd.DataFrame,
    npr: pd.DataFrame,
    expected_folds: tuple[str, ...] | None = None,
) -> tuple[pd.DataFrame, dict]:
    aligned = align_oof_predictions(clip, npr, expected_folds)
    fused = fuse_aligned_probabilities(aligned, PRIMARY_ALPHA_CLIP)
    summary = summarize_oof(fused)
    grid = descriptive_alpha_grid(aligned)
    summary["fusion_contract"] = {
        "primary_alpha_clip": PRIMARY_ALPHA_CLIP,
        "primary_alpha_npr": 1.0 - PRIMARY_ALPHA_CLIP,
        "threshold": 0.5,
        "descriptive_alpha_grid": list(DESCRIPTIVE_ALPHA_GRID),
        "grid_can_replace_primary": False,
        "descriptive_tie_break": [
            "fold_mean_auroc_desc",
            "distance_to_primary_alpha_asc",
            "alpha_clip_asc",
        ],
    }
    summary["descriptive_alpha_results"] = grid
    summary["descriptive_alpha_ranking"] = rank_descriptive_alphas(grid)
    return fused, summary


def write_oof_outputs(
    clip_prediction_path,
    npr_prediction_path,
    output_dir,
    *,
    config_path,
    split_paths,
    checkpoint_paths,
    branch,
    commit,
    expected_folds=None,
) -> dict:
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite OOF output directory: {output_dir}")
    output_dir.mkdir(parents=True)
    clip = pd.read_csv(clip_prediction_path)
    npr = pd.read_csv(npr_prediction_path)
    fused, summary = evaluate_oof_frames(clip, npr, expected_folds)
    prediction_output = output_dir / "fused_oof_predictions.csv"
    metric_output = output_dir / "oof_metrics.json"
    registry_output = output_dir / "registry.json"
    fused.to_csv(prediction_output, index=False)
    metric_output.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    registry = {
        "schema": "research_oof_fusion_registry_v3",
        "branch": branch,
        "commit": commit,
        "config": {"path": str(config_path), "sha256": sha256_file(config_path)},
        "splits": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in sorted(split_paths.items())
        },
        "checkpoints": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in sorted(checkpoint_paths.items())
        },
        "input_predictions": {
            "clip": {
                "path": str(clip_prediction_path),
                "sha256": sha256_file(clip_prediction_path),
            },
            "npr": {
                "path": str(npr_prediction_path),
                "sha256": sha256_file(npr_prediction_path),
            },
        },
        "outputs": {
            "fused_predictions": {
                "path": str(prediction_output),
                "sha256": sha256_file(prediction_output),
            },
            "metrics": {
                "path": str(metric_output),
                "sha256": sha256_file(metric_output),
            },
        },
        "sample_count": len(fused),
        "sample_id_sha256": hashlib.sha256(
            "\n".join(fused["sample_id"]).encode("utf-8")
        ).hexdigest(),
        "primary_alpha_clip": PRIMARY_ALPHA_CLIP,
        "threshold": 0.5,
        "development_predictions": "out_of_fold_only",
        "genimage_unseen_accessed": False,
        "defactify_accessed": False,
    }
    registry_output.write_text(
        json.dumps(registry, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return registry
