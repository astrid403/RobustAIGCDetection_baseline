import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from analysis.cross_dataset_analysis import (
    descriptive_alpha_grid,
    error_complementarity,
    rank_descriptive_alphas,
    summarize_oof,
)
from evaluation.evaluate_cross_dataset import evaluate_oof_frames
from models.late_fusion import align_oof_predictions, fuse_aligned_probabilities


def toy_frames():
    clip = pd.DataFrame(
        {
            "sample_id": ["a0", "a1", "b0", "b1", "c0", "c1"],
            "fold": ["a", "a", "b", "b", "c", "c"],
            "label": [0, 1, 0, 1, 0, 1],
            "probability": [0.1, 0.6, 0.7, 0.4, 0.2, 0.8],
        }
    )
    npr = clip.copy()
    npr["probability"] = [0.3, 0.9, 0.2, 0.8, 0.6, 0.4]
    return clip, npr


class OofFusionTests(unittest.TestCase):
    def test_exact_alignment_and_fixed_half_fusion(self):
        clip, npr = toy_frames()
        aligned = align_oof_predictions(clip, npr, ("a", "b", "c"))
        fused = fuse_aligned_probabilities(aligned)
        expected = (clip["probability"] + npr["probability"]) / 2
        self.assertTrue(np.array_equal(fused["fused_probability"], expected))
        self.assertEqual(fused["sample_id"].tolist(), clip["sample_id"].tolist())

    def test_duplicate_missing_reordered_fold_and_label_mismatches_hard_fail(self):
        clip, npr = toy_frames()
        duplicate = clip.copy()
        duplicate.loc[1, "sample_id"] = "a0"
        with self.assertRaisesRegex(ValueError, "multiple held-out folds"):
            align_oof_predictions(duplicate, npr)
        with self.assertRaisesRegex(ValueError, "sets differ"):
            align_oof_predictions(clip, npr.iloc[:-1])
        with self.assertRaisesRegex(ValueError, "reordered"):
            align_oof_predictions(clip, npr.iloc[::-1].reset_index(drop=True))
        wrong_fold = npr.copy()
        wrong_fold.loc[0, "fold"] = "b"
        with self.assertRaisesRegex(ValueError, "fold assignment"):
            align_oof_predictions(clip, wrong_fold)
        wrong_label = npr.copy()
        wrong_label.loc[0, "label"] = 1
        with self.assertRaisesRegex(ValueError, "labels"):
            align_oof_predictions(clip, wrong_label)
        with self.assertRaisesRegex(ValueError, "fold set mismatch"):
            align_oof_predictions(clip, npr, ("a", "b", "missing"))

    def test_metrics_are_independently_recomputable(self):
        clip, npr = toy_frames()
        fused, summary = evaluate_oof_frames(clip, npr, ("a", "b", "c"))
        recomputed = summarize_oof(fused)
        self.assertEqual(summary["overall"], recomputed["overall"])
        self.assertEqual(summary["folds"], recomputed["folds"])
        self.assertEqual(summary["fold_aggregate"], recomputed["fold_aggregate"])
        self.assertEqual(summary["complementarity"], recomputed["complementarity"])
        self.assertEqual(set(summary["overall"]), {
            "auroc", "auprc", "balanced_accuracy", "macro_f1",
            "real_recall", "fake_recall",
        })

    def test_error_overlap_and_complementarity_counts(self):
        clip, npr = toy_frames()
        aligned = align_oof_predictions(clip, npr)
        result = error_complementarity(aligned)
        self.assertEqual(result["clip_error_count"], 2)
        self.assertEqual(result["npr_error_count"], 2)
        self.assertEqual(result["both_error_count"], 0)
        self.assertEqual(result["clip_wrong_npr_right_count"], 2)
        self.assertEqual(result["npr_wrong_clip_right_count"], 2)
        self.assertEqual(result["error_jaccard"], 0.0)

    def test_descriptive_grid_and_tie_break_cannot_replace_primary(self):
        clip, npr = toy_frames()
        aligned = align_oof_predictions(clip, npr)
        rows = descriptive_alpha_grid(aligned)
        self.assertEqual([row["alpha_clip"] for row in rows], [0, 0.25, 0.5, 0.75, 1])
        tied = [
            {"alpha_clip": 0.0, "fold_mean_auroc": 0.8},
            {"alpha_clip": 1.0, "fold_mean_auroc": 0.8},
            {"alpha_clip": 0.5, "fold_mean_auroc": 0.8},
            {"alpha_clip": 0.25, "fold_mean_auroc": 0.8},
            {"alpha_clip": 0.75, "fold_mean_auroc": 0.8},
        ]
        self.assertEqual(
            [row["alpha_clip"] for row in rank_descriptive_alphas(tied)],
            [0.5, 0.25, 0.75, 0.0, 1.0],
        )
        _, summary = evaluate_oof_frames(clip, npr)
        self.assertEqual(summary["fusion_contract"]["primary_alpha_clip"], 0.5)
        self.assertFalse(summary["fusion_contract"]["grid_can_replace_primary"])

    def test_invalid_probabilities_and_alpha_fail(self):
        clip, npr = toy_frames()
        clip.loc[0, "probability"] = np.nan
        with self.assertRaisesRegex(ValueError, "finite"):
            align_oof_predictions(clip, npr)
        clip, npr = toy_frames()
        with self.assertRaisesRegex(ValueError, "alpha_clip"):
            fuse_aligned_probabilities(align_oof_predictions(clip, npr), 1.1)


if __name__ == "__main__":
    unittest.main()
