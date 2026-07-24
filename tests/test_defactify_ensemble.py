import tempfile
import unittest
from pathlib import Path

import pandas as pd

from evaluation.evaluate_defactify_ensemble import derive_balanced


class DefactifyEnsembleTests(unittest.TestCase):
    def test_balanced_is_exact_ordered_subset(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "balanced.csv"
            pd.DataFrame({"sample_id": ["c", "a"]}).to_csv(manifest, index=False)
            full = pd.DataFrame({
                "sample_id": ["a", "b", "c"], "label": [0, 1, 1],
                "fake_prob": [0.1, 0.8, 0.9],
            })
            result = derive_balanced(full, manifest)
            self.assertEqual(result.sample_id.tolist(), ["c", "a"])

    def test_missing_and_duplicate_ids_hard_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "balanced.csv"
            full = pd.DataFrame({"sample_id": ["a"], "label": [0],
                                 "fake_prob": [0.1]})
            pd.DataFrame({"sample_id": ["a", "a"]}).to_csv(manifest, index=False)
            with self.assertRaises(ValueError):
                derive_balanced(full, manifest)
            pd.DataFrame({"sample_id": ["z"]}).to_csv(manifest, index=False)
            with self.assertRaises(ValueError):
                derive_balanced(full, manifest)
