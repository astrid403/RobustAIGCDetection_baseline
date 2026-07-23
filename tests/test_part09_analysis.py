import unittest

import pandas as pd

from analysis.part09_analysis import aggregate_metrics


class Part09AnalysisTest(unittest.TestCase):
    def test_mean_sd_and_paired_direction(self):
        rows = []
        for role, values in (("B2", [0.7, 0.8, 0.9]), ("A1", [0.8, 0.9, 1.0])):
            for seed, value in zip((42, 43, 44), values):
                row = {"role": role, "scope": "test", "seed": seed}
                row.update({metric: value for metric in (
                    "accuracy", "balanced_accuracy", "precision", "real_recall",
                    "fake_recall", "binary_f1", "macro_f1", "auroc", "auprc",
                )})
                rows.append(row)
        mean_sd, delta = aggregate_metrics(pd.DataFrame(rows))
        self.assertEqual(len(mean_sd), 18)
        self.assertTrue((delta["a1_minus_b2"].round(12) == 0.1).all())


if __name__ == "__main__":
    unittest.main()
