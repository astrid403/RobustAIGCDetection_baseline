import unittest

import pandas as pd

from analysis.part10_freeze import REQUIRED_METRICS, build_aggregate


class Part10FreezeTest(unittest.TestCase):
    def test_required_metric_contract(self):
        self.assertEqual(len(REQUIRED_METRICS), 9)
        self.assertIn("auroc", REQUIRED_METRICS)
        self.assertIn("balanced_accuracy", REQUIRED_METRICS)

    def test_build_aggregate_rejects_incomplete_matrix(self):
        with self.assertRaisesRegex(ValueError, "Expected"):
            build_aggregate_frame = pd.DataFrame([
                {"role": "B2", "scope": "genimage_unseen", "metric": "auroc",
                 "mean": 0.9, "sample_sd": 0.01}
            ])
            build_aggregate_frame.to_csv("/tmp/part10_incomplete_mean_sd.csv", index=False)
            pd.DataFrame([
                {"scope": "genimage_unseen", "metric": "auroc",
                 "a1_minus_b2": 0.01}
            ]).to_csv("/tmp/part10_incomplete_delta.csv", index=False)
            build_aggregate(
                "/tmp/part10_incomplete_mean_sd.csv",
                "/tmp/part10_incomplete_delta.csv",
            )


if __name__ == "__main__":
    unittest.main()
