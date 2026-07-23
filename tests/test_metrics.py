import math
import unittest

import numpy as np

from evaluation.metrics import binary_metrics


class BinaryMetricsTest(unittest.TestCase):
    def test_perfect_predictions(self):
        result = binary_metrics([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
        for name in (
            "accuracy", "balanced_accuracy", "precision", "fake_recall",
            "real_recall", "binary_f1", "macro_f1", "auroc", "auprc",
        ):
            self.assertAlmostEqual(result[name], 1.0)
        np.testing.assert_array_equal(result["confusion_matrix"], [[2, 0], [0, 2]])

    def test_completely_wrong_predictions(self):
        result = binary_metrics([0, 0, 1, 1], [0.8, 0.9, 0.1, 0.2])
        for name in (
            "accuracy", "balanced_accuracy", "precision", "fake_recall",
            "real_recall", "binary_f1", "macro_f1", "auroc",
        ):
            self.assertAlmostEqual(result[name], 0.0)
        self.assertGreaterEqual(result["auprc"], 0.0)
        self.assertLessEqual(result["auprc"], 1.0)
        np.testing.assert_array_equal(result["confusion_matrix"], [[0, 2], [2, 0]])

    def test_imbalanced_classes_and_metric_directions(self):
        result = binary_metrics([0, 0, 0, 1], [0.1, 0.2, 0.8, 0.9])
        self.assertAlmostEqual(result["accuracy"], 0.75)
        self.assertAlmostEqual(result["balanced_accuracy"], 5 / 6)
        self.assertAlmostEqual(result["precision"], 0.5)
        self.assertAlmostEqual(result["fake_recall"], 1.0)
        self.assertAlmostEqual(result["real_recall"], 2 / 3)
        self.assertAlmostEqual(result["binary_f1"], 2 / 3)
        self.assertAlmostEqual(result["macro_f1"], (2 / 3 + 0.8) / 2)
        self.assertAlmostEqual(result["auprc"], 1.0)

    def test_backward_compatible_aliases(self):
        result = binary_metrics([0, 1], [0.2, 0.8])
        self.assertEqual(result["recall"], result["fake_recall"])
        self.assertEqual(result["f1"], result["binary_f1"])

    def test_threshold_changes_predictions(self):
        default = binary_metrics([0, 1], [0.4, 0.6])
        strict = binary_metrics([0, 1], [0.4, 0.6], threshold=0.7)
        self.assertEqual(default["accuracy"], 1.0)
        self.assertEqual(strict["accuracy"], 0.5)
        self.assertEqual(strict["fake_recall"], 0.0)
        np.testing.assert_array_equal(strict["confusion_matrix"], [[1, 0], [1, 0]])

    def test_single_class_behavior_is_explicit(self):
        result = binary_metrics([0, 0], [0.1, 0.2])
        self.assertTrue(math.isnan(result["balanced_accuracy"]))
        self.assertTrue(math.isnan(result["auroc"]))
        self.assertTrue(math.isnan(result["auprc"]))
        self.assertEqual(result["accuracy"], 1.0)
        self.assertEqual(result["real_recall"], 1.0)
        self.assertEqual(result["fake_recall"], 0.0)
        np.testing.assert_array_equal(result["confusion_matrix"], [[2, 0], [0, 0]])

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):
            binary_metrics([], [])
        with self.assertRaises(ValueError):
            binary_metrics([0], [0.1, 0.2])
        with self.assertRaises(ValueError):
            binary_metrics([2], [0.5])
        with self.assertRaises(ValueError):
            binary_metrics([0], [float("nan")])
        with self.assertRaises(ValueError):
            binary_metrics([0], [0.5], threshold=1.1)


if __name__ == "__main__":
    unittest.main()
