import unittest
import torch

from evaluation.evaluate_genimage_unseen_ensemble import mean_probabilities


class EnsembleContractTests(unittest.TestCase):
    def test_exact_equal_probability_mean(self):
        result = mean_probabilities([
            torch.tensor([0.0, 0.3]), torch.tensor([0.3, 0.6]),
            torch.tensor([0.6, 0.9]),
        ])
        torch.testing.assert_close(result, torch.tensor([0.3, 0.6],
                                   dtype=torch.float64))

    def test_requires_exactly_three_members(self):
        with self.assertRaises(ValueError):
            mean_probabilities([torch.tensor([0.1]), torch.tensor([0.2])])
