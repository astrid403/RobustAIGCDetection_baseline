from pathlib import Path
import unittest

import yaml


ROOT = Path(__file__).resolve().parents[1]


class S2ContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.contract = yaml.safe_load(
            (ROOT / "configs/research_v3/S2_CONTRACT.yaml").read_text()
        )

    def test_multiblock_projection_and_tie_are_closed(self):
        contract = self.contract
        self.assertEqual(contract["encoder"]["block_ids"], [3, 6, 9, 12])
        self.assertEqual(contract["encoder"]["extraction"], "single_forward")
        self.assertEqual(
            contract["encoder"]["output_shape_per_sample"], [4, 768]
        )
        self.assertEqual(contract["projection"]["output_dim"], 128)
        self.assertEqual(
            contract["projection"]["sharing"],
            "one_module_shared_across_all_blocks",
        )
        self.assertEqual(
            contract["tie"]["normalization"], "softmax_across_four_blocks"
        )
        self.assertTrue(contract["tie"]["sample_conditioned"])

    def test_supcon_budget_and_gate_are_frozen(self):
        contract = self.contract
        supcon = contract["objective"]["supervised_contrastive"]
        self.assertEqual(supcon["weight"], 0.1)
        self.assertEqual(supcon["temperature"], 0.07)
        self.assertEqual(contract["training"]["batch_size"], 32)
        self.assertEqual(contract["training"]["epochs"], 10)
        self.assertEqual(
            contract["training"]["total_sample_draws_per_fold"], 20000
        )
        gate = contract["development_gate"]
        self.assertEqual(gate["minimum_clean_fold_mean_auroc_delta"], 0.005)
        self.assertEqual(gate["minimum_clean_worst_fold_auroc_delta"], -0.002)
        self.assertEqual(gate["minimum_improved_clean_folds"], 2)
        self.assertTrue(gate["all_criteria_required"])

    def test_external_selection_and_search_are_forbidden(self):
        forbidden = self.contract["forbidden"]
        for key in (
            "architecture_search",
            "block_search",
            "projection_search",
            "supcon_weight_search",
            "threshold_search",
            "external_data_selection",
            "genimage_unseen_before_model_freeze",
            "defactify_before_separate_approval",
            "s1_result_modification",
        ):
            self.assertTrue(forbidden[key])


if __name__ == "__main__":
    unittest.main()
