import hashlib
import json
from pathlib import Path
import unittest

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "configs/research_v3/s1_contract.yaml"
PROTOCOL_PATH = ROOT / "docs/cross_dataset_plan/PROTOCOL_V3.md"
MANIFEST_PATH = ROOT / "artifacts/research_v3/protocol_v3_candidate_manifest.json"
APPROVAL_PATH = ROOT / "artifacts/research_v3/protocol_v3_approval.json"


class ResearchV3ContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.contract = yaml.safe_load(CONTRACT_PATH.read_text(encoding="utf-8"))
        cls.protocol = PROTOCOL_PATH.read_text(encoding="utf-8")

    def test_contract_is_candidate_and_isolated(self):
        self.assertEqual(self.contract["status"], "candidate_awaiting_user_approval")
        self.assertEqual(self.contract["branch"], "research/cross-dataset-robustness-v3")
        self.assertEqual(
            self.contract["frozen_base_commit"],
            "cedc2a18d956948acfed87d575d41d4b93689d1d",
        )
        self.assertTrue(self.contract["forbidden"]["overwrite_protocol_v2"])

    def test_data_roles_and_folds_are_closed(self):
        data = self.contract["data"]
        self.assertEqual(data["development_scope"], "train_and_validation_only")
        self.assertEqual(len(data["folds"]), 3)
        self.assertEqual(
            {item["held_out_generator"] for item in data["folds"]},
            {"ADM", "BigGAN", "Stable Diffusion V1.5"},
        )
        for fold in data["folds"]:
            self.assertIn("outputs/research_v3/audits/", fold["train_csv"])
            self.assertIn("outputs/research_v3/audits/", fold["validation_csv"])
            self.assertNotIn("unseen", fold["train_csv"].lower())
            self.assertNotIn("defactify", fold["validation_csv"].lower())

    def test_npr_tensor_contract_is_exact(self):
        npr = self.contract["npr"]
        self.assertEqual(npr["definition"], "rgb[:, :-1, :-1] - rgb[:, 1:, 1:]")
        self.assertEqual(npr["padding"], "none")
        self.assertEqual(npr["output_shape"], [3, 223, 223])
        self.assertEqual(npr["signed_range"], [-1.0, 1.0])
        self.assertEqual(npr["range_mapping"], "(npr + 1) / 2")

    def test_semantic_breaking_has_no_open_search_space(self):
        transform = self.contract["semantic_breaking"]
        self.assertEqual(transform["grid"], [4, 4])
        self.assertEqual(transform["patch_size"], [56, 56])
        self.assertEqual(transform["training_probability"], 0.5)
        self.assertEqual(transform["validation_probability"], 0.0)
        self.assertTrue(transform["reject_identity_permutation"])

    def test_model_training_and_sampler_are_fixed(self):
        model = self.contract["npr_expert"]
        self.assertEqual(model["backbone"], "resnet18")
        self.assertEqual(model["pretrained"], "imagenet_default")
        self.assertEqual(model["learning_rate"], 0.0001)
        self.assertEqual(model["batch_size"], 32)
        self.assertEqual(model["epochs"], 10)
        self.assertFalse(model["consistency_enabled_main"])
        sampler = self.contract["sampler"]
        self.assertEqual(sampler["samples_per_group_per_epoch"], 500)
        self.assertEqual(sampler["total_samples_per_epoch"], 2000)

    def test_degradations_and_optional_consistency_are_fixed(self):
        degradations = self.contract["degradations"]
        self.assertEqual(degradations["jpeg"]["quality"], 70)
        self.assertEqual(degradations["resize"]["scale"], 0.5)
        self.assertEqual(degradations["blur"]["radius"], 1.0)
        ablation = degradations["consistency_ablation"]
        self.assertFalse(ablation["enabled_main"])
        self.assertEqual(ablation["partner_probability"], 0.5)
        self.assertEqual(ablation["weight"], 0.1)

    def test_primary_fusion_cannot_be_replaced_by_grid(self):
        fusion = self.contract["fusion"]
        self.assertEqual(fusion["primary_alpha_clip"], 0.5)
        self.assertEqual(fusion["primary_alpha_npr"], 0.5)
        self.assertEqual(fusion["descriptive_alpha_grid"], [0.0, 0.25, 0.5, 0.75, 1.0])
        self.assertFalse(fusion["alpha_grid_can_replace_primary"])
        self.assertEqual(fusion["development_predictions"], "out_of_fold_only")

    def test_seeds_gates_naming_cache_and_provenance_are_fixed(self):
        self.assertEqual(self.contract["training"]["formal_seeds"], [42, 43, 44])
        self.assertEqual(
            self.contract["selection"]["task08_gate"]["minimum_mean_auroc_delta"], 0.005
        )
        self.assertEqual(
            self.contract["selection"]["task10_gate"]["maximum_peak_gpu_memory_mib"], 8000
        )
        self.assertEqual(
            self.contract["naming"]["template"],
            "{stage}_{model}_{training_scope}_{evaluation_scope}_seed{seed}_v3",
        )
        self.assertEqual(self.contract["cache"]["schema"], "research_clip_cache_v3")
        self.assertIn("failure_reason", self.contract["provenance"]["required"])

    def test_protocol_contains_the_machine_contract_values(self):
        required_text = (
            "rgb[c,y,x] = rgb[c,y,x] - rgb[c,y+1,x+1]",
            "4 x 4",
            "ResNet18",
            "0.0001",
            "0.5 * p_clip + 0.5 * p_npr",
            "[0.0,0.25,0.5,0.75,1.0]",
            "42, 43, 44",
            "8,000 MiB",
            "separate immediate approval",
        )
        # The first phrase is expressed as the npr equation in the document.
        required_text = ("npr[c,y,x] = rgb[c,y,x] - rgb[c,y+1,x+1]",) + required_text[1:]
        for value in required_text:
            self.assertIn(value, self.protocol)

    def test_candidate_manifest_hashes_and_authorization_boundary(self):
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        for key in ("protocol", "contract", "data_audit", "audit_registry"):
            record = manifest[key]
            payload = (ROOT / record["path"]).read_bytes()
            self.assertEqual(hashlib.sha256(payload).hexdigest(), record["sha256"])
        self.assertTrue(manifest["approval_required"])
        self.assertFalse(manifest["implementation_authorized"])
        self.assertFalse(manifest["training_authorized"])
        self.assertFalse(manifest["genimage_unseen_authorized"])
        self.assertFalse(manifest["defactify_authorized"])

    def test_approval_record_authorizes_only_tasks_four_and_five(self):
        approval = json.loads(APPROVAL_PATH.read_text(encoding="utf-8"))
        self.assertEqual(approval["status"], "approved")
        self.assertEqual(approval["candidate_commit"], "caf45034542ae35090977347b9685ab132e34754")
        for key in ("protocol", "contract"):
            record = approval[key]
            payload = (ROOT / record["path"]).read_bytes()
            self.assertEqual(hashlib.sha256(payload).hexdigest(), record["sha256"])
        self.assertEqual(approval["implementation_authorized_through_task"], "Task 05")
        self.assertFalse(approval["training_authorized"])
        self.assertFalse(approval["genimage_unseen_authorized"])
        self.assertFalse(approval["defactify_authorized"])


if __name__ == "__main__":
    unittest.main()
