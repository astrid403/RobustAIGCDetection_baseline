import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch
from torch.utils.data import DataLoader, Dataset
import yaml

from models.npr_detector import build_npr_detector
from training.engine import evaluate_loader, train_npr_epoch
from training.group_samplers import (
    DeterministicGroupBalancedSampler,
    validate_protocol_groups,
)
from training.objectives import build_npr_objective
from training.train import load_training_checkpoint, save_checkpoint, seed_everything


class TinyNprDataset(Dataset):
    def __init__(self):
        generator = torch.Generator().manual_seed(9)
        self.images = torch.rand(4, 3, 223, 223, generator=generator)
        self.labels = [0.0, 1.0, 0.0, 1.0]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.images[index], self.labels[index], f"sample-{index}"


class TinyDetector(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = torch.nn.AdaptiveAvgPool2d(1)
        self.fc = torch.nn.Linear(3, 1)

    def forward(self, inputs):
        return self.fc(self.pool(inputs).flatten(1))


class NprTrainingTests(unittest.TestCase):
    def test_detector_factory_uses_default_resnet18_weights_and_single_logit(self):
        fake = TinyDetector()
        fake.fc = torch.nn.Linear(3, 1000)
        with mock.patch("models.npr_detector.resnet18", return_value=fake) as factory:
            model = build_npr_detector(pretrained=True)
        self.assertEqual(factory.call_args.kwargs["weights"].name, "IMAGENET1K_V1")
        self.assertEqual(model.fc.in_features, 3)
        self.assertEqual(model.fc.out_features, 1)

    def test_sampler_balances_groups_and_is_epoch_deterministic(self):
        generators = ["ADM", "ADM", "BigGAN", "BigGAN"]
        labels = [0, 1, 0, 1]
        sampler = DeterministicGroupBalancedSampler(generators, labels, 5, seed=42)
        sampler.set_epoch(3)
        first = list(sampler)
        sampler.set_epoch(3)
        self.assertEqual(first, list(sampler))
        self.assertEqual(len(first), 20)
        self.assertEqual({first.count(index) for index in range(4)}, {5})
        sampler.set_epoch(4)
        self.assertNotEqual(first, list(sampler))

    def test_sampler_validation_rejects_wrong_contract(self):
        validate_protocol_groups(["A", "A", "B", "B"], [0, 1, 0, 1])
        with self.assertRaisesRegex(ValueError, "Expected 4"):
            validate_protocol_groups(["A", "A"], [0, 1])
        with self.assertRaisesRegex(ValueError, "same non-zero length"):
            DeterministicGroupBalancedSampler(["A"], [], 1, 42)

    def test_objective_is_bce_and_consistency_is_disabled(self):
        self.assertIsInstance(build_npr_objective(0.0), torch.nn.BCEWithLogitsLoss)
        with self.assertRaisesRegex(ValueError, "consistency_weight=0.0"):
            build_npr_objective(0.1)

    def test_forward_backward_metrics_are_finite(self):
        torch.manual_seed(4)
        model = TinyDetector()
        loader = DataLoader(TinyNprDataset(), batch_size=4, shuffle=False)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
        objective = build_npr_objective()
        loss = train_npr_epoch(model, loader, optimizer, objective, torch.device("cpu"))
        metrics = evaluate_loader(model, loader, objective, torch.device("cpu"))
        self.assertTrue(torch.isfinite(torch.tensor(loss)))
        for key in (
            "loss",
            "accuracy",
            "balanced_accuracy",
            "macro_f1",
            "auroc",
            "auprc",
            "real_recall",
            "fake_recall",
        ):
            self.assertTrue(torch.isfinite(torch.tensor(metrics[key])), key)

    def test_checkpoint_save_load_and_resume_state(self):
        torch.manual_seed(5)
        source = TinyDetector()
        optimizer = torch.optim.AdamW(source.parameters(), lr=1e-4)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            save_checkpoint(path, source, optimizer, 1, 0.75, {"seed": 42})
            target = TinyDetector()
            target_optimizer = torch.optim.AdamW(target.parameters(), lr=1e-4)
            checkpoint = load_training_checkpoint(
                path, target, target_optimizer, torch.device("cpu")
            )
            self.assertEqual(checkpoint["epoch"] + 1, 2)
            self.assertEqual(checkpoint["best_score"], 0.75)
            self.assertEqual(checkpoint["config"], {"seed": 42})
            for left, right in zip(source.parameters(), target.parameters()):
                self.assertTrue(torch.equal(left, right))

    def test_smoke_config_is_isolated_and_within_budget(self):
        path = Path("configs/research_v3/smoke_npr_synthetic_seed42_v3.yaml")
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        self.assertEqual(config["stage"], "smoke")
        self.assertEqual(config["model_type"], "npr_resnet18")
        self.assertEqual(config["epochs"], 1)
        self.assertEqual(config["batch_size"], 4)
        self.assertEqual(config["samples_per_group_per_epoch"], 1)
        self.assertEqual(config["output_root"], "outputs/research_v3")
        self.assertEqual(config["consistency_weight"], 0.0)
        self.assertNotIn("defactify", str(config).lower())
        self.assertNotIn("unseen", str(config).lower())

    def test_deterministic_seed_settings_are_enabled_before_model_creation(self):
        seed_everything(42, deterministic=True)
        self.assertEqual(torch.get_num_threads(), 1)
        self.assertTrue(torch.are_deterministic_algorithms_enabled())
        self.assertTrue(torch.backends.cudnn.deterministic)
        self.assertFalse(torch.backends.cudnn.benchmark)


if __name__ == "__main__":
    unittest.main()
