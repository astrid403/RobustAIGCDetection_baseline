import unittest

import torch

from models.rine_lite_detector import (
    RineLiteDetector,
    build_rine_lite_detector,
)
from training.contrastive_losses import (
    RineLiteObjective,
    supervised_contrastive_loss,
)


class RineLiteHeadTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.features = torch.randn(8, 4, 768)
        self.labels = torch.tensor([0, 0, 1, 1, 0, 0, 1, 1])

    def test_frozen_architecture_shapes_importance_and_parameter_count(self):
        model = build_rine_lite_detector()
        output = model.forward_with_aux(self.features)
        self.assertEqual(tuple(output["logits"].shape), (8,))
        self.assertEqual(tuple(output["embedding"].shape), (8, 128))
        self.assertEqual(tuple(output["importance"].shape), (8, 4))
        torch.testing.assert_close(
            output["importance"].sum(dim=1),
            torch.ones(8),
            rtol=0,
            atol=2e-7,
        )
        torch.testing.assert_close(
            output["embedding"].norm(dim=1),
            torch.ones(8),
            rtol=0,
            atol=1e-6,
        )
        self.assertEqual(
            sum(parameter.numel() for parameter in model.parameters()), 98690
        )
        self.assertEqual(len(list(model.modules())), 5)
        self.assertFalse(
            any(isinstance(module, torch.nn.Dropout) for module in model.modules())
        )

    def test_shared_projection_and_sample_conditioned_tie(self):
        model = RineLiteDetector()
        output = model.forward_with_aux(self.features)
        projected = model.activation(model.projection(self.features))
        expected_importance = torch.softmax(
            model.importance_scorer(projected).squeeze(-1), dim=1
        )
        expected_embedding = torch.nn.functional.normalize(
            (projected * expected_importance.unsqueeze(-1)).sum(dim=1), dim=1
        )
        torch.testing.assert_close(output["importance"], expected_importance)
        torch.testing.assert_close(output["embedding"], expected_embedding)
        self.assertFalse(
            torch.equal(output["importance"][0], output["importance"][1])
        )

    def test_forward_backward_objective_and_checkpoint_round_trip(self):
        model = RineLiteDetector()
        objective = RineLiteObjective()
        output = model.forward_with_aux(self.features)
        losses = objective(output["logits"], output["embedding"], self.labels)
        self.assertEqual(set(losses), {"total", "bce", "supcon"})
        torch.testing.assert_close(
            losses["total"],
            losses["bce"] + 0.1 * losses["supcon"],
            rtol=0,
            atol=0,
        )
        self.assertTrue(all(torch.isfinite(value) for value in losses.values()))
        losses["total"].backward()
        for parameter in model.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())

        rebuilt = RineLiteDetector()
        rebuilt.load_state_dict(model.state_dict())
        with torch.no_grad():
            expected = model.forward_with_aux(self.features)
            actual = rebuilt.forward_with_aux(self.features)
        for key in expected:
            torch.testing.assert_close(expected[key], actual[key], rtol=0, atol=0)

    def test_supcon_is_permutation_invariant_and_skips_empty_anchors(self):
        embeddings = torch.tensor(
            [
                [1.0, 0.0],
                [0.9, 0.1],
                [0.0, 1.0],
                [0.1, 0.9],
            ]
        )
        labels = torch.tensor([0, 0, 1, 1])
        original = supervised_contrastive_loss(embeddings, labels)
        permutation = torch.tensor([2, 0, 3, 1])
        permuted = supervised_contrastive_loss(
            embeddings[permutation], labels[permutation]
        )
        torch.testing.assert_close(original, permuted, rtol=1e-6, atol=1e-6)
        self.assertGreaterEqual(float(original), 0.0)

        unique_embeddings = torch.randn(2, 5, requires_grad=True)
        unique_labels = torch.tensor([0, 1])
        empty = supervised_contrastive_loss(unique_embeddings, unique_labels)
        self.assertEqual(float(empty.detach()), 0.0)
        empty.backward()
        self.assertTrue(torch.equal(unique_embeddings.grad, torch.zeros_like(unique_embeddings)))

    def test_invalid_features_labels_temperature_and_logits_hard_fail(self):
        model = RineLiteDetector()
        for invalid in (
            torch.randn(8, 768),
            torch.randn(8, 3, 768),
            torch.randn(8, 4, 512),
            torch.empty(0, 4, 768),
        ):
            with self.assertRaisesRegex(ValueError, "shape|non-empty"):
                model(invalid)
        nonfinite = self.features.clone()
        nonfinite[0, 0, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite"):
            model(nonfinite)
        with self.assertRaisesRegex(ValueError, "labels"):
            supervised_contrastive_loss(
                torch.randn(3, 4), torch.tensor([0, 1])
            )
        with self.assertRaisesRegex(ValueError, "binary"):
            supervised_contrastive_loss(
                torch.randn(3, 4), torch.tensor([0, 1, 2])
            )
        with self.assertRaisesRegex(ValueError, "temperature"):
            supervised_contrastive_loss(
                torch.randn(3, 4), torch.tensor([0, 0, 1]), temperature=0
            )
        with self.assertRaisesRegex(ValueError, "logits"):
            RineLiteObjective()(
                torch.randn(3, 1), torch.randn(3, 128), torch.tensor([0, 0, 1])
            )

    def test_cpu_gpu_consistency_when_available(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA is unavailable")
        cpu_model = RineLiteDetector().eval()
        gpu_model = RineLiteDetector().eval().cuda()
        gpu_model.load_state_dict(cpu_model.state_dict())
        with torch.no_grad():
            cpu = cpu_model.forward_with_aux(self.features)
            gpu = gpu_model.forward_with_aux(self.features.cuda())
        for key in cpu:
            torch.testing.assert_close(
                cpu[key], gpu[key].cpu(), rtol=1e-5, atol=2e-6
            )


if __name__ == "__main__":
    unittest.main()
