import unittest
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import torch
import yaml

from models.clip_mlp_detector import (
    ClipLinearDetector,
    ClipMlpDetector,
    build_clip_feature_detector,
    load_open_clip_model,
)


class ClipFeatureDetectorTest(unittest.TestCase):
    def test_linear_probe_shape_and_parameter_count(self):
        model = ClipLinearDetector(feature_dim=8)
        output = model(torch.randn(3, 8))
        self.assertEqual(tuple(output.shape), (3,))
        self.assertEqual(sum(parameter.numel() for parameter in model.parameters()), 9)

    def test_existing_mlp_shape_and_state_dict_compatibility(self):
        old_model = ClipMlpDetector(feature_dim=8, hidden_dim=4, dropout=0.0)
        rebuilt = build_clip_feature_detector("clip_mlp", 8, hidden_dim=4, dropout=0.0)
        rebuilt.load_state_dict(old_model.state_dict())
        old_model.eval()
        rebuilt.eval()
        features = torch.randn(3, 8)
        torch.testing.assert_close(old_model(features), rebuilt(features))

    def test_mlp_checkpoint_round_trip(self):
        model = ClipMlpDetector(feature_dim=8, hidden_dim=4, dropout=0.0)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "model.pt"
            torch.save({"model_state": model.state_dict()}, checkpoint)
            rebuilt = build_clip_feature_detector("clip_mlp", 8, hidden_dim=4, dropout=0.0)
            rebuilt.load_state_dict(torch.load(checkpoint, map_location="cpu")["model_state"])
        model.eval()
        rebuilt.eval()
        features = torch.randn(3, 8)
        torch.testing.assert_close(model(features), rebuilt(features))

    def test_open_clip_loader_freezes_encoder_and_forwards_pretrained_tag(self):
        fake_model = torch.nn.Linear(2, 2)
        create = lambda model_name, pretrained: (fake_model, None, "preprocess")
        fake_open_clip = SimpleNamespace(create_model_and_transforms=create)
        with patch.dict(sys.modules, {"open_clip": fake_open_clip}):
            model, preprocess = load_open_clip_model("ViT-B-32", "cpu", "openai")
        self.assertIs(model, fake_model)
        self.assertEqual(preprocess, "preprocess")
        self.assertFalse(model.training)
        self.assertTrue(all(not parameter.requires_grad for parameter in model.parameters()))

    def test_penultimate_pilot_config_keeps_b2_head_budget(self):
        path = Path(__file__).resolve().parents[1] / "configs" / "pilot_clip_penultimate_genimage_unseen_seed42_v2.yaml"
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        self.assertEqual(config["experiment_name"], "pilot_clip_penultimate_genimage_unseen_seed42_v2")
        self.assertEqual(config["model_type"], "clip_mlp")
        self.assertEqual(config["clip_feature_mode"], "penultimate")
        self.assertEqual(config["mlp_hidden_dim"], 512)
        self.assertEqual(config["dropout"], 0.2)
        self.assertEqual(config["seed"], 42)
        self.assertEqual(config["threshold"], 0.5)

    def test_factory_builds_linear_probe(self):
        model = build_clip_feature_detector("clip_linear", 8)
        self.assertIsInstance(model, ClipLinearDetector)
        self.assertTrue(all(parameter.requires_grad for parameter in model.parameters()))

    def test_unknown_feature_detector_is_rejected(self):
        with self.assertRaises(ValueError):
            build_clip_feature_detector("clip_fusion", 8)


if __name__ == "__main__":
    unittest.main()
