import unittest

import torch

from models.clip_mlp_detector import (
    ClipLinearDetector,
    ClipMlpDetector,
    build_clip_feature_detector,
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

    def test_factory_builds_linear_probe(self):
        model = build_clip_feature_detector("clip_linear", 8)
        self.assertIsInstance(model, ClipLinearDetector)
        self.assertTrue(all(parameter.requires_grad for parameter in model.parameters()))

    def test_unknown_feature_detector_is_rejected(self):
        with self.assertRaises(ValueError):
            build_clip_feature_detector("clip_fusion", 8)


if __name__ == "__main__":
    unittest.main()
