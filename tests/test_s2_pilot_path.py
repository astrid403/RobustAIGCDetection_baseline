from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd
from PIL import Image
import torch
from torch.utils.data import DataLoader, TensorDataset
import yaml

from evaluation.research_inference import predict_rine_lite
from models.rine_lite_detector import RineLiteDetector
from training.contrastive_losses import RineLiteObjective
from training.train import _rine_lite_epoch, _validate_s2_pilot_config


ROOT = Path(__file__).resolve().parents[1]
FOLDS = (
    "holdout_adm",
    "holdout_biggan",
    "holdout_stable_diffusion_v15",
)


class S2PilotPathTests(unittest.TestCase):
    def test_three_configs_match_frozen_budget(self):
        for fold in FOLDS:
            path = ROOT / f"configs/research_v3/pilot_s2_{fold}_seed42_v3.yaml"
            config = yaml.safe_load(path.read_text())
            _validate_s2_pilot_config(config)
            self.assertEqual(config["fold"], fold)
            self.assertEqual(config["model_type"], "rine_lite")
            self.assertEqual(config["clip_block_ids"], [3, 6, 9, 12])
            self.assertEqual(config["output_root"], "outputs/research_v3")

    def test_config_mismatch_hard_fails(self):
        config = yaml.safe_load(
            (
                ROOT / "configs/research_v3/pilot_s2_holdout_adm_seed42_v3.yaml"
            ).read_text()
        )
        for key, value in (
            ("clip_block_ids", [3, 6, 12, 9]),
            ("batch_size", 64),
            ("epochs", 9),
            ("supcon_weight", 0.2),
        ):
            invalid = dict(config)
            invalid[key] = value
            with self.assertRaisesRegex(ValueError, "frozen contract"):
                _validate_s2_pilot_config(invalid)

    def test_synthetic_cached_feature_epoch_is_finite(self):
        torch.manual_seed(42)
        features = torch.randn(32, 4, 768)
        labels = torch.tensor([0, 1] * 16)
        loader = DataLoader(
            TensorDataset(features, labels), batch_size=32, shuffle=False
        )
        model = RineLiteDetector()
        objective = RineLiteObjective()
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=0.001, weight_decay=0.01
        )
        train = _rine_lite_epoch(
            model, loader, objective, torch.device("cpu"), optimizer
        )
        with torch.no_grad():
            validation = _rine_lite_epoch(
                model, loader, objective, torch.device("cpu")
            )
        for result in (train, validation):
            self.assertTrue(torch.isfinite(torch.tensor(result["loss"])))
            self.assertEqual(len(result["importance_mean"]), 4)
            self.assertAlmostEqual(sum(result["importance_mean"]), 1.0, places=6)

    def test_prediction_schema_and_importance_order(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "image.png"
            Image.new("RGB", (8, 8), "gray").save(image)
            manifest = root / "validation.csv"
            pd.DataFrame(
                [{
                    "path": str(image),
                    "label": 1,
                    "sample_id": "tiny-s2",
                    "generator": "ADM",
                }]
            ).to_csv(manifest, index=False)
            model = RineLiteDetector()
            fake_features = torch.ones(1, 4, 768)
            with patch(
                "evaluation.research_inference.encode_clip_multiblock_cls",
                return_value=fake_features,
            ):
                predictions, importance = predict_rine_lite(
                    torch.nn.Identity(),
                    model,
                    lambda _: torch.ones(3, 224, 224),
                    manifest,
                    "holdout_adm",
                    "clean",
                    torch.device("cpu"),
                    batch_size=1,
                )
            self.assertEqual(
                predictions.columns.tolist(),
                ["sample_id", "fold", "label", "probability"],
            )
            self.assertEqual(
                importance.columns.tolist(),
                [
                    "sample_id",
                    "fold",
                    "block_3",
                    "block_6",
                    "block_9",
                    "block_12",
                ],
            )
            self.assertAlmostEqual(
                importance.filter(like="block_").iloc[0].sum(), 1.0, places=6
            )


if __name__ == "__main__":
    unittest.main()
