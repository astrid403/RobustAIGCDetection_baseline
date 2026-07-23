import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from PIL import Image

from data_pipeline.dataset_factory import create_image_dataset
from data_pipeline.defactify import balanced_test, calibration_test, rows_for_split, smoke_test, validate_labels
from data_pipeline.fingerprints import feature_cache_metadata, validate_feature_cache
from evaluation.summarize_predictions import metric_row, select_manifest


class DefactifyManifestTests(unittest.TestCase):
    def test_label_contract(self):
        validate_labels(0, 0)
        for label_b in range(1, 6):
            validate_labels(1, label_b)
        with self.assertRaises(ValueError):
            validate_labels(0, 1)
        with self.assertRaises(ValueError):
            validate_labels(1, 0)

    def test_rows_are_stable_and_complete(self):
        source = [
            {"Caption": "real", "Label_A": 0, "Label_B": 0},
            {"Caption": "fake", "Label_A": 1, "Label_B": 4},
        ]
        first = rows_for_split(source, "org/data", "abc123", "test")
        second = rows_for_split(source, "org/data", "abc123", "test")
        self.assertEqual(first, second)
        self.assertEqual(first[1]["generator"], "DALL-E 3")
        self.assertTrue(first[0]["sample_id"].startswith("defactify-"))

    def test_balanced_and_smoke_manifests(self):
        rows = []
        for split in ("train", "validation", "test"):
            for label_b in range(6):
                for index in range(10):
                    rows.append({"hf_split": split, "label_b": label_b, "label": int(label_b > 0), "row_index": index})
        frame = pd.DataFrame(rows)
        balanced = balanced_test(frame)
        self.assertEqual(int((balanced.label == 0).sum()), int((balanced.label == 1).sum()))
        smoke = smoke_test(frame, per_label_per_split=2)
        self.assertEqual(len(smoke), 36)
        calibration = calibration_test(frame, per_label=3)
        self.assertEqual(len(calibration), 18)
        self.assertEqual(set(calibration.hf_split), {"test"})


class DatasetAndCacheTests(unittest.TestCase):
    def test_local_manifest_remains_compatible(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_path = root / "sample.png"
            Image.new("RGB", (8, 8), "red").save(image_path)
            csv_path = root / "local.csv"
            pd.DataFrame([{"path": str(image_path), "label": 1, "is_valid": True}]).to_csv(csv_path, index=False)
            dataset = create_image_dataset(csv_path)
            image, label, sample_id = dataset[0]
            self.assertEqual(image.size, (8, 8))
            self.assertEqual(label, 1.0)
            self.assertEqual(sample_id, str(image_path))

    def test_huggingface_backend_decodes_native_image(self):
        from datasets import Dataset

        native = Dataset.from_dict({"Image": [Image.new("RGB", (9, 7), "blue")]})
        with tempfile.TemporaryDirectory() as directory:
            csv_path = Path(directory) / "hf.csv"
            pd.DataFrame([{
                "sample_id": "defactify-test",
                "storage_backend": "huggingface",
                "dataset_id": "org/data",
                "dataset_revision": "abc123",
                "hf_split": "test",
                "row_index": 0,
                "label": 1,
                "is_valid": True,
            }]).to_csv(csv_path, index=False)
            with patch("datasets.load_dataset", return_value=native) as mocked:
                dataset = create_image_dataset(csv_path)
                image, label, sample_id = dataset[0]
            self.assertEqual(image.size, (9, 7))
            self.assertEqual(label, 1.0)
            self.assertEqual(sample_id, "defactify-test")
            mocked.assert_called_once()

    def test_cache_fingerprint_detects_manifest_change(self):
        with tempfile.TemporaryDirectory() as directory:
            csv_path = Path(directory) / "manifest.csv"
            pd.DataFrame([{"path": "a.jpg", "label": 0}]).to_csv(csv_path, index=False)
            config = {"clip_model": "ViT-B-32", "pretrained": "openai"}
            first = feature_cache_metadata(csv_path, config)
            validate_feature_cache({"metadata": first}, first)
            pd.DataFrame([{"path": "b.jpg", "label": 1}]).to_csv(csv_path, index=False)
            second = feature_cache_metadata(csv_path, config)
            self.assertNotEqual(first["cache_signature"], second["cache_signature"])
            with self.assertRaises(ValueError):
                validate_feature_cache({"metadata": first}, second)

    def test_cache_fingerprint_separates_feature_modes(self):
        with tempfile.TemporaryDirectory() as directory:
            csv_path = Path(directory) / "manifest.csv"
            pd.DataFrame([{"path": "a.jpg", "label": 0}]).to_csv(csv_path, index=False)
            final = feature_cache_metadata(
                csv_path, {"clip_model": "ViT-B-32", "pretrained": "openai", "clip_feature_mode": "final"}
            )
            penultimate = feature_cache_metadata(
                csv_path,
                {"clip_model": "ViT-B-32", "pretrained": "openai", "clip_feature_mode": "penultimate"},
            )
            self.assertNotEqual(final["cache_signature"], penultimate["cache_signature"])
            with self.assertRaisesRegex(ValueError, "feature-mode"):
                validate_feature_cache({"metadata": final}, penultimate)

    def test_legacy_cache_is_rejected_explicitly(self):
        with tempfile.TemporaryDirectory() as directory:
            csv_path = Path(directory) / "manifest.csv"
            pd.DataFrame([{"path": "a.jpg", "label": 0}]).to_csv(csv_path, index=False)
            expected = feature_cache_metadata(csv_path, {"clip_model": "ViT-B-32"})
            with self.assertRaisesRegex(ValueError, "Legacy CLIP feature cache"):
                validate_feature_cache({"metadata": {"cache_signature": "old"}}, expected)

    def test_balanced_metrics_are_selected_from_full_predictions(self):
        predictions = pd.DataFrame([
            {"sample_id": "r1", "label": 0, "pred_label": 0, "fake_prob": 0.1, "label_b": 0, "generator": "real"},
            {"sample_id": "r2", "label": 0, "pred_label": 1, "fake_prob": 0.6, "label_b": 0, "generator": "real"},
            {"sample_id": "f1", "label": 1, "pred_label": 1, "fake_prob": 0.9, "label_b": 1, "generator": "SD21"},
            {"sample_id": "f2", "label": 1, "pred_label": 0, "fake_prob": 0.4, "label_b": 2, "generator": "SDXL"},
        ])
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "balanced.csv"
            pd.DataFrame({"sample_id": ["r1", "f1"]}).to_csv(manifest, index=False)
            selected = select_manifest(predictions, manifest)
        self.assertEqual(set(selected.sample_id), {"r1", "f1"})
        row, cm = metric_row(selected, "test", "balanced")
        self.assertEqual(row["accuracy"], 1.0)
        self.assertEqual(int(cm.sum()), 2)


if __name__ == "__main__":
    unittest.main()
