import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from PIL import Image

from data_pipeline.dataset_factory import create_image_dataset
from data_pipeline.defactify import balanced_test, rows_for_split, smoke_test, validate_labels
from data_pipeline.fingerprints import feature_cache_metadata, validate_feature_cache


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


if __name__ == "__main__":
    unittest.main()
