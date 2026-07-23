from pathlib import Path
import tempfile
import unittest

import pandas as pd
from PIL import Image

from analysis.audit_cross_dataset_splits import (
    build_logo_fold,
    hamming_hex,
    load_development_manifest,
    source_group,
    stable_sample_id,
)


class CrossDatasetSplitAuditTests(unittest.TestCase):
    def test_stable_sample_id_is_deterministic_and_path_specific(self):
        first = stable_sample_id("data/a.jpg")
        self.assertEqual(first, stable_sample_id("data/a.jpg"))
        self.assertNotEqual(first, stable_sample_id("data/b.jpg"))

    def test_real_source_group_uses_basename_across_generators(self):
        left = pd.Series(
            {"path": "root/ADM/train/nature/n1.JPEG", "label": 0, "generator": "ADM"}
        )
        right = pd.Series(
            {"path": "root/BigGAN/train/nature/n1.JPEG", "label": 0, "generator": "BigGAN"}
        )
        self.assertEqual(source_group(left), "real:n1.jpeg")
        self.assertEqual(source_group(right), "real:n1.jpeg")

    def test_hamming_hex(self):
        self.assertEqual(hamming_hex("0000000000000000", "0000000000000000"), 0)
        self.assertEqual(hamming_hex("0000000000000000", "000000000000000f"), 4)

    def test_logo_filters_source_exact_and_near_duplicates(self):
        train = pd.DataFrame(
            [
                {
                    "sample_id": "source_dup",
                    "generator": "ADM",
                    "label": 0,
                    "source_group": "real:n1.jpeg",
                    "content_sha256": "source-content",
                    "dhash64": "0000000000000000",
                },
                {
                    "sample_id": "near_dup",
                    "generator": "ADM",
                    "label": 1,
                    "source_group": "fake:ADM:a.jpg",
                    "content_sha256": "near-content",
                    "dhash64": "000000000000000f",
                },
                {
                    "sample_id": "keep_real",
                    "generator": "BigGAN",
                    "label": 0,
                    "source_group": "real:n2.jpeg",
                    "content_sha256": "keep-real",
                    "dhash64": "ffffffffffffffff",
                },
                {
                    "sample_id": "keep_fake",
                    "generator": "BigGAN",
                    "label": 1,
                    "source_group": "fake:BigGAN:b.jpg",
                    "content_sha256": "keep-fake",
                    "dhash64": "fffffffffffffff0",
                },
            ]
        )
        validation = pd.DataFrame(
            [
                {
                    "sample_id": "val_real",
                    "generator": "Stable Diffusion V1.5",
                    "label": 0,
                    "source_group": "real:n1.jpeg",
                    "content_sha256": "val-real",
                    "dhash64": "0000000000000000",
                },
                {
                    "sample_id": "val_fake",
                    "generator": "Stable Diffusion V1.5",
                    "label": 1,
                    "source_group": "fake:Stable Diffusion V1.5:c.jpg",
                    "content_sha256": "val-fake",
                    "dhash64": "0000000000000000",
                },
            ]
        )
        fold_train, fold_val, details = build_logo_fold(
            train, validation, "Stable Diffusion V1.5", max_hamming=4
        )
        self.assertEqual(set(fold_train.sample_id), {"keep_real", "keep_fake"})
        self.assertEqual(set(fold_val.sample_id), {"val_real", "val_fake"})
        self.assertEqual(details["blocked_source_group_rows"], 1)
        self.assertEqual(details["blocked_near_duplicate_rows"], 2)

    def test_loader_rejects_forbidden_manifest_name(self):
        with tempfile.TemporaryDirectory() as directory:
            forbidden = Path(directory) / "genimage_unseen.csv"
            forbidden.write_text("path,label\nx,0\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Forbidden"):
                load_development_manifest(forbidden, "validation", compute_hashes=False)

    def test_loader_builds_ids_and_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            rows = []
            for generator in ("ADM", "BigGAN", "Stable Diffusion V1.5"):
                for label in (0, 1):
                    image_path = tmp_path / f"{generator.replace(' ', '_')}_{label}.png"
                    Image.new("RGB", (12, 12), color=(label * 255, 20, 40)).save(image_path)
                    rows.append(
                        {
                            "path": str(image_path),
                            "label": label,
                            "dataset": "GenImage",
                            "split": "train",
                            "generator": generator,
                            "source": generator,
                            "width": 12,
                            "height": 12,
                            "format": "PNG",
                            "is_valid": True,
                        }
                    )
            manifest = tmp_path / "development.csv"
            pd.DataFrame(rows).to_csv(manifest, index=False)
            loaded = load_development_manifest(manifest, "train")
            self.assertEqual(loaded.sample_id.nunique(), 6)
            self.assertTrue(loaded.content_sha256.str.len().eq(64).all())
            self.assertTrue(loaded.dhash64.str.len().eq(16).all())


if __name__ == "__main__":
    unittest.main()
