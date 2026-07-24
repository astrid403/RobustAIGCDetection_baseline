import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import pandas as pd
from PIL import Image
import torch
from torchvision import transforms
import yaml

from data_pipeline.fingerprints import (
    RESEARCH_CLIP_CACHE_SCHEMA,
    research_clip_cache_metadata,
    research_clip_cache_path,
    validate_research_clip_cache,
)
from evaluation.research_inference import (
    DEGRADATIONS,
    create_run_directories,
    load_best_checkpoint,
    predict_b2,
    predict_npr,
    save_degradation_predictions,
    sha256_file,
    write_run_registry,
)
from models.clip_mlp_detector import save_feature_cache, load_feature_cache


class ConstantNpr(torch.nn.Module):
    def __init__(self, bias=0.0):
        super().__init__()
        self.bias = torch.nn.Parameter(torch.tensor(float(bias)))

    def forward(self, images):
        return self.bias.expand(images.shape[0])


class TinyEncoder(torch.nn.Module):
    def encode_image(self, images):
        return images.mean(dim=(2, 3))


class TinyHead(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 1)

    def forward(self, features):
        return self.linear(features).view(-1)


class ResearchV3PreflightTests(unittest.TestCase):
    def _tiny_manifest(self, root):
        paths = []
        for index, value in enumerate((32, 224)):
            path = root / f"image_{index}.png"
            Image.new("RGB", (8, 8), (value, value // 2, 255 - value)).save(path)
            paths.append(path)
        manifest = root / "validation.csv"
        pd.DataFrame(
            {
                "path": [str(path) for path in paths],
                "label": [0, 1],
                "generator": ["ADM", "ADM"],
                "sample_id": ["tiny-real", "tiny-fake"],
            }
        ).to_csv(manifest, index=False)
        return manifest

    def test_research_cache_schema_fingerprint_and_isolation(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = self._tiny_manifest(Path(directory))
            config = {
                "clip_model": "ViT-B-32",
                "pretrained": "openai",
                "clip_feature_mode": "final",
                "image_size": 224,
            }
            first = research_clip_cache_metadata(manifest, config, "holdout_adm", "train")
            second = research_clip_cache_metadata(
                manifest, config, "holdout_biggan", "train"
            )
            self.assertEqual(first["schema"], RESEARCH_CLIP_CACHE_SCHEMA)
            self.assertNotEqual(first["cache_signature"], second["cache_signature"])
            self.assertTrue(
                str(research_clip_cache_path(first)).startswith(
                    "outputs/research_v3/features/"
                )
            )
            cache_path = Path(directory) / "cache.pt"
            save_feature_cache(
                cache_path,
                torch.ones(2, 512),
                torch.tensor([0.0, 1.0]),
                ["tiny-real", "tiny-fake"],
                first,
            )
            cache = load_feature_cache(cache_path)
            validate_research_clip_cache(cache, first)
            with self.assertRaisesRegex(ValueError, "signature mismatch"):
                validate_research_clip_cache(cache, second)

    def test_isolated_run_directories_refuse_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = create_run_directories(directory, "tiny_run")
            self.assertEqual(
                set(paths), {"checkpoints", "logs", "metrics", "predictions", "registries"}
            )
            with self.assertRaisesRegex(FileExistsError, "Refusing to overwrite"):
                create_run_directories(directory, "tiny_run")

    def test_best_checkpoint_npr_four_degradations_and_registry(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self._tiny_manifest(root)
            config = root / "config.yaml"
            config.write_text("fold: holdout_adm\n", encoding="utf-8")
            checkpoint_path = root / "best_model.pt"
            best = ConstantNpr(1.25)
            torch.save(
                {"model_state": best.state_dict(), "epoch": 3, "best_score": 0.8},
                checkpoint_path,
            )
            loaded = ConstantNpr(-4.0)
            checkpoint = load_best_checkpoint(checkpoint_path, loaded, torch.device("cpu"))
            self.assertEqual(checkpoint["epoch"], 3)
            self.assertEqual(float(loaded.bias.detach()), 1.25)
            predictions = {
                degradation: predict_npr(
                    loaded,
                    manifest,
                    "holdout_adm",
                    degradation,
                    torch.device("cpu"),
                    batch_size=2,
                )
                for degradation in DEGRADATIONS
            }
            prediction_dir = root / "predictions"
            prediction_dir.mkdir()
            records, order_hash = save_degradation_predictions(
                predictions, prediction_dir
            )
            self.assertEqual(set(records), set(DEGRADATIONS))
            for frame in predictions.values():
                self.assertEqual(
                    frame.columns.tolist(),
                    ["sample_id", "fold", "label", "probability"],
                )
                self.assertEqual(frame["fold"].unique().tolist(), ["holdout_adm"])
                self.assertEqual(frame["sample_id"].tolist(), ["tiny-real", "tiny-fake"])
            registry_path = root / "registry.json"
            registry = write_run_registry(
                registry_path,
                branch="research/cross-dataset-robustness-v3",
                commit="tiny",
                run_id="tiny",
                fold="holdout_adm",
                config_path=config,
                split_paths={"train": manifest, "validation": manifest},
                checkpoint_path=checkpoint_path,
                prediction_records=records,
                sample_order_sha256=order_hash,
            )
            self.assertEqual(registry["fold"], "holdout_adm")
            self.assertEqual(
                registry["sample_order_sha256"],
                hashlib.sha256(b"tiny-real\ntiny-fake").hexdigest(),
            )
            self.assertEqual(registry["checkpoint"]["sha256"], sha256_file(checkpoint_path))

    def test_b2_four_degradations_produce_strict_oof_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = self._tiny_manifest(Path(directory))
            preprocess = transforms.Compose(
                [transforms.Resize((16, 16)), transforms.ToTensor()]
            )
            encoder = TinyEncoder()
            head = TinyHead()
            frames = [
                predict_b2(
                    encoder,
                    head,
                    preprocess,
                    manifest,
                    "holdout_adm",
                    degradation,
                    torch.device("cpu"),
                    batch_size=2,
                )
                for degradation in DEGRADATIONS
            ]
            for frame in frames:
                self.assertEqual(
                    frame.columns.tolist(),
                    ["sample_id", "fold", "label", "probability"],
                )
                self.assertEqual(frame["sample_id"].tolist(), ["tiny-real", "tiny-fake"])
                self.assertTrue(frame["probability"].between(0, 1).all())

    def test_six_pilot_configs_match_frozen_contract_and_split_hashes(self):
        contract = yaml.safe_load(Path("configs/research_v3/s1_contract.yaml").read_text())
        audit = json.loads(Path("outputs/research_v3/audits/audit_report.json").read_text())
        expected_hashes = {
            record["path"]: record["sha256"] for record in audit["manifests"]
        }
        configs = sorted(
            list(Path("configs/research_v3").glob("pilot_b2_*_seed42_v3.yaml"))
            + list(Path("configs/research_v3").glob("pilot_npr_*_seed42_v3.yaml"))
        )
        self.assertEqual(len(configs), 6)
        for path in configs:
            config = yaml.safe_load(path.read_text())
            self.assertTrue(config["research_v3"])
            self.assertEqual(config["seed"], 42)
            self.assertEqual(config["epochs"], 10)
            self.assertEqual(config["batch_size"], 32)
            self.assertEqual(config["num_workers"], 0)
            self.assertEqual(config["output_root"], "outputs/research_v3")
            for key in ("train_csv", "val_csv"):
                digest = hashlib.sha256(Path(config[key]).read_bytes()).hexdigest()
                self.assertEqual(digest, expected_hashes[config[key]])
            if config["model_type"] == "clip_mlp":
                frozen = contract["b2_control"]
                self.assertEqual(config["learning_rate"], frozen["learning_rate"])
                self.assertEqual(config["weight_decay"], frozen["weight_decay"])
                self.assertEqual(config["dropout"], frozen["dropout"])
            else:
                frozen = contract["npr_expert"]
                self.assertEqual(config["learning_rate"], frozen["learning_rate"])
                self.assertEqual(config["weight_decay"], frozen["weight_decay"])
                self.assertEqual(
                    config["samples_per_group_per_epoch"],
                    contract["sampler"]["samples_per_group_per_epoch"],
                )


if __name__ == "__main__":
    unittest.main()
