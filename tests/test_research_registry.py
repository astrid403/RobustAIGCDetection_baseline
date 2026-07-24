import json
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from evaluation.evaluate_cross_dataset import sha256_file, write_oof_outputs


class ResearchRegistryTests(unittest.TestCase):
    def test_toy_registry_hashes_every_input_and_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ids = ["a0", "a1", "b0", "b1", "c0", "c1"]
            folds = ["a", "a", "b", "b", "c", "c"]
            labels = [0, 1, 0, 1, 0, 1]
            clip = pd.DataFrame({
                "sample_id": ids, "fold": folds, "label": labels,
                "probability": [0.1, 0.9, 0.2, 0.8, 0.3, 0.7],
            })
            npr = clip.copy()
            npr["probability"] = [0.2, 0.8, 0.1, 0.9, 0.4, 0.6]
            clip_path, npr_path = root / "clip.csv", root / "npr.csv"
            clip.to_csv(clip_path, index=False)
            npr.to_csv(npr_path, index=False)
            config = root / "config.yaml"
            config.write_text("alpha: 0.5\n", encoding="utf-8")
            splits, checkpoints = {}, {}
            for fold in ("a", "b", "c"):
                split = root / f"split_{fold}.csv"
                split.write_text(f"fold\n{fold}\n", encoding="utf-8")
                checkpoint = root / f"checkpoint_{fold}.pt"
                checkpoint.write_bytes(f"toy-{fold}".encode())
                splits[fold] = split
                checkpoints[fold] = checkpoint
            output = root / "new_output"
            registry = write_oof_outputs(
                clip_path, npr_path, output,
                config_path=config,
                split_paths=splits,
                checkpoint_paths=checkpoints,
                branch="research/cross-dataset-robustness-v3",
                commit="toy-commit",
                expected_folds=("a", "b", "c"),
            )
            self.assertEqual(registry["sample_count"], 6)
            self.assertEqual(registry["primary_alpha_clip"], 0.5)
            self.assertFalse(registry["genimage_unseen_accessed"])
            self.assertFalse(registry["defactify_accessed"])
            for group in ("splits", "checkpoints", "input_predictions", "outputs"):
                for record in registry[group].values():
                    self.assertEqual(record["sha256"], sha256_file(record["path"]))
            persisted = json.loads((output / "registry.json").read_text())
            self.assertEqual(persisted, registry)
            with self.assertRaisesRegex(FileExistsError, "Refusing to overwrite"):
                write_oof_outputs(
                    clip_path, npr_path, output,
                    config_path=config, split_paths=splits,
                    checkpoint_paths=checkpoints, branch="x", commit="y",
                )


if __name__ == "__main__":
    unittest.main()
