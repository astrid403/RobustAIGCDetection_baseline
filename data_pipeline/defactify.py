"""Build lightweight manifests for the pinned Defactify dataset."""

import argparse
import hashlib
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_pipeline.dataset_factory import DEFACTIFY_DATASET_ID


DEFAULT_REVISION = "787334f7857fa54f29027a7f09c30e895ad486ef"
LABEL_B_GENERATORS = {
    0: "real",
    1: "Stable Diffusion 2.1",
    2: "Stable Diffusion XL",
    3: "Stable Diffusion 3",
    4: "DALL-E 3",
    5: "Midjourney 6",
}
OFFICIAL_SPLITS = ("train", "validation", "test")
INVENTORY_COLUMNS = [
    "sample_id", "storage_backend", "path", "dataset", "dataset_id",
    "dataset_revision", "hf_split", "row_index", "label", "label_b",
    "generator", "source", "caption", "width", "height", "format",
    "is_valid", "split",
]


def stable_sample_id(dataset_id, revision, split, row_index):
    value = f"{dataset_id}@{revision}:{split}:{row_index}"
    return "defactify-" + hashlib.sha256(value.encode("utf-8")).hexdigest()[:24]


def validate_labels(label_a, label_b):
    if label_a not in (0, 1) or label_b not in LABEL_B_GENERATORS:
        raise ValueError(f"Unexpected Defactify labels: Label_A={label_a}, Label_B={label_b}")
    if (label_b == 0) != (label_a == 0):
        raise ValueError(f"Inconsistent Defactify labels: Label_A={label_a}, Label_B={label_b}")


def load_metadata_split(dataset_id, revision, split, cache_dir=None):
    try:
        from datasets import Image, load_dataset
    except ImportError as exc:
        raise ImportError("Install datasets and pyarrow before preparing Defactify.") from exc
    dataset = load_dataset(dataset_id, split=split, revision=revision, cache_dir=cache_dir)
    return dataset.cast_column("Image", Image(decode=False))


def rows_for_split(dataset, dataset_id, revision, split):
    rows = []
    for row_index, record in enumerate(dataset):
        label_a = int(record["Label_A"])
        label_b = int(record["Label_B"])
        validate_labels(label_a, label_b)
        sample_id = stable_sample_id(dataset_id, revision, split, row_index)
        rows.append({
            "sample_id": sample_id,
            "storage_backend": "huggingface",
            "path": f"hf://{dataset_id}@{revision}/{split}/{row_index}",
            "dataset": "Defactify",
            "dataset_id": dataset_id,
            "dataset_revision": revision,
            "hf_split": split,
            "row_index": row_index,
            "label": label_a,
            "label_b": label_b,
            "generator": LABEL_B_GENERATORS[label_b],
            "source": "MS COCO" if label_b == 0 else LABEL_B_GENERATORS[label_b],
            "caption": record.get("Caption", ""),
            "width": None,
            "height": None,
            "format": "huggingface_image",
            "is_valid": True,
            "split": split,
        })
    return rows


def balanced_test(df, seed=42):
    test = df[df["hf_split"] == "test"]
    real = test[test["label_b"] == 0]
    per_generator = len(real) // 5
    fake = pd.concat([
        test[test["label_b"] == label_b].sample(
            n=min(per_generator, len(test[test["label_b"] == label_b])), random_state=seed
        )
        for label_b in range(1, 6)
    ])
    return pd.concat([real, fake]).sample(frac=1, random_state=seed).reset_index(drop=True)


def smoke_test(df, per_label_per_split=2, seed=42):
    groups = []
    for (_, _), group in df.groupby(["hf_split", "label_b"]):
        groups.append(group.sample(n=min(per_label_per_split, len(group)), random_state=seed))
    return pd.concat(groups).sample(frac=1, random_state=seed).reset_index(drop=True)


def build_manifests(dataset_id, revision, out_inventory, out_dir, cache_dir=None, seed=42):
    if not revision or revision == "main":
        raise ValueError("Defactify must use a pinned commit revision, not an empty value or 'main'.")
    rows = []
    for split in OFFICIAL_SPLITS:
        dataset = load_metadata_split(dataset_id, revision, split, cache_dir)
        rows.extend(rows_for_split(dataset, dataset_id, revision, split))
    inventory = pd.DataFrame(rows, columns=INVENTORY_COLUMNS)
    if inventory["sample_id"].duplicated().any():
        raise ValueError("Defactify sample_id values are not unique.")
    out_inventory = Path(out_inventory)
    out_dir = Path(out_dir)
    out_inventory.parent.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    inventory.to_csv(out_inventory, index=False)
    for split in OFFICIAL_SPLITS:
        inventory[inventory["hf_split"] == split].to_csv(out_dir / f"defactify_official_{split}.csv", index=False)
    balanced_test(inventory, seed).to_csv(out_dir / "defactify_balanced_test.csv", index=False)
    smoke_test(inventory, 2, seed).to_csv(out_dir / "defactify_smoke_test.csv", index=False)
    print(inventory.groupby(["hf_split", "label_b", "generator"]).size().to_string())


def main():
    parser = argparse.ArgumentParser(description="Build Defactify manifests without exporting images.")
    parser.add_argument("--dataset-id", default=DEFACTIFY_DATASET_ID)
    parser.add_argument("--revision", default=DEFAULT_REVISION, help="Pinned Hugging Face commit SHA.")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--inventory", default="outputs/inventories/defactify_inventory.csv")
    parser.add_argument("--out-dir", default="outputs/splits")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    build_manifests(args.dataset_id, args.revision, args.inventory, args.out_dir, args.cache_dir, args.seed)


if __name__ == "__main__":
    main()
