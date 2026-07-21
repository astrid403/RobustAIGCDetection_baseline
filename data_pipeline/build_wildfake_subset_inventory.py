import argparse
import hashlib
import os
from pathlib import Path
import sys

import pandas as pd
from PIL import Image
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
INVENTORY_COLUMNS = [
    "sample_id",
    "storage_backend",
    "path",
    "label",
    "dataset",
    "split",
    "generator",
    "source",
    "width",
    "height",
    "format",
    "is_valid",
    "dataset_id",
    "dataset_revision",
    "hf_split",
    "row_index",
    "label_b",
    "caption",
]


def iter_images(root):
    root = Path(root)
    return sorted(
        Path(dirpath) / filename
        for dirpath, _, filenames in os.walk(root, followlinks=True)
        for filename in filenames
        if Path(filename).suffix.lower() in IMAGE_EXTENSIONS
    )


def image_info(path):
    try:
        with Image.open(path) as img:
            width, height = img.size
            fmt = img.format or "unknown"
            img.verify()
        return width, height, fmt, True
    except Exception:
        return None, None, "unknown", False


def add_rows(rows, root, label, generator, source):
    for path in tqdm(iter_images(root), desc=f"Scanning {source}"):
        width, height, fmt, is_valid = image_info(path)
        rows.append(
            {
                "sample_id": "wildfake-" + hashlib.sha256(str(path.relative_to(PROJECT_ROOT)).encode("utf-8")).hexdigest()[:24],
                "storage_backend": "local",
                "path": str(path.relative_to(PROJECT_ROOT)),
                "label": label,
                "dataset": "WildFake_subset_CelebAHQ_DDIM",
                "split": "external",
                "generator": generator,
                "source": source,
                "width": width,
                "height": height,
                "format": fmt,
                "is_valid": is_valid,
                "dataset_id": "WildFake_subset_CelebAHQ_DDIM",
                "dataset_revision": "local",
                "hf_split": "",
                "row_index": "",
                "label_b": "",
                "caption": "",
            }
        )


def sample_balanced(df, n_per_class, seed):
    groups = []
    for label, group in df.groupby("label"):
        groups.append(group.sample(n=min(n_per_class, len(group)), random_state=seed))
    return pd.concat(groups).sample(frac=1.0, random_state=seed).reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description="Build inventory and splits for the CelebA-HQ/DDIM WildFake subset.")
    parser.add_argument("--root", default="data/raw/WildFake/subset_celebahq_ddim")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    root = PROJECT_ROOT / args.root
    real_root = root / "real" / "celebahq"
    fake_root = root / "fake" / "DDIM"
    if not real_root.exists() or not fake_root.exists():
        raise FileNotFoundError(f"Expected subset layout under {root}")

    rows = []
    add_rows(rows, real_root, label=0, generator="real", source="CelebA-HQ")
    add_rows(rows, fake_root, label=1, generator="DDIM", source="DDIM")
    df = pd.DataFrame(rows, columns=INVENTORY_COLUMNS)

    inventory_dir = PROJECT_ROOT / "outputs" / "inventories"
    split_dir = PROJECT_ROOT / "outputs" / "splits"
    inventory_dir.mkdir(parents=True, exist_ok=True)
    split_dir.mkdir(parents=True, exist_ok=True)

    inventory_path = inventory_dir / "wildfake_inventory.csv"
    df.to_csv(inventory_path, index=False)
    print(f"Saved {len(df)} rows to {inventory_path}")
    print(df.groupby(["label", "source", "generator", "is_valid"]).size().to_string())

    valid = df[df["is_valid"].astype(bool) & df["label"].isin([0, 1])]
    for name, n in [("debug", 100), ("baseline", 2000), ("final", 5000)]:
        split = sample_balanced(valid, n, args.seed)
        out_path = split_dir / f"wildfake_subset_{name}_test.csv"
        split.to_csv(out_path, index=False)
        print(f"Saved {len(split)} rows to {out_path}")


if __name__ == "__main__":
    main()
