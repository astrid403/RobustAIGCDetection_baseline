import argparse
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
]


def infer_label(parts):
    lowered = [p.lower() for p in parts]
    if any(p in {"fake", "ai", "generated", "synthetic"} for p in lowered):
        return 1
    if any(p in {"real", "nature", "human", "authentic"} for p in lowered):
        return 0
    return None


def infer_split(parts):
    lowered = [p.lower() for p in parts]
    for split in ["train", "val", "test"]:
        if split in lowered:
            return split
    return "unknown"


def infer_generator(dataset, rel_parts):
    if dataset == "GenImage" and rel_parts:
        return rel_parts[0]
    return "unknown"


def scan_folder(dataset, dataset_root):
    rows = []
    dataset_root = Path(dataset_root)
    if not dataset_root.exists():
        print(f"{dataset} folder not found at {dataset_root}.")
        return pd.DataFrame(columns=INVENTORY_COLUMNS)

    image_paths = [
        Path(dirpath) / filename
        for dirpath, _, filenames in os.walk(dataset_root, followlinks=True)
        for filename in filenames
        if Path(filename).suffix.lower() in IMAGE_EXTENSIONS
    ]
    for path in tqdm(image_paths, desc=f"Scanning {dataset}"):
        rel_to_dataset = path.relative_to(dataset_root)
        rel_to_project = path.relative_to(PROJECT_ROOT) if path.is_relative_to(PROJECT_ROOT) else path
        parts = rel_to_dataset.parts
        label = infer_label(parts)
        width = height = None
        fmt = "unknown"
        is_valid = True
        try:
            with Image.open(path) as img:
                width, height = img.size
                fmt = img.format or "unknown"
                img.verify()
        except Exception:
            is_valid = False
        rows.append(
            {
                "path": str(rel_to_project),
                "label": label if label is not None else -1,
                "dataset": dataset,
                "split": infer_split(parts),
                "generator": infer_generator(dataset, parts),
                "source": parts[0] if parts else "unknown",
                "width": width,
                "height": height,
                "format": fmt,
                "is_valid": is_valid,
            }
        )
    return pd.DataFrame(rows, columns=INVENTORY_COLUMNS)


def scan_wildfake_metadata(metadata_csv):
    df = pd.read_csv(metadata_csv)
    if "path" not in df.columns or "label" not in df.columns:
        raise ValueError("WildFake metadata CSV must include path,label columns.")
    for col, default in {
        "dataset": "WildFake",
        "split": "external",
        "generator": "unknown",
        "source": "metadata_csv",
        "width": None,
        "height": None,
        "format": "unknown",
        "is_valid": True,
    }.items():
        if col not in df.columns:
            df[col] = default
    return df[INVENTORY_COLUMNS]


def save_inventory(df, dataset, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = dataset.lower()
    out_path = out_dir / f"{name}_inventory.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} rows to {out_path}")
    if df.empty:
        return out_path
    summary = df.groupby(["dataset", "split", "generator", "label", "is_valid"]).size()
    print(summary.to_string())
    return out_path


def build(dataset, root, out_dir, wildfake_metadata=None):
    root = Path(root)
    if dataset == "all":
        for item in ["CIFAKE", "GenImage", "WildFake"]:
            build(item, root, out_dir, wildfake_metadata)
        return
    if dataset == "WildFake" and wildfake_metadata:
        df = scan_wildfake_metadata(wildfake_metadata)
    else:
        df = scan_folder(dataset, root / dataset)
    save_inventory(df, dataset, out_dir)


def main():
    parser = argparse.ArgumentParser(description="Build CSV inventories for image datasets.")
    parser.add_argument("--dataset", choices=["CIFAKE", "GenImage", "WildFake", "all"], default="all")
    parser.add_argument("--root", default="data/raw")
    parser.add_argument("--out-dir", default="outputs/inventories")
    parser.add_argument("--wildfake-metadata", default=None)
    args = parser.parse_args()
    build(args.dataset, args.root, args.out_dir, args.wildfake_metadata)


if __name__ == "__main__":
    main()
