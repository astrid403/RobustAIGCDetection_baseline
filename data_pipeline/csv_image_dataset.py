from pathlib import Path

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class CsvImageDataset(Dataset):
    """Load images and binary labels from an inventory or split CSV."""

    def __init__(self, csv_path, transform=None, max_samples_per_class=None, root_dir="."):
        self.csv_path = Path(csv_path)
        if not self.csv_path.exists():
            raise FileNotFoundError(
                f"CSV file not found: {self.csv_path}\n"
                "Build inventories and splits first, for example:\n"
                "python data_pipeline/build_inventory.py --dataset CIFAKE\n"
                "python data_pipeline/make_splits.py --config configs/debug_resnet18.yaml"
            )
        self.root_dir = Path(root_dir)
        self.transform = transform
        df = pd.read_csv(self.csv_path)
        if "is_valid" in df.columns:
            df = df[df["is_valid"].astype(str).str.lower().isin(["true", "1", "yes"])]
        if max_samples_per_class:
            df = (
                df.groupby("label", group_keys=False)
                .apply(lambda x: x.sample(min(len(x), max_samples_per_class), random_state=42))
                .reset_index(drop=True)
            )
        if df.empty:
            raise ValueError(f"No valid rows found in {self.csv_path}.")
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = Path(row["path"])
        if not path.is_absolute():
            path = self.root_dir / path
        try:
            image = Image.open(path).convert("RGB")
        except Exception as exc:
            raise RuntimeError(f"Could not open image: {path}") from exc
        if self.transform:
            image = self.transform(image)
        sample_id = row.get("sample_id", row["path"])
        return image, float(row["label"]), str(sample_id)


def read_csv_metadata(csv_path):
    df = pd.read_csv(csv_path)
    required = {"path", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path} is missing required columns: {sorted(missing)}")
    return df
