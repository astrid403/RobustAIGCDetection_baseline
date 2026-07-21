"""Backend-aware image dataset factory.

Local CSV manifests remain fully backwards compatible. Hugging Face rows are
resolved lazily from the native cache and are never exported as another image
tree.
"""

from pathlib import Path

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

from data_pipeline.csv_image_dataset import CsvImageDataset


DEFACTIFY_DATASET_ID = "Rajarshi-Roy-research/Defactify_Image_Dataset"


def _valid_rows(df):
    if "is_valid" in df.columns:
        df = df[df["is_valid"].astype(str).str.lower().isin(["true", "1", "yes"])]
    return df.reset_index(drop=True)


class HuggingFaceImageDataset(Dataset):
    """Resolve manifest rows against a pinned Hugging Face dataset revision."""

    def __init__(self, csv_path, transform=None, max_samples_per_class=None, cache_dir=None):
        self.csv_path = Path(csv_path)
        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {self.csv_path}")
        self.transform = transform
        self.cache_dir = cache_dir
        df = _valid_rows(pd.read_csv(self.csv_path))
        required = {"row_index", "hf_split", "label"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"{csv_path} is missing Hugging Face columns: {sorted(missing)}")
        if max_samples_per_class:
            df = pd.concat(
                [
                    group.sample(n=min(len(group), max_samples_per_class), random_state=42)
                    for _, group in df.groupby("label")
                ],
                ignore_index=True,
            )
        if df.empty:
            raise ValueError(f"No valid rows found in {csv_path}.")
        self.df = df
        self._datasets = {}

    def __len__(self):
        return len(self.df)

    def _get_split(self, dataset_id, revision, split):
        key = (dataset_id, revision, split)
        if key not in self._datasets:
            try:
                from datasets import load_dataset
            except ImportError as exc:
                raise ImportError(
                    "Hugging Face datasets is required for this manifest. "
                    "Install the project requirements in aigc_det_baseline."
                ) from exc
            self._datasets[key] = load_dataset(
                dataset_id,
                split=split,
                revision=revision or None,
                cache_dir=self.cache_dir,
            )
        return self._datasets[key]

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        dataset_id = str(row.get("dataset_id", DEFACTIFY_DATASET_ID))
        revision = str(row.get("dataset_revision", ""))
        if revision.lower() == "nan":
            revision = ""
        split = str(row["hf_split"])
        row_index = int(row["row_index"])
        sample_id = str(row.get("sample_id", f"hf://{dataset_id}/{split}/{row_index}"))
        try:
            record = self._get_split(dataset_id, revision, split)[row_index]
            image = record["Image"]
            if not isinstance(image, Image.Image):
                raise TypeError(f"Image field returned {type(image).__name__}")
            image = image.convert("RGB")
        except Exception as exc:
            raise RuntimeError(
                f"Could not decode {sample_id} (split={split}, row_index={row_index})"
            ) from exc
        if self.transform:
            image = self.transform(image)
        return image, float(row["label"]), sample_id


def create_image_dataset(csv_path, transform=None, max_samples_per_class=None, root_dir=".", cache_dir=None):
    """Create the correct dataset implementation from a manifest."""
    header = pd.read_csv(csv_path, nrows=1)
    backend = str(header.iloc[0].get("storage_backend", "local")) if not header.empty else "local"
    if backend == "huggingface":
        return HuggingFaceImageDataset(csv_path, transform, max_samples_per_class, cache_dir)
    if backend != "local":
        raise ValueError(f"Unsupported storage_backend: {backend}")
    return CsvImageDataset(csv_path, transform, max_samples_per_class, root_dir)
