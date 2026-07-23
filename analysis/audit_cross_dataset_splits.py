"""Audit GenImage train/validation and propose source-aware LOGO manifests.

This tool is deliberately limited to the two development manifests. It refuses
paths containing held-out/external scope names so it cannot silently expand the
research-v3 development boundary.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
GENERATORS = ("ADM", "BigGAN", "Stable Diffusion V1.5")
FORBIDDEN_SCOPE_TOKENS = ("unseen", "defactify")
OUTPUT_COLUMNS = (
    "sample_id",
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
    "source_group",
    "content_sha256",
    "dhash64",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_sample_id(path: str) -> str:
    return "genimage-v3-" + hashlib.sha256(path.encode("utf-8")).hexdigest()[:24]


def source_group(row: pd.Series) -> str:
    path = Path(str(row["path"]))
    if int(row["label"]) == 0:
        # GenImage real images originate from ImageNet-style files replicated
        # under generator directories. The basename is the available source ID.
        return "real:" + path.name.lower()
    return "fake:" + str(row["generator"]) + ":" + path.name.lower()


def dhash64(path: Path) -> str:
    with Image.open(path) as image:
        gray = image.convert("L").resize((9, 8), Image.Resampling.BILINEAR)
        pixels = list(gray.getdata())
    value = 0
    for y in range(8):
        offset = y * 9
        for x in range(8):
            value = (value << 1) | int(pixels[offset + x] > pixels[offset + x + 1])
    return f"{value:016x}"


def hamming_hex(left: str, right: str) -> int:
    return (int(left, 16) ^ int(right, 16)).bit_count()


def load_development_manifest(path: Path, role: str, compute_hashes: bool = True) -> pd.DataFrame:
    lowered = str(path).lower()
    if any(token in lowered for token in FORBIDDEN_SCOPE_TOKENS):
        raise ValueError(f"Forbidden research-v3 development manifest: {path}")
    frame = pd.read_csv(path)
    required = {
        "path", "label", "dataset", "split", "generator", "source",
        "width", "height", "format", "is_valid",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    if set(frame["generator"]) != set(GENERATORS):
        raise ValueError(f"Unexpected generators in {path}: {sorted(frame['generator'].unique())}")
    if set(frame["label"].astype(int)) != {0, 1}:
        raise ValueError(f"Both binary labels are required in {path}")
    frame = frame.copy()
    frame["manifest_role"] = role
    frame["sample_id"] = frame["path"].astype(str).map(stable_sample_id)
    frame["source_group"] = frame.apply(source_group, axis=1)
    if compute_hashes:
        resolved = [
            item if item.is_absolute() else PROJECT_ROOT / item
            for item in frame["path"].astype(str).map(Path)
        ]
        missing_files = [str(item) for item in resolved if not item.is_file()]
        if missing_files:
            raise FileNotFoundError(f"Missing development images: {missing_files[:3]}")
        frame["content_sha256"] = [sha256_file(item) for item in resolved]
        frame["dhash64"] = [dhash64(item) for item in resolved]
    return frame


def cross_near_duplicate_train_ids(
    train: pd.DataFrame, validation: pd.DataFrame, max_hamming: int
) -> set[str]:
    """Return train IDs perceptually near any validation image of the same label."""
    blocked: set[str] = set()
    for label in (0, 1):
        train_rows = train[train["label"].astype(int) == label]
        val_hashes = validation[validation["label"].astype(int) == label]["dhash64"].tolist()
        for row in train_rows.itertuples():
            if any(hamming_hex(row.dhash64, candidate) <= max_hamming for candidate in val_hashes):
                blocked.add(row.sample_id)
    return blocked


def build_logo_fold(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    held_out_generator: str,
    max_hamming: int = 4,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    if held_out_generator not in GENERATORS:
        raise ValueError(f"Unknown held-out generator: {held_out_generator}")
    fold_train = train[train["generator"] != held_out_generator].copy()
    fold_val = validation[validation["generator"] == held_out_generator].copy()

    val_sources = set(fold_val["source_group"])
    val_content = set(fold_val["content_sha256"])
    source_blocked = set(fold_train.loc[fold_train["source_group"].isin(val_sources), "sample_id"])
    content_blocked = set(fold_train.loc[fold_train["content_sha256"].isin(val_content), "sample_id"])
    near_blocked = cross_near_duplicate_train_ids(fold_train, fold_val, max_hamming)
    blocked = source_blocked | content_blocked | near_blocked
    fold_train = fold_train[~fold_train["sample_id"].isin(blocked)].copy()

    overlap = set(fold_train["sample_id"]) & set(fold_val["sample_id"])
    source_overlap = set(fold_train["source_group"]) & set(fold_val["source_group"])
    content_overlap = set(fold_train["content_sha256"]) & set(fold_val["content_sha256"])
    if overlap or source_overlap or content_overlap:
        raise AssertionError("LOGO leakage remained after source/content filtering")
    if set(fold_train["label"].astype(int)) != {0, 1} or set(fold_val["label"].astype(int)) != {0, 1}:
        raise ValueError("LOGO fold lost a class")

    details = {
        "held_out_generator": held_out_generator,
        "max_dhash_hamming": max_hamming,
        "train_rows_before_filter": int((train["generator"] != held_out_generator).sum()),
        "train_rows_after_filter": len(fold_train),
        "validation_rows": len(fold_val),
        "blocked_source_group_rows": len(source_blocked),
        "blocked_exact_content_rows": len(content_blocked),
        "blocked_near_duplicate_rows": len(near_blocked),
        "blocked_unique_rows": len(blocked),
    }
    return fold_train, fold_val, details


def distribution(frame: pd.DataFrame, fold: str, role: str) -> pd.DataFrame:
    columns = ["generator", "label", "format", "width", "height"]
    out = frame.groupby(columns, dropna=False).size().reset_index(name="count")
    out.insert(0, "role", role)
    out.insert(0, "fold", fold)
    return out


def write_csv(frame: pd.DataFrame, path: Path) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.loc[:, OUTPUT_COLUMNS].to_csv(path, index=False)
    return {"path": path.as_posix(), "rows": len(frame), "sha256": sha256_file(path)}


def run_audit(train_csv: Path, val_csv: Path, output_dir: Path, max_hamming: int = 4) -> dict:
    train = load_development_manifest(train_csv, "train")
    validation = load_development_manifest(val_csv, "validation")
    if set(train["sample_id"]) & set(validation["sample_id"]):
        raise ValueError("Original train/validation manifests overlap by sample_id")

    manifests = []
    fold_details = []
    distributions = []
    for held_out in GENERATORS:
        slug = held_out.lower().replace(" ", "_").replace(".", "")
        fold_name = f"holdout_{slug}"
        fold_train, fold_val, details = build_logo_fold(
            train, validation, held_out, max_hamming=max_hamming
        )
        manifests.append(write_csv(fold_train, output_dir / f"{fold_name}_train.csv"))
        manifests.append(write_csv(fold_val, output_dir / f"{fold_name}_validation.csv"))
        distributions.extend(
            [distribution(fold_train, fold_name, "train"), distribution(fold_val, fold_name, "validation")]
        )
        fold_details.append(details)

    distribution_frame = pd.concat(distributions, ignore_index=True)
    distribution_path = output_dir / "logo_distribution.csv"
    distribution_frame.to_csv(distribution_path, index=False)
    report = {
        "schema_version": 1,
        "scope": "GenImage train/validation only",
        "forbidden_scopes_accessed": [],
        "inputs": {
            "train": {"path": train_csv.as_posix(), "rows": len(train), "sha256": sha256_file(train_csv)},
            "validation": {
                "path": val_csv.as_posix(),
                "rows": len(validation),
                "sha256": sha256_file(val_csv),
            },
        },
        "generators": list(GENERATORS),
        "original_sample_id_overlap": 0,
        "original_exact_content_overlap": len(
            set(train["content_sha256"]) & set(validation["content_sha256"])
        ),
        "original_source_group_overlap": len(
            set(train["source_group"]) & set(validation["source_group"])
        ),
        "folds": fold_details,
        "manifests": manifests,
        "distribution": {
            "path": distribution_path.as_posix(),
            "rows": len(distribution_frame),
            "sha256": sha256_file(distribution_path),
        },
    }
    report_path = output_dir / "audit_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-csv", default="outputs/splits/genimage_splitB_available_train.csv"
    )
    parser.add_argument(
        "--val-csv", default="outputs/splits/genimage_splitB_available_val.csv"
    )
    parser.add_argument("--output-dir", default="outputs/research_v3/audits")
    parser.add_argument("--max-dhash-hamming", type=int, default=4)
    args = parser.parse_args()
    report = run_audit(
        Path(args.train_csv),
        Path(args.val_csv),
        Path(args.output_dir),
        max_hamming=args.max_dhash_hamming,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
