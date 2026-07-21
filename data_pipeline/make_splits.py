import argparse
from pathlib import Path
import sys

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def read_inventory(name):
    path = PROJECT_ROOT / "outputs" / "inventories" / f"{name.lower()}_inventory.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Inventory not found: {path}\n"
            f"Run: python data_pipeline/build_inventory.py --dataset {name}"
        )
    df = pd.read_csv(path)
    if "is_valid" in df.columns:
        df = df[df["is_valid"].astype(str).str.lower().isin(["true", "1", "yes"])]
    df = df[df["label"].isin([0, 1])]
    return df


def sample_balanced(df, n_per_class=None, seed=42):
    groups = []
    for label, group in df.groupby("label"):
        n = len(group) if n_per_class is None else min(len(group), n_per_class)
        groups.append(group.sample(n=n, random_state=seed))
    return pd.concat(groups).sample(frac=1.0, random_state=seed).reset_index(drop=True) if groups else df


def write_split(df, name):
    out = PROJECT_ROOT / "outputs" / "splits" / name
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Saved {len(df)} rows to {out}")


def make_cifake(seed=42, max_samples_per_class=None):
    df = read_inventory("CIFAKE")
    train = sample_balanced(df[df["split"] == "train"], max_samples_per_class, seed)
    test = sample_balanced(df[df["split"] == "test"], max_samples_per_class, seed)
    write_split(train, "cifake_full_train.csv")
    write_split(test, "cifake_full_test.csv")


def check_generators(df, requested):
    available = sorted(g for g in df["generator"].dropna().unique() if g != "unknown")
    missing = [g for g in requested if g not in available]
    if missing:
        raise ValueError(
            "GenImage requested generators are missing.\n"
            f"Available generators: {', '.join(available) or '(none)'}\n"
            f"Requested generators: {', '.join(requested)}\n"
            f"Missing generators: {', '.join(missing)}"
        )


def genimage_generator_subset(df, generators, splits, n_per_class, seed):
    check_generators(df, generators)
    out = df[df["generator"].isin(generators) & df["split"].isin(splits)]
    groups = []
    for gen in generators:
        gen_df = out[out["generator"] == gen]
        groups.append(sample_balanced(gen_df, n_per_class, seed))
    return pd.concat(groups).sample(frac=1.0, random_state=seed).reset_index(drop=True) if groups else out


def make_genimage(config):
    seed = config.get("seed", 42)
    n = config.get("samples_per_generator_per_class") or config.get("max_samples_per_class")
    df = read_inventory("GenImage")
    split_cfg = config.get("genimage_splits", {})
    split_a_train = split_cfg.get("splitA_train_generators", ["Stable Diffusion V1.4"])
    split_a_test = split_cfg.get(
        "splitA_test_generators",
        ["ADM", "GLIDE", "BigGAN", "Midjourney", "Wukong", "VQDM", "Stable Diffusion V1.5"],
    )
    split_b_train = split_cfg.get("splitB_train_generators", ["Stable Diffusion V1.4", "ADM", "BigGAN"])
    split_b_test = split_cfg.get("splitB_test_generators", ["GLIDE", "Midjourney", "Wukong", "VQDM"])

    write_split(genimage_generator_subset(df, split_a_train, ["train"], n, seed), "genimage_splitA_train.csv")
    write_split(genimage_generator_subset(df, split_a_train, ["val", "test"], n, seed), "genimage_splitA_val.csv")
    write_split(genimage_generator_subset(df, split_a_test, ["val", "test"], n, seed), "genimage_splitA_test_unseen.csv")
    write_split(genimage_generator_subset(df, split_b_train, ["train"], n, seed), "genimage_splitB_train.csv")
    write_split(genimage_generator_subset(df, split_b_train, ["val", "test"], n, seed), "genimage_splitB_val.csv")
    write_split(genimage_generator_subset(df, split_b_train, ["val", "test"], n, seed), "genimage_splitB_test_seen.csv")
    write_split(genimage_generator_subset(df, split_b_test, ["val", "test"], n, seed), "genimage_splitB_test_unseen.csv")


def make_wildfake(seed=42):
    try:
        df = read_inventory("WildFake")
    except FileNotFoundError:
        print(
            "WildFake is not available.\n"
            "Skipping WildFake split generation.\n"
            "Add WildFake under data/raw/WildFake or provide a metadata CSV later."
        )
        return
    for name, n in [("debug", 100), ("baseline", 2000), ("final", 5000)]:
        write_split(sample_balanced(df, n, seed), f"wildfake_{name}_test.csv")


def main():
    parser = argparse.ArgumentParser(description="Generate reproducible split CSV files.")
    parser.add_argument("--config", default="configs/debug_resnet18.yaml")
    parser.add_argument("--dataset", choices=["CIFAKE", "GenImage", "WildFake", "all"], default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    dataset = args.dataset or config.get("dataset", "CIFAKE")
    if dataset in ["CIFAKE", "all"]:
        make_cifake(config.get("seed", 42), config.get("max_samples_per_class"))
    if dataset in ["GenImage", "all"]:
        make_genimage(config)
    if dataset in ["WildFake", "all"]:
        make_wildfake(config.get("seed", 42))


if __name__ == "__main__":
    main()
