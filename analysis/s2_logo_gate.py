import argparse
import hashlib
import json
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.metrics import binary_metrics


FOLDS = ("holdout_adm", "holdout_biggan", "holdout_stable_diffusion_v15")
CONDITIONS = ("clean", "jpeg", "resize", "blur")


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_run(root, run_id, fold, require_amp):
    registry_path = root / "registries" / run_id / "run_registry.json"
    registry = json.loads(registry_path.read_text())
    if registry["run_id"] != run_id or registry["fold"] != fold:
        raise ValueError("Registry run/fold mismatch")
    if require_amp:
        amp = registry.get("amp", {})
        expected = {
            "requested": True,
            "effective": True,
            "autocast_dtype": "float16",
            "grad_scaler_enabled": True,
            "validation_autocast": True,
            "inference_autocast": True,
            "checkpoint_scaler_state": True,
        }
        if any(amp.get(key) != value for key, value in expected.items()):
            raise ValueError(f"Registry AMP mismatch: {amp}")
    for record in [
        registry["config"],
        *registry["splits"].values(),
        registry["checkpoint"],
        *registry["predictions"].values(),
        *registry.get("feature_caches", {}).values(),
        *registry.get("block_importance", {}).values(),
    ]:
        if sha256_file(record["path"]) != record["sha256"]:
            raise ValueError(f"Hash mismatch: {record['path']}")
    frames = {}
    reference = None
    for condition in CONDITIONS:
        frame = pd.read_csv(registry["predictions"][condition]["path"])
        if frame.columns.tolist() != ["sample_id", "fold", "label", "probability"]:
            raise ValueError("Prediction schema mismatch")
        identity = frame[["sample_id", "fold", "label"]].reset_index(drop=True)
        if identity["sample_id"].duplicated().any():
            raise ValueError("Duplicate sample ID")
        if set(identity["fold"]) != {fold}:
            raise ValueError("Prediction fold mismatch")
        if reference is None:
            reference = identity
        elif not identity.equals(reference):
            raise ValueError("Condition sample order mismatch")
        frames[condition] = frame
    order_hash = hashlib.sha256(
        "\n".join(reference["sample_id"]).encode("utf-8")
    ).hexdigest()
    if order_hash != registry["sample_order_sha256"]:
        raise ValueError("Sample-order hash mismatch")
    return registry_path, registry, frames


def build_gate(root):
    rows, registry_records = [], {}
    run_templates = {
        "b2": "pilot_b2_{fold}_seed42_v3",
        "s2": "pilot_s2_{fold}_seed42_amp_retry1_v3",
    }
    for model, template in run_templates.items():
        for fold in FOLDS:
            run_id = template.format(fold=fold)
            registry_path, registry, frames = validate_run(
                root, run_id, fold, require_amp=model == "s2"
            )
            registry_records[run_id] = {
                "path": str(registry_path),
                "sha256": sha256_file(registry_path),
                "checkpoint": registry["checkpoint"],
                "config": registry["config"],
                "sample_order_sha256": registry["sample_order_sha256"],
            }
            for condition, frame in frames.items():
                metrics = binary_metrics(frame["label"], frame["probability"])
                rows.append({
                    "model": model, "fold": fold, "condition": condition,
                    **{key: value for key, value in metrics.items()
                       if key != "confusion_matrix"},
                })
    metrics = pd.DataFrame(rows)
    clean = metrics[metrics["condition"] == "clean"]
    piv = clean.pivot(index="fold", columns="model", values="auroc")
    deltas = piv["s2"] - piv["b2"]
    clean_s2 = clean[clean["model"] == "s2"]
    degraded = metrics[metrics["condition"] != "clean"]
    degraded_means = degraded.groupby("model")["auroc"].mean()
    gate = {
        "clean_fold_mean_auroc_delta": float(deltas.mean()),
        "clean_worst_fold_auroc_delta": float(deltas.min()),
        "improved_clean_folds": int((deltas > 0).sum()),
        "degraded_mean_fold_auroc_delta": float(
            degraded_means["s2"] - degraded_means["b2"]
        ),
        "clean_real_recall": float(clean_s2["real_recall"].mean()),
        "clean_fake_recall": float(clean_s2["fake_recall"].mean()),
    }
    criteria = {
        "clean_mean": gate["clean_fold_mean_auroc_delta"] >= 0.005,
        "clean_worst": gate["clean_worst_fold_auroc_delta"] >= -0.002,
        "improved_folds": gate["improved_clean_folds"] >= 2,
        "degraded_mean": gate["degraded_mean_fold_auroc_delta"] >= 0.0,
        "real_recall": gate["clean_real_recall"] >= 0.70,
        "fake_recall": gate["clean_fake_recall"] >= 0.70,
    }
    gate["criteria"] = criteria
    gate["decision"] = "go" if all(criteria.values()) else "no_go"
    return metrics, gate, registry_records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="outputs/research_v3")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite output: {output}")
    metrics, gate, registries = build_gate(Path(args.root))
    output.mkdir(parents=True)
    metrics_path = output / "fold_condition_metrics.csv"
    metrics.to_csv(metrics_path, index=False)
    summary = {
        "schema": "f04_s2_amp_gate_v3",
        "gate": gate,
        "registries": registries,
        "metrics": {"path": str(metrics_path), "sha256": sha256_file(metrics_path)},
        "genimage_unseen_accessed": False,
        "defactify_accessed": False,
    }
    (output / "f04_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(gate, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
