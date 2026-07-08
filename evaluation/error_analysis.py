from pathlib import Path
import shutil


def save_error_cases(pred_df, experiment_name, max_per_type=50):
    base = Path("outputs/error_cases") / experiment_name
    fake_as_real = base / "fake_predicted_real"
    real_as_fake = base / "real_predicted_fake"
    fake_as_real.mkdir(parents=True, exist_ok=True)
    real_as_fake.mkdir(parents=True, exist_ok=True)

    for _, row in pred_df[(pred_df.label == 1) & (pred_df.pred_label == 0)].head(max_per_type).iterrows():
        src = Path(row["path"])
        if src.exists():
            shutil.copy2(src, fake_as_real / src.name)
    for _, row in pred_df[(pred_df.label == 0) & (pred_df.pred_label == 1)].head(max_per_type).iterrows():
        src = Path(row["path"])
        if src.exists():
            shutil.copy2(src, real_as_fake / src.name)

