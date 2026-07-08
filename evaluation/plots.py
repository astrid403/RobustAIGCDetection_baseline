from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def plot_confusion_matrix(cm, out_path):
    ensure_dir(Path(out_path).parent)
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], labels=["real", "fake"])
    ax.set_yticks([0, 1], labels=["real", "fake"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_roc(labels, probs, out_path):
    ensure_dir(Path(out_path).parent)
    fig, ax = plt.subplots(figsize=(5, 4))
    try:
        RocCurveDisplay.from_predictions(labels, probs, ax=ax)
    except ValueError:
        ax.text(0.5, 0.5, "ROC unavailable: one class present", ha="center")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_training_curves(log_df, fig_dir):
    ensure_dir(fig_dir)
    if log_df.empty:
        return
    for column, filename, ylabel in [
        ("train_loss", "train_loss_curve.png", "Train loss"),
        ("val_auroc", "val_auroc_curve.png", "Validation AUROC"),
    ]:
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(log_df["epoch"], log_df[column], marker="o")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        fig.tight_layout()
        fig.savefig(Path(fig_dir) / filename, dpi=160)
        plt.close(fig)

