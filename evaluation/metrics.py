import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def binary_metrics(labels, probs, threshold=0.5):
    """Compute binary real-vs-fake metrics.

    Label 0 is real and label 1 is fake. ``recall`` and ``f1`` are retained as
    backward-compatible aliases for ``fake_recall`` and ``binary_f1``.
    Balanced accuracy, AUROC, and AUPRC are undefined (NaN) when only one
    class is present. The confusion matrix always uses the [real, fake] order.
    """
    labels = np.asarray(labels, dtype=int).reshape(-1)
    probs = np.asarray(probs, dtype=float).reshape(-1)
    if labels.size == 0:
        raise ValueError("labels and probs must not be empty")
    if labels.shape != probs.shape:
        raise ValueError("labels and probs must have the same number of elements")
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("labels must contain only 0 (real) and 1 (fake)")
    if not np.isfinite(probs).all():
        raise ValueError("probs must contain only finite values")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1")

    preds = (probs >= threshold).astype(int)
    fake_recall = recall_score(labels, preds, pos_label=1, zero_division=0)
    real_recall = recall_score(labels, preds, pos_label=0, zero_division=0)
    binary_f1 = f1_score(labels, preds, pos_label=1, zero_division=0)
    unique_labels = np.unique(labels)
    if unique_labels.size == 2:
        balanced_accuracy = (fake_recall + real_recall) / 2.0
        auroc = roc_auc_score(labels, probs)
        auprc = average_precision_score(labels, probs)
    else:
        balanced_accuracy = float("nan")
        auroc = float("nan")
        auprc = float("nan")

    return {
        "accuracy": accuracy_score(labels, preds),
        "balanced_accuracy": balanced_accuracy,
        "precision": precision_score(labels, preds, zero_division=0),
        "fake_recall": fake_recall,
        "real_recall": real_recall,
        "binary_f1": binary_f1,
        "macro_f1": f1_score(labels, preds, labels=[0, 1], average="macro", zero_division=0),
        "auroc": auroc,
        "auprc": auprc,
        "confusion_matrix": confusion_matrix(labels, preds, labels=[0, 1]),
        # Backward-compatible names used by existing training and result files.
        "recall": fake_recall,
        "f1": binary_f1,
    }
