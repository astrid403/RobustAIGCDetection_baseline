"""Research-v3 objectives fixed by the S1 contract."""

import torch
import torch.nn as nn


def build_npr_objective(consistency_weight: float = 0.0) -> nn.Module:
    if float(consistency_weight) != 0.0:
        raise ValueError("S1 main requires consistency_weight=0.0")
    return nn.BCEWithLogitsLoss()


def probability_consistency_loss(clean_logits: torch.Tensor, degraded_logits: torch.Tensor):
    """Pre-registered Task-09 helper; not enabled by the S1 main path."""
    return torch.mean((torch.sigmoid(clean_logits) - torch.sigmoid(degraded_logits)) ** 2)
