"""Supervised contrastive objective frozen by the S2 contract."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


S2_SUPCON_TEMPERATURE = 0.07
S2_SUPCON_WEIGHT = 0.1


def supervised_contrastive_loss(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = S2_SUPCON_TEMPERATURE,
) -> torch.Tensor:
    """Binary-label SupCon with self pairs excluded and empty anchors skipped."""
    if embeddings.ndim != 2 or embeddings.shape[0] == 0:
        raise ValueError("SupCon embeddings must have shape [batch,dim]")
    if labels.ndim != 1 or labels.shape[0] != embeddings.shape[0]:
        raise ValueError("SupCon labels must have shape [batch]")
    if not torch.is_floating_point(embeddings):
        raise TypeError("SupCon embeddings must be floating point")
    if not torch.isfinite(embeddings).all():
        raise ValueError("SupCon embeddings must be finite")
    if not torch.isfinite(labels).all() or not torch.isin(
        labels, torch.tensor([0, 1], device=labels.device)
    ).all():
        raise ValueError("SupCon labels must use the binary 0/1 contract")
    if temperature <= 0:
        raise ValueError("SupCon temperature must be positive")

    normalized = F.normalize(embeddings, dim=1)
    logits = normalized @ normalized.transpose(0, 1)
    logits = logits / temperature
    batch_size = embeddings.shape[0]
    self_mask = torch.eye(batch_size, dtype=torch.bool, device=embeddings.device)
    positive_mask = labels[:, None].eq(labels[None, :]) & ~self_mask
    valid_anchor = positive_mask.any(dim=1)
    if not valid_anchor.any():
        return embeddings.sum() * 0.0

    logits = logits.masked_fill(self_mask, float("-inf"))
    log_probability = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    positive_count = positive_mask.sum(dim=1).clamp_min(1)
    positive_log_probability = torch.where(
        positive_mask, log_probability, torch.zeros_like(log_probability)
    ).sum(dim=1) / positive_count
    return -positive_log_probability[valid_anchor].mean()


class RineLiteObjective(nn.Module):
    """BCE + fixed 0.1 binary-label SupCon."""

    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(
        self,
        logits: torch.Tensor,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if logits.ndim != 1 or logits.shape[0] != labels.shape[0]:
            raise ValueError("RINE-lite logits must have shape [batch]")
        float_labels = labels.to(dtype=logits.dtype)
        bce = self.bce(logits, float_labels)
        supcon = supervised_contrastive_loss(embeddings, labels)
        total = bce + S2_SUPCON_WEIGHT * supcon
        return {"total": total, "bce": bce, "supcon": supcon}
