"""Frozen-contract RINE-lite head for cached multi-block CLIP CLS features."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


S2_BLOCK_COUNT = 4
S2_INPUT_DIM = 768
S2_PROJECTION_DIM = 128
S2_TIE_TEMPERATURE = 1.0


class RineLiteDetector(nn.Module):
    """Shared projection, sample-conditioned TIE, and one-logit classifier."""

    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(S2_INPUT_DIM, S2_PROJECTION_DIM, bias=True)
        self.activation = nn.ReLU()
        self.importance_scorer = nn.Linear(
            S2_PROJECTION_DIM, 1, bias=True
        )
        self.classifier = nn.Linear(S2_PROJECTION_DIM, 1, bias=True)

    def encode(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self._validate_features(features)
        projected = self.activation(self.projection(features))
        importance_logits = self.importance_scorer(projected).squeeze(-1)
        importance = torch.softmax(
            importance_logits / S2_TIE_TEMPERATURE, dim=1
        )
        aggregated = torch.sum(projected * importance.unsqueeze(-1), dim=1)
        embedding = F.normalize(aggregated, dim=1)
        if not torch.isfinite(embedding).all() or not torch.isfinite(
            importance
        ).all():
            raise ValueError("RINE-lite head produced non-finite values")
        return embedding, importance

    def forward_with_aux(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        embedding, importance = self.encode(features)
        logits = self.classifier(embedding).squeeze(-1)
        return {
            "logits": logits,
            "embedding": embedding,
            "importance": importance,
        }

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.forward_with_aux(features)["logits"]

    @staticmethod
    def _validate_features(features: torch.Tensor) -> None:
        if not isinstance(features, torch.Tensor):
            raise TypeError("RINE-lite features must be a torch.Tensor")
        expected_tail = (S2_BLOCK_COUNT, S2_INPUT_DIM)
        if features.ndim != 3 or tuple(features.shape[1:]) != expected_tail:
            raise ValueError(
                "RINE-lite requires multi-block features with shape "
                f"[batch,{S2_BLOCK_COUNT},{S2_INPUT_DIM}], "
                f"got {tuple(features.shape)}"
            )
        if features.shape[0] == 0:
            raise ValueError("RINE-lite requires a non-empty batch")
        if not torch.is_floating_point(features):
            raise TypeError("RINE-lite features must be floating point")
        if not torch.isfinite(features).all():
            raise ValueError("RINE-lite features must be finite")


def build_rine_lite_detector() -> RineLiteDetector:
    return RineLiteDetector()
