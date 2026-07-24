"""Protocol-v3 neighboring-pixel-relation (NPR) input representation.

The detector is the Protocol-v3 ImageNet-pretrained ResNet18 with its standard
three-channel stem and a single-logit classifier.
"""

from __future__ import annotations

from PIL import Image
import torch
import torch.nn as nn
from torchvision.transforms.functional import pil_to_tensor
from torchvision.models import ResNet18_Weights, resnet18


IMAGE_SIZE = 224
NPR_SIZE = IMAGE_SIZE - 1
RANGE_TOLERANCE = 1e-6
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def pil_to_unit_rgb_tensor(image: Image.Image) -> torch.Tensor:
    """Convert a PIL image to resized RGB float32 CHW in the closed interval [0, 1]."""
    if not isinstance(image, Image.Image):
        raise TypeError(f"Expected PIL.Image.Image, got {type(image).__name__}")
    rgb = image.convert("RGB").resize(
        (IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BICUBIC
    )
    return pil_to_tensor(rgb).to(dtype=torch.float32).div(255.0)


def _validate_rgb_tensor(rgb: torch.Tensor) -> None:
    if not isinstance(rgb, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor, got {type(rgb).__name__}")
    if rgb.ndim not in (3, 4):
        raise ValueError(f"Expected CHW or BCHW tensor, got shape {tuple(rgb.shape)}")
    channel_axis = 0 if rgb.ndim == 3 else 1
    if rgb.shape[channel_axis] != 3:
        raise ValueError(f"Expected three RGB channels, got shape {tuple(rgb.shape)}")
    if tuple(rgb.shape[-2:]) != (IMAGE_SIZE, IMAGE_SIZE):
        raise ValueError(
            f"Expected spatial shape {(IMAGE_SIZE, IMAGE_SIZE)}, got {tuple(rgb.shape[-2:])}"
        )
    if not rgb.is_floating_point():
        raise TypeError(f"Expected floating-point RGB input, got {rgb.dtype}")
    if not torch.isfinite(rgb).all():
        raise ValueError("RGB input contains non-finite values")
    minimum = rgb.detach().amin().item()
    maximum = rgb.detach().amax().item()
    if minimum < -RANGE_TOLERANCE or maximum > 1.0 + RANGE_TOLERANCE:
        raise ValueError(f"RGB input range must be [0,1], got [{minimum}, {maximum}]")


def signed_diagonal_npr(rgb: torch.Tensor) -> torch.Tensor:
    """Compute signed down-right diagonal differences without padding.

    For CHW or BCHW RGB input, the last two dimensions follow
    ``rgb[..., :-1, :-1] - rgb[..., 1:, 1:]``. The output is 223x223.
    """
    _validate_rgb_tensor(rgb)
    return rgb[..., :-1, :-1] - rgb[..., 1:, 1:]


def map_and_normalize_npr(signed_npr: torch.Tensor) -> torch.Tensor:
    """Map signed NPR from [-1,1] to [0,1], then apply ImageNet normalization."""
    if not isinstance(signed_npr, torch.Tensor) or signed_npr.ndim not in (3, 4):
        raise ValueError("Expected a CHW or BCHW NPR tensor")
    if not signed_npr.is_floating_point():
        raise TypeError(f"Expected floating-point NPR input, got {signed_npr.dtype}")
    if tuple(signed_npr.shape[-2:]) != (NPR_SIZE, NPR_SIZE):
        raise ValueError(
            f"Expected NPR spatial shape {(NPR_SIZE, NPR_SIZE)}, "
            f"got {tuple(signed_npr.shape[-2:])}"
        )
    if not torch.isfinite(signed_npr).all():
        raise ValueError("NPR input contains non-finite values")
    minimum = signed_npr.detach().amin().item()
    maximum = signed_npr.detach().amax().item()
    if minimum < -1.0 - RANGE_TOLERANCE or maximum > 1.0 + RANGE_TOLERANCE:
        raise ValueError(f"Signed NPR range must be [-1,1], got [{minimum}, {maximum}]")

    mapped = (signed_npr + 1.0) / 2.0
    channel_axis = 0 if mapped.ndim == 3 else 1
    shape = [1] * mapped.ndim
    shape[channel_axis] = 3
    mean = mapped.new_tensor(IMAGENET_MEAN).view(shape)
    std = mapped.new_tensor(IMAGENET_STD).view(shape)
    return (mapped - mean) / std


def prepare_npr_tensor(rgb: torch.Tensor) -> torch.Tensor:
    """Apply the complete tensor portion of the approved NPR contract."""
    return map_and_normalize_npr(signed_diagonal_npr(rgb))


def pil_to_npr_tensor(image: Image.Image) -> torch.Tensor:
    """Apply RGB conversion, bicubic resize, NPR mapping, and normalization."""
    return prepare_npr_tensor(pil_to_unit_rgb_tensor(image))


class NprInputTransform(nn.Module):
    """Stateless module wrapper for CHW/BCHW NPR preparation."""

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        return prepare_npr_tensor(rgb)


def build_npr_detector(pretrained: bool = True) -> nn.Module:
    """Build the frozen S1 NPR architecture without adding auxiliary modules."""
    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model = resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, 1)
    return model
