"""Deterministic Protocol-v3 semantic-breaking and degradation views.

These transforms are isolated to the NPR research path. They do not alter the
existing dataset classes or CLIP preprocessing.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Optional

from PIL import Image
import torch

from models.npr_detector import pil_to_unit_rgb_tensor, prepare_npr_tensor
from robustness.transforms import GaussianBlur, JpegCompression, ResizeDownUp


GRID = (4, 4)
PATCH_SIZE = (56, 56)
SEMANTIC_PROBABILITY_TRAIN = 0.5
HORIZONTAL_FLIP_PROBABILITY = 0.5
CONSISTENCY_PARTNER_PROBABILITY = 0.5
DEGRADATION_NAMES = ("jpeg", "resize", "blur")
JPEG_QUALITY = 70
RESIZE_SCALE = 0.5
BLUR_RADIUS = 1.0


@dataclass(frozen=True)
class ResearchNprViews:
    clean: torch.Tensor
    degraded: Optional[torch.Tensor]
    label: float
    sample_id: str
    provenance: dict


def _seed(global_seed: int, epoch: int, sample_id: str, purpose: str) -> int:
    payload = f"{int(global_seed)}|{int(epoch)}|{sample_id}|{purpose}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & ((1 << 63) - 1)


def _generator(global_seed: int, epoch: int, sample_id: str, purpose: str) -> torch.Generator:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(_seed(global_seed, epoch, sample_id, purpose))
    return generator


def deterministic_decision(
    probability: float,
    global_seed: int,
    epoch: int,
    sample_id: str,
    purpose: str,
) -> bool:
    if not 0.0 <= probability <= 1.0:
        raise ValueError(f"Probability must be in [0,1], got {probability}")
    if probability == 0.0:
        return False
    if probability == 1.0:
        return True
    draw = torch.rand((), generator=_generator(global_seed, epoch, sample_id, purpose))
    return bool(draw.item() < probability)


def deterministic_patch_permutation(
    global_seed: int,
    epoch: int,
    sample_id: str,
    grid: tuple[int, int] = GRID,
) -> torch.Tensor:
    count = grid[0] * grid[1]
    if count < 2:
        raise ValueError("Patch grid must contain at least two patches")
    permutation = torch.randperm(
        count, generator=_generator(global_seed, epoch, sample_id, "patch-permutation")
    )
    identity = torch.arange(count)
    if torch.equal(permutation, identity):
        permutation = torch.roll(permutation, shifts=1)
    return permutation


def shuffle_tensor_patches(
    rgb: torch.Tensor,
    permutation: torch.Tensor,
    grid: tuple[int, int] = GRID,
) -> torch.Tensor:
    """Return CHW with output patch i copied from input patch permutation[i]."""
    if not isinstance(rgb, torch.Tensor) or rgb.ndim != 3:
        raise ValueError("Patch shuffle expects a CHW tensor")
    channels, height, width = rgb.shape
    grid_h, grid_w = grid
    if height % grid_h or width % grid_w:
        raise ValueError(f"Image shape {(height, width)} is not divisible by grid {grid}")
    count = grid_h * grid_w
    if tuple(permutation.shape) != (count,):
        raise ValueError(f"Expected permutation shape {(count,)}, got {tuple(permutation.shape)}")
    if permutation.dtype != torch.long:
        raise TypeError("Patch permutation must have dtype torch.long")
    if set(permutation.cpu().tolist()) != set(range(count)):
        raise ValueError("Patch permutation must contain each patch index exactly once")

    patch_h, patch_w = height // grid_h, width // grid_w
    patches = (
        rgb.reshape(channels, grid_h, patch_h, grid_w, patch_w)
        .permute(0, 1, 3, 2, 4)
        .reshape(channels, count, patch_h, patch_w)
    )
    shuffled = patches[:, permutation.to(rgb.device)]
    return (
        shuffled.reshape(channels, grid_h, grid_w, patch_h, patch_w)
        .permute(0, 1, 3, 2, 4)
        .reshape(channels, height, width)
    )


def apply_degradation(image: Image.Image, name: str) -> Image.Image:
    if name == "jpeg":
        return JpegCompression(JPEG_QUALITY)(image)
    if name == "resize":
        return ResizeDownUp(RESIZE_SCALE)(image)
    if name == "blur":
        return GaussianBlur(BLUR_RADIUS)(image)
    if name == "clean":
        return image.convert("RGB")
    raise ValueError(f"Unknown research degradation: {name}")


def deterministic_degradation_name(
    global_seed: int, epoch: int, sample_id: str
) -> str:
    index = torch.randint(
        len(DEGRADATION_NAMES),
        (),
        generator=_generator(global_seed, epoch, sample_id, "degradation-choice"),
    ).item()
    return DEGRADATION_NAMES[index]


def transform_provenance() -> dict:
    """Return the complete immutable transform configuration for run registries."""
    return {
        "schema": "research_npr_transforms_v3",
        "semantic_breaking": {
            "branch": "npr_only",
            "grid": list(GRID),
            "patch_size": list(PATCH_SIZE),
            "training_probability": SEMANTIC_PROBABILITY_TRAIN,
            "validation_probability": 0.0,
            "reject_identity_permutation": True,
            "rng_key": ["global_seed", "epoch", "sample_id"],
        },
        "horizontal_flip_probability": HORIZONTAL_FLIP_PROBABILITY,
        "degradations": {
            "jpeg_quality": JPEG_QUALITY,
            "resize_scale": RESIZE_SCALE,
            "blur_radius": BLUR_RADIUS,
            "evaluation_probability": 1.0,
            "consistency_partner_probability": CONSISTENCY_PARTNER_PROBABILITY,
            "consistency_selection": "uniform",
            "choices": list(DEGRADATION_NAMES),
        },
        "npr": {
            "direction": "diagonal_down_right",
            "padding": "none",
            "output_shape": [3, 223, 223],
        },
    }


class ResearchNprViewBuilder:
    """Build deterministic clean and optional degraded NPR views for one sample."""

    def __init__(
        self,
        global_seed: int,
        epoch: int = 0,
        training: bool = False,
        include_consistency_partner: bool = False,
        semantic_probability: float = SEMANTIC_PROBABILITY_TRAIN,
        partner_probability: float = CONSISTENCY_PARTNER_PROBABILITY,
        forced_degradation: Optional[str] = None,
    ):
        if forced_degradation is not None and forced_degradation not in (
            "clean",
            *DEGRADATION_NAMES,
        ):
            raise ValueError(f"Unknown forced degradation: {forced_degradation}")
        self.global_seed = int(global_seed)
        self.epoch = int(epoch)
        self.training = bool(training)
        self.include_consistency_partner = bool(include_consistency_partner)
        self.semantic_probability = float(semantic_probability)
        self.partner_probability = float(partner_probability)
        self.forced_degradation = forced_degradation

    def __call__(
        self, image: Image.Image, label: float, sample_id: str
    ) -> ResearchNprViews:
        sample_id = str(sample_id)
        base_image = image.convert("RGB")
        clean_rgb = pil_to_unit_rgb_tensor(base_image)

        flip_applied = self.training and deterministic_decision(
            HORIZONTAL_FLIP_PROBABILITY,
            self.global_seed,
            self.epoch,
            sample_id,
            "horizontal-flip",
        )
        if flip_applied:
            clean_rgb = torch.flip(clean_rgb, dims=(-1,))

        shuffle_applied = self.training and deterministic_decision(
            self.semantic_probability,
            self.global_seed,
            self.epoch,
            sample_id,
            "semantic-apply",
        )
        permutation = None
        if shuffle_applied:
            permutation = deterministic_patch_permutation(
                self.global_seed, self.epoch, sample_id
            )
            clean_rgb = shuffle_tensor_patches(clean_rgb, permutation)

        degradation_name = self.forced_degradation
        if degradation_name is None and self.include_consistency_partner:
            use_partner = deterministic_decision(
                self.partner_probability,
                self.global_seed,
                self.epoch,
                sample_id,
                "degradation-apply",
            )
            if use_partner:
                degradation_name = deterministic_degradation_name(
                    self.global_seed, self.epoch, sample_id
                )

        degraded_npr = None
        if degradation_name is not None:
            degraded_rgb = pil_to_unit_rgb_tensor(
                apply_degradation(base_image, degradation_name)
            )
            if flip_applied:
                degraded_rgb = torch.flip(degraded_rgb, dims=(-1,))
            if permutation is not None:
                degraded_rgb = shuffle_tensor_patches(degraded_rgb, permutation)
            degraded_npr = prepare_npr_tensor(degraded_rgb)

        provenance = transform_provenance()
        provenance["sample_transform"] = {
            "global_seed": self.global_seed,
            "epoch": self.epoch,
            "sample_id": sample_id,
            "training": self.training,
            "horizontal_flip_applied": flip_applied,
            "semantic_shuffle_applied": shuffle_applied,
            "patch_permutation": permutation.tolist() if permutation is not None else None,
            "degradation": degradation_name,
        }
        return ResearchNprViews(
            clean=prepare_npr_tensor(clean_rgb),
            degraded=degraded_npr,
            label=float(label),
            sample_id=sample_id,
            provenance=provenance,
        )
