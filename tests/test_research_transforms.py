import unittest

from PIL import Image
import torch

from data_pipeline.research_transforms import (
    ResearchNprViewBuilder,
    deterministic_patch_permutation,
    shuffle_tensor_patches,
    transform_provenance,
)
from models.npr_detector import pil_to_npr_tensor


def patch_index_tensor() -> torch.Tensor:
    image = torch.empty(3, 224, 224)
    for index in range(16):
        row, col = divmod(index, 4)
        image[:, row * 56 : (row + 1) * 56, col * 56 : (col + 1) * 56] = index / 15.0
    return image


def patterned_pil() -> Image.Image:
    rows = torch.arange(224, dtype=torch.uint8).view(224, 1)
    cols = torch.arange(224, dtype=torch.uint8).view(1, 224)
    tensor = torch.stack(
        (
            rows.expand(224, 224),
            cols.expand(224, 224),
            (rows.to(torch.int16) + cols.to(torch.int16)).remainder(256).to(torch.uint8),
        ),
        dim=-1,
    )
    return Image.fromarray(tensor.numpy(), mode="RGB")


class ResearchTransformTests(unittest.TestCase):
    def test_same_key_is_deterministic(self):
        image = patterned_pil()
        builder = ResearchNprViewBuilder(
            global_seed=42,
            epoch=3,
            training=True,
            include_consistency_partner=True,
            semantic_probability=1.0,
            partner_probability=1.0,
        )
        first = builder(image, 1, "sample-a")
        second = builder(image, 1, "sample-a")
        self.assertTrue(torch.equal(first.clean, second.clean))
        self.assertTrue(torch.equal(first.degraded, second.degraded))
        self.assertEqual(first.provenance, second.provenance)

    def test_seed_and_epoch_change_the_permutation_and_result(self):
        image = patterned_pil()
        first = ResearchNprViewBuilder(42, epoch=0, training=True, semantic_probability=1.0)(
            image, 0, "sample-a"
        )
        second = ResearchNprViewBuilder(43, epoch=0, training=True, semantic_probability=1.0)(
            image, 0, "sample-a"
        )
        third = ResearchNprViewBuilder(42, epoch=1, training=True, semantic_probability=1.0)(
            image, 0, "sample-a"
        )
        permutations = {
            tuple(item.provenance["sample_transform"]["patch_permutation"])
            for item in (first, second, third)
        }
        self.assertEqual(len(permutations), 3)
        self.assertFalse(torch.equal(first.clean, second.clean))
        self.assertFalse(torch.equal(first.clean, third.clean))

    def test_patch_count_permutation_and_boundaries_are_exact(self):
        source = patch_index_tensor()
        permutation = deterministic_patch_permutation(42, 0, "patch-test")
        shuffled = shuffle_tensor_patches(source, permutation)
        self.assertEqual(sorted(permutation.tolist()), list(range(16)))
        self.assertNotEqual(permutation.tolist(), list(range(16)))
        for output_index, input_index in enumerate(permutation.tolist()):
            row, col = divmod(output_index, 4)
            patch = shuffled[:, row * 56 : (row + 1) * 56, col * 56 : (col + 1) * 56]
            self.assertTrue(torch.equal(patch, torch.full_like(patch, input_index / 15.0)))

    def test_paired_views_keep_label_id_and_share_spatial_decisions(self):
        result = ResearchNprViewBuilder(
            42,
            epoch=2,
            training=True,
            semantic_probability=1.0,
            forced_degradation="jpeg",
        )(patterned_pil(), 1, "paired-sample")
        self.assertEqual(result.label, 1.0)
        self.assertEqual(result.sample_id, "paired-sample")
        self.assertIsNotNone(result.degraded)
        self.assertEqual(tuple(result.clean.shape), (3, 223, 223))
        self.assertEqual(tuple(result.degraded.shape), (3, 223, 223))
        self.assertEqual(result.provenance["sample_transform"]["degradation"], "jpeg")
        self.assertIsNotNone(result.provenance["sample_transform"]["patch_permutation"])

    def test_default_evaluation_path_is_clean_and_nonrandom(self):
        image = patterned_pil()
        result = ResearchNprViewBuilder(999, epoch=100, training=False)(
            image, 0, "validation-sample"
        )
        self.assertTrue(torch.equal(result.clean, pil_to_npr_tensor(image)))
        self.assertIsNone(result.degraded)
        sample = result.provenance["sample_transform"]
        self.assertFalse(sample["horizontal_flip_applied"])
        self.assertFalse(sample["semantic_shuffle_applied"])
        self.assertIsNone(sample["patch_permutation"])
        self.assertIsNone(sample["degradation"])

    def test_shuffle_never_mixes_samples(self):
        permutation = deterministic_patch_permutation(42, 0, "batch-isolation")
        low = patch_index_tensor() * 0.25
        high = 0.75 + patch_index_tensor() * 0.25
        shuffled_low = shuffle_tensor_patches(low, permutation)
        shuffled_high = shuffle_tensor_patches(high, permutation)
        self.assertLessEqual(shuffled_low.max().item(), 0.25)
        self.assertGreaterEqual(shuffled_high.min().item(), 0.75)

    def test_transform_does_not_advance_global_torch_rng(self):
        torch.manual_seed(1234)
        before = torch.random.get_rng_state().clone()
        ResearchNprViewBuilder(
            42,
            epoch=0,
            training=True,
            include_consistency_partner=True,
            semantic_probability=1.0,
            partner_probability=1.0,
        )(patterned_pil(), 1, "rng-sample")
        after = torch.random.get_rng_state()
        self.assertTrue(torch.equal(before, after))

    def test_each_fixed_degradation_is_available_and_deterministic(self):
        image = patterned_pil()
        for name in ("clean", "jpeg", "resize", "blur"):
            with self.subTest(name=name):
                builder = ResearchNprViewBuilder(
                    42, training=False, forced_degradation=name
                )
                first = builder(image, 0, "fixed")
                second = builder(image, 0, "fixed")
                self.assertTrue(torch.equal(first.degraded, second.degraded))
                self.assertEqual(first.provenance["sample_transform"]["degradation"], name)

    def test_provenance_contains_complete_frozen_transform_config(self):
        provenance = transform_provenance()
        self.assertEqual(provenance["schema"], "research_npr_transforms_v3")
        self.assertEqual(provenance["semantic_breaking"]["branch"], "npr_only")
        self.assertEqual(provenance["semantic_breaking"]["grid"], [4, 4])
        self.assertEqual(provenance["semantic_breaking"]["patch_size"], [56, 56])
        self.assertEqual(provenance["semantic_breaking"]["training_probability"], 0.5)
        self.assertEqual(provenance["horizontal_flip_probability"], 0.5)
        self.assertEqual(provenance["degradations"]["jpeg_quality"], 70)
        self.assertEqual(provenance["degradations"]["resize_scale"], 0.5)
        self.assertEqual(provenance["degradations"]["blur_radius"], 1.0)
        self.assertEqual(provenance["degradations"]["choices"], ["jpeg", "resize", "blur"])
        self.assertEqual(provenance["npr"]["output_shape"], [3, 223, 223])

    def test_invalid_patch_permutation_and_degradation_fail(self):
        with self.assertRaisesRegex(ValueError, "exactly once"):
            shuffle_tensor_patches(torch.zeros(3, 224, 224), torch.zeros(16, dtype=torch.long))
        with self.assertRaisesRegex(ValueError, "forced degradation"):
            ResearchNprViewBuilder(42, forced_degradation="unknown")


if __name__ == "__main__":
    unittest.main()
