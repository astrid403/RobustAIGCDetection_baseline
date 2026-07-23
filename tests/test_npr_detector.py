import unittest

from PIL import Image
import torch

from models.npr_detector import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    NprInputTransform,
    map_and_normalize_npr,
    pil_to_npr_tensor,
    pil_to_unit_rgb_tensor,
    prepare_npr_tensor,
    signed_diagonal_npr,
)


class NprInputRepresentationTests(unittest.TestCase):
    def test_constant_image_has_zero_signed_npr_and_expected_normalization(self):
        rgb = torch.full((3, 224, 224), 0.25)
        signed = signed_diagonal_npr(rgb)
        self.assertEqual(tuple(signed.shape), (3, 223, 223))
        self.assertTrue(torch.equal(signed, torch.zeros_like(signed)))
        output = prepare_npr_tensor(rgb)
        expected = torch.tensor(
            [(0.5 - mean) / std for mean, std in zip(IMAGENET_MEAN, IMAGENET_STD)]
        ).view(3, 1, 1)
        self.assertTrue(torch.allclose(output, expected.expand_as(output), atol=1e-7))

    def test_horizontal_and_vertical_gradients_have_analytic_difference(self):
        ramp = torch.linspace(0.0, 1.0, 224)
        horizontal = ramp.view(1, 1, 224).expand(3, 224, 224).clone()
        vertical = ramp.view(1, 224, 1).expand(3, 224, 224).clone()
        expected = torch.full((3, 223, 223), -1.0 / 223.0)
        self.assertTrue(torch.allclose(signed_diagonal_npr(horizontal), expected, atol=1e-7))
        self.assertTrue(torch.allclose(signed_diagonal_npr(vertical), expected, atol=1e-7))

    def test_checkerboard_diagonal_neighbors_are_equal(self):
        rows = torch.arange(224).view(224, 1)
        cols = torch.arange(224).view(1, 224)
        checkerboard = ((rows + cols) % 2).float().unsqueeze(0).expand(3, -1, -1)
        self.assertTrue(
            torch.equal(signed_diagonal_npr(checkerboard), torch.zeros(3, 223, 223))
        )

    def test_batch_matches_stacked_single_results(self):
        torch.manual_seed(7)
        batch = torch.rand(4, 3, 224, 224)
        together = prepare_npr_tensor(batch)
        separate = torch.stack([prepare_npr_tensor(item) for item in batch])
        self.assertEqual(tuple(together.shape), (4, 3, 223, 223))
        self.assertTrue(torch.equal(together, separate))

    def test_transform_is_deterministic_and_does_not_modify_input(self):
        torch.manual_seed(11)
        rgb = torch.rand(3, 224, 224)
        original = rgb.clone()
        transform = NprInputTransform()
        first = transform(rgb)
        second = transform(rgb)
        self.assertTrue(torch.equal(rgb, original))
        self.assertTrue(torch.equal(first, second))
        self.assertTrue(torch.isfinite(first).all())

    def test_autograd_flows_to_input(self):
        rgb = torch.rand(2, 3, 224, 224, requires_grad=True)
        prepare_npr_tensor(rgb).sum().backward()
        self.assertIsNotNone(rgb.grad)
        self.assertEqual(tuple(rgb.grad.shape), tuple(rgb.shape))
        self.assertTrue(torch.isfinite(rgb.grad).all())

    def test_pil_rgb_conversion_and_normalization_order(self):
        # P mode forces an explicit RGB conversion; channel values then verify
        # that conversion happens before NPR and ImageNet normalization.
        palette = Image.new("P", (224, 224))
        palette.putpalette([255, 0, 0] + [0, 0, 0] * 255)
        unit_rgb = pil_to_unit_rgb_tensor(palette)
        self.assertEqual(unit_rgb.dtype, torch.float32)
        self.assertEqual(tuple(unit_rgb.shape), (3, 224, 224))
        self.assertTrue(torch.equal(unit_rgb[0], torch.ones(224, 224)))
        self.assertTrue(torch.equal(unit_rgb[1], torch.zeros(224, 224)))
        self.assertTrue(torch.equal(unit_rgb[2], torch.zeros(224, 224)))
        self.assertTrue(torch.equal(pil_to_npr_tensor(palette), prepare_npr_tensor(unit_rgb)))

    def test_mapping_preserves_dtype_and_device(self):
        signed = torch.zeros(3, 223, 223, dtype=torch.float64)
        output = map_and_normalize_npr(signed)
        self.assertEqual(output.dtype, torch.float64)
        self.assertEqual(output.device, signed.device)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_cpu_cuda_consistency(self):
        torch.manual_seed(17)
        rgb = torch.rand(2, 3, 224, 224)
        cpu = prepare_npr_tensor(rgb)
        cuda = prepare_npr_tensor(rgb.cuda()).cpu()
        self.assertTrue(torch.allclose(cpu, cuda, atol=1e-6, rtol=1e-6))

    def test_invalid_shape_dtype_range_and_finite_values_fail(self):
        with self.assertRaisesRegex(ValueError, "CHW or BCHW"):
            prepare_npr_tensor(torch.zeros(1, 1, 3, 224, 224))
        with self.assertRaisesRegex(ValueError, "three RGB channels"):
            prepare_npr_tensor(torch.zeros(1, 224, 224))
        with self.assertRaisesRegex(ValueError, "spatial shape"):
            prepare_npr_tensor(torch.zeros(3, 223, 223))
        with self.assertRaisesRegex(TypeError, "floating-point"):
            prepare_npr_tensor(torch.zeros(3, 224, 224, dtype=torch.uint8))
        invalid_range = torch.zeros(3, 224, 224)
        invalid_range[0, 0, 0] = 1.1
        with self.assertRaisesRegex(ValueError, r"range must be \[0,1\]"):
            prepare_npr_tensor(invalid_range)
        nonfinite = torch.zeros(3, 224, 224)
        nonfinite[0, 0, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "non-finite"):
            prepare_npr_tensor(nonfinite)


if __name__ == "__main__":
    unittest.main()
