import unittest

import torch
from torch import nn

from models.clip_mlp_detector import encode_clip_image_features


class _ResidualBlock(nn.Module):
    def __init__(self, width, scale):
        super().__init__()
        self.register_buffer("offset", torch.full((width,), scale))

    def forward(self, tokens, attn_mask=None):
        return tokens + self.offset


class _Transformer(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.batch_first = True
        self.resblocks = nn.ModuleList([
            _ResidualBlock(width, 0.1),
            _ResidualBlock(width, 0.2),
            _ResidualBlock(width, 0.3),
        ])


class _Visual(nn.Module):
    def __init__(self, width=4, output_dim=3):
        super().__init__()
        self.transformer = _Transformer(width)
        self.proj = nn.Parameter(torch.arange(width * output_dim, dtype=torch.float32).reshape(width, output_dim) / 10)

    def _embeds(self, images):
        pooled = images.flatten(1)[:, :4]
        patch = torch.flip(pooled, dims=[1])
        return torch.stack([pooled, patch], dim=1)

    def _pool(self, tokens):
        return tokens[:, 0], tokens[:, 1:]

    def forward(self, images):
        tokens = self._embeds(images)
        for block in self.transformer.resblocks:
            tokens = block(tokens, attn_mask=None)
        pooled, _ = self._pool(tokens)
        return pooled @ self.proj


class _Clip(nn.Module):
    def __init__(self):
        super().__init__()
        self.visual = _Visual()

    def encode_image(self, images):
        return self.visual(images)


class ClipFeatureExtractionTest(unittest.TestCase):
    def setUp(self):
        self.model = _Clip().eval()
        for parameter in self.model.parameters():
            parameter.requires_grad = False
        self.images = torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4)

    def test_final_is_normalized_and_backward_compatible(self):
        result = encode_clip_image_features(self.model, self.images, "final")
        expected = self.model.encode_image(self.images)
        expected = expected / expected.norm(dim=-1, keepdim=True)
        torch.testing.assert_close(result, expected)
        torch.testing.assert_close(result.norm(dim=-1), torch.ones(1))

    def test_penultimate_skips_only_last_block(self):
        result = encode_clip_image_features(self.model, self.images, "penultimate")
        tokens = self.model.visual._embeds(self.images)
        for block in list(self.model.visual.transformer.resblocks)[:-1]:
            tokens = block(tokens, attn_mask=None)
        expected = tokens[:, 0] @ self.model.visual.proj
        expected = expected / expected.norm(dim=-1, keepdim=True)
        torch.testing.assert_close(result, expected)
        self.assertEqual(tuple(result.shape), (1, 3))

    def test_both_modes_are_deterministic_finite_and_no_grad(self):
        images = self.images.clone().requires_grad_(True)
        for mode in ("final", "penultimate"):
            first = encode_clip_image_features(self.model, images, mode)
            second = encode_clip_image_features(self.model, images, mode)
            torch.testing.assert_close(first, second, rtol=0, atol=0)
            self.assertTrue(torch.isfinite(first).all())
            self.assertFalse(first.requires_grad)

    def test_unknown_mode_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unsupported CLIP feature mode"):
            encode_clip_image_features(self.model, self.images, "tokens")


if __name__ == "__main__":
    unittest.main()
