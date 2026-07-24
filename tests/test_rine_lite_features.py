from pathlib import Path
import tempfile
import unittest

import pandas as pd
import torch
from torch import nn

from data_pipeline.fingerprints import (
    RESEARCH_MULTIBLOCK_CACHE_SCHEMA,
    research_multiblock_clip_cache_metadata,
    research_multiblock_clip_cache_path,
    validate_research_multiblock_clip_cache,
)
from models.clip_mlp_detector import (
    S2_MULTIBLOCK_IDS,
    encode_clip_multiblock_cls,
)


class _CountingBlock(nn.Module):
    def __init__(self, width, block_id):
        super().__init__()
        self.block_id = block_id
        self.calls = 0
        self.register_buffer(
            "offset", torch.full((width,), float(block_id) / 100)
        )

    def forward(self, tokens, attn_mask=None):
        self.calls += 1
        return tokens + self.offset


class _Transformer(nn.Module):
    def __init__(self, width, batch_first=True):
        super().__init__()
        self.batch_first = batch_first
        self.resblocks = nn.ModuleList(
            [_CountingBlock(width, block_id) for block_id in range(1, 13)]
        )


class _TinyVisual(nn.Module):
    def __init__(self, width=768, batch_first=True):
        super().__init__()
        self.width = width
        self.embed_calls = 0
        self.transformer = _Transformer(width, batch_first=batch_first)
        self.ln_post = nn.LayerNorm(width)

    def _embeds(self, images):
        self.embed_calls += 1
        batch = images.shape[0]
        base = images.mean(dim=(1, 2, 3), keepdim=False).unsqueeze(1)
        cls = base.repeat(1, self.width)
        patch = torch.flip(cls, dims=[1])
        return torch.stack([cls, patch], dim=1)


class _TinyClip(nn.Module):
    def __init__(self, batch_first=True):
        super().__init__()
        self.visual = _TinyVisual(batch_first=batch_first)


def _independent_block_reference(model, images, target):
    visual = model.visual
    tokens = visual._embeds(images)
    if not visual.transformer.batch_first:
        tokens = tokens.transpose(0, 1).contiguous()
    for block in list(visual.transformer.resblocks)[:target]:
        tokens = block(tokens, attn_mask=None)
    cls = tokens[:, 0] if visual.transformer.batch_first else tokens[0]
    return visual.ln_post(cls)


class RineLiteFeatureTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.images = torch.randn(2, 3, 4, 4)

    def test_shape_order_determinism_and_single_traversal(self):
        model = _TinyClip().eval()
        first = encode_clip_multiblock_cls(model, self.images)
        self.assertEqual(tuple(first.shape), (2, 4, 768))
        self.assertEqual(model.visual.embed_calls, 1)
        self.assertEqual(
            [block.calls for block in model.visual.transformer.resblocks],
            [1] * 12,
        )

        reference_model = _TinyClip().eval()
        reference_model.load_state_dict(model.state_dict())
        expected = torch.stack(
            [
                _independent_block_reference(reference_model, self.images, block_id)
                for block_id in S2_MULTIBLOCK_IDS
            ],
            dim=1,
        )
        torch.testing.assert_close(first, expected, rtol=0, atol=0)
        second = encode_clip_multiblock_cls(model, self.images)
        torch.testing.assert_close(first, second, rtol=0, atol=0)
        self.assertFalse(first.requires_grad)
        self.assertTrue(torch.isfinite(first).all())

    def test_sequence_first_transformer_is_equivalent(self):
        batch_first = _TinyClip(batch_first=True).eval()
        sequence_first = _TinyClip(batch_first=False).eval()
        sequence_first.load_state_dict(batch_first.state_dict())
        left = encode_clip_multiblock_cls(batch_first, self.images)
        right = encode_clip_multiblock_cls(sequence_first, self.images)
        torch.testing.assert_close(left, right, rtol=0, atol=0)

    def test_wrong_blocks_order_width_and_visual_contract_hard_fail(self):
        model = _TinyClip().eval()
        for invalid in ((3, 6, 12, 9), (3, 6, 9), (2, 6, 9, 12)):
            with self.assertRaisesRegex(ValueError, "frozen ordered block IDs"):
                encode_clip_multiblock_cls(model, self.images, invalid)
        wrong_width = _TinyClip()
        wrong_width.visual = _TinyVisual(width=16)
        with self.assertRaisesRegex(ValueError, "shape"):
            encode_clip_multiblock_cls(wrong_width, self.images)
        with self.assertRaisesRegex(TypeError, "clip_model.visual"):
            encode_clip_multiblock_cls(nn.Identity(), self.images)

    def test_cpu_gpu_consistency_when_available(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA is unavailable")
        cpu_model = _TinyClip().eval()
        gpu_model = _TinyClip().eval().cuda()
        gpu_model.load_state_dict(cpu_model.state_dict())
        cpu = encode_clip_multiblock_cls(cpu_model, self.images)
        gpu = encode_clip_multiblock_cls(gpu_model, self.images.cuda()).cpu()
        torch.testing.assert_close(cpu, gpu, rtol=1e-6, atol=2e-6)

    def test_cache_signature_and_metadata_mismatch_hard_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "train.csv"
            pd.DataFrame(
                [{"sample_id": "a", "path": "a.png", "label": 0}]
            ).to_csv(manifest, index=False)
            config = {
                "clip_model": "ViT-B-32",
                "pretrained": "openai",
                "clip_block_ids": [3, 6, 9, 12],
                "image_size": 224,
            }
            expected = research_multiblock_clip_cache_metadata(
                manifest, config, "holdout_adm", "train"
            )
            self.assertEqual(
                expected["schema"], RESEARCH_MULTIBLOCK_CACHE_SCHEMA
            )
            self.assertEqual(expected["block_ids"], [3, 6, 9, 12])
            self.assertIn("open_clip_version", expected)
            self.assertIn("preprocess_fingerprint", expected)
            self.assertEqual(
                expected["extraction_version"], "clip_multiblock_cls_v1"
            )
            self.assertIn(
                "multiblock",
                str(research_multiblock_clip_cache_path(expected)),
            )
            cache = {
                "features": torch.ones(1, 4, 768),
                "labels": torch.tensor([0.0]),
                "paths": ["a"],
                "metadata": expected,
            }
            validate_research_multiblock_clip_cache(cache, expected)

            for key, value in (
                ("block_ids", [3, 6, 12, 9]),
                ("open_clip_version", "wrong"),
                ("preprocess_fingerprint", "wrong"),
                ("manifest_sha256", "wrong"),
                ("extraction_version", "wrong"),
            ):
                altered = dict(expected)
                altered[key] = value
                with self.assertRaisesRegex(ValueError, "metadata mismatch"):
                    validate_research_multiblock_clip_cache(
                        {**cache, "metadata": altered}, expected
                    )
            with self.assertRaisesRegex(ValueError, "frozen ordered block IDs"):
                research_multiblock_clip_cache_metadata(
                    manifest,
                    {**config, "clip_block_ids": [3, 6, 12, 9]},
                    "holdout_adm",
                    "train",
                )

    def test_cache_tensor_shape_is_strict(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "train.csv"
            pd.DataFrame([{"path": "a.png", "label": 0}]).to_csv(
                manifest, index=False
            )
            expected = research_multiblock_clip_cache_metadata(
                manifest, {}, "holdout_adm", "train"
            )
            cache = {
                "features": torch.ones(1, 768),
                "labels": torch.tensor([0.0]),
                "paths": ["a"],
                "metadata": expected,
            }
            with self.assertRaisesRegex(ValueError, "feature shape mismatch"):
                validate_research_multiblock_clip_cache(cache, expected)


if __name__ == "__main__":
    unittest.main()
