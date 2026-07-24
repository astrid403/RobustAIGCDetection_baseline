import tempfile
import unittest
from pathlib import Path

import pandas as pd

from analysis.s2_logo_gate import CONDITIONS, FOLDS


class S2LogoGateTests(unittest.TestCase):
    def test_frozen_dimensions_are_closed(self):
        self.assertEqual(FOLDS, (
            "holdout_adm", "holdout_biggan", "holdout_stable_diffusion_v15"
        ))
        self.assertEqual(CONDITIONS, ("clean", "jpeg", "resize", "blur"))

    def test_prediction_identity_comparison_detects_reorder(self):
        frame = pd.DataFrame({
            "sample_id": ["a", "b"], "fold": ["f", "f"],
            "label": [0, 1], "probability": [0.1, 0.9],
        })
        identity = frame[["sample_id", "fold", "label"]].reset_index(drop=True)
        reordered = frame.iloc[::-1][["sample_id", "fold", "label"]].reset_index(
            drop=True
        )
        self.assertFalse(identity.equals(reordered))


if __name__ == "__main__":
    unittest.main()
