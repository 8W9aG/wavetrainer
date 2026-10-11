"""Tests for the power transformer normaliser."""
import json
import os
import tempfile
import unittest

import numpy as np
import optuna
import pandas as pd

from wavetrainer.normaliser.combined_normaliser import CombinedNormaliser
from wavetrainer.normaliser.powertransformer_normaliser import PowerTransformerNormaliser


def _frame(rows=50):
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "skewed": np.exp(rng.normal(size=rows)),
            "normal": rng.normal(size=rows),
            "constant": np.ones(rows),
        }
    )


class TestPowerTransformerNormaliser(unittest.TestCase):

    def test_transform_returns_input_unchanged(self):
        # Saved models were trained on untransformed features (the old
        # transform wrote into a copy), so inference must not transform them.
        df = _frame()
        expected = df.copy()
        normaliser = PowerTransformerNormaliser().fit(df)
        self.assertTrue(normaliser._pt_cols)
        out = normaliser.transform(df)
        pd.testing.assert_frame_equal(out, expected, check_exact=True)
        pd.testing.assert_frame_equal(df, expected, check_exact=True)
        out = normaliser.fit_transform(df)
        pd.testing.assert_frame_equal(out, expected, check_exact=True)

    def test_saved_fold_loads(self):
        # fit/save/load still run so existing fold folders keep loading and
        # new ones have the same files.
        df = _frame()
        trial = optuna.trial.FixedTrial({})
        with tempfile.TemporaryDirectory() as folder:
            CombinedNormaliser(use_power_transformer=True).fit(df).save(folder, trial)
            self.assertEqual(
                sorted(os.listdir(folder)),
                [
                    "combined_normaliser.json",
                    "power_transformer_columns.json",
                    "power_transformer_normaliser.joblib",
                ],
            )
            with open(os.path.join(folder, "combined_normaliser.json"), encoding="utf8") as handle:
                self.assertEqual(json.load(handle), {"normalisers": ["powertransformer"]})

            loaded = CombinedNormaliser()
            loaded.load(folder)
            self.assertEqual(len(loaded._normalisers), 1)
            pt = loaded._normalisers[0]
            self.assertIsInstance(pt, PowerTransformerNormaliser)
            self.assertEqual(pt._pt_cols, ["skewed", "normal", "constant"])
            self.assertEqual(pt._pt.n_features_in_, 3)
            pd.testing.assert_frame_equal(loaded.transform(df), _frame(), check_exact=True)
