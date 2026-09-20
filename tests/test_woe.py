"""Meaningful WOE invariants: isolation, exact counts and deployable bins."""

import json
import unittest

import numpy as np
import pandas as pd

from src.models.woe import WOEEncoder


class WOEEncoderTests(unittest.TestCase):
    def training(self):
        values = np.repeat(np.arange(10, dtype=float), 100)
        target = np.concatenate([np.r_[np.ones(5 + 9 * index), np.zeros(95 - 9 * index)] for index in range(10)]).astype(int)
        values[::47] = np.nan
        return pd.DataFrame({"numeric": values, "category": np.where(np.arange(1000) % 3, "A", "B")}), target

    def test_transform_is_train_only_and_handles_unseen(self):
        frame, target = self.training()
        encoder = WOEEncoder().fit(frame, target, categorical=["category"])
        before = json.dumps(encoder.to_dict(), sort_keys=True, allow_nan=False)
        holdout = pd.DataFrame({"numeric": [-1e100, 1e100, np.nan], "category": ["NEW", "A", None]})
        encoded = encoder.transform(holdout)
        identifiers = encoder.bin_ids(holdout)
        self.assertTrue(np.isfinite(encoded.to_numpy()).all())
        self.assertEqual(encoded.loc[0, "category"], 0)
        self.assertEqual(encoded.loc[2, "category"], 0)
        self.assertEqual(identifiers.loc[0, "category"], "UNSEEN")
        self.assertEqual(identifiers.loc[2, "numeric"], "MISSING")
        self.assertEqual(before, json.dumps(encoder.to_dict(), sort_keys=True, allow_nan=False))

    def test_count_conservation_and_consistent_iv(self):
        frame, target = self.training()
        encoder = WOEEncoder().fit(frame, target)
        for table in encoder.tables():
            for feature, rows in table.groupby("feature"):
                self.assertEqual(rows["count"].sum(), len(frame))
                self.assertEqual(rows["bad"].sum(), target.sum())
                self.assertEqual(rows["good"].sum(), len(target) - target.sum())
                self.assertAlmostEqual(rows["bad_dist"].sum(), 1)
                self.assertAlmostEqual(rows["good_dist"].sum(), 1)
                observed = rows[rows["count"] > 0]
                np.testing.assert_allclose(observed["woe"], np.log(observed["bad_dist"] / observed["good_dist"]))
        _, coarse = encoder.tables()
        summary = encoder.summary().set_index("feature")
        self.assertAlmostEqual(coarse.loc[coarse.feature == "numeric", "iv_component"].sum(), summary.loc["numeric", "iv"])

    def test_numeric_bad_rates_monotone_and_constraints_met(self):
        frame, target = self.training()
        encoder = WOEEncoder().fit(frame[["numeric"]], target)
        _, coarse = encoder.tables()
        regular = coarse[coarse.bin_id.str.startswith("B")]
        self.assertLessEqual(len(regular), 5)
        self.assertGreater(len(regular), 1)
        self.assertTrue((np.diff(regular.bad_rate) >= -1e-12).all())
        self.assertTrue((regular["count"] >= 50).all())
        self.assertTrue((regular.bad >= 20).all())
        self.assertTrue((regular.good >= 20).all())

    def test_json_round_trip_preserves_all_definitions_and_predictions(self):
        frame, target = self.training()
        encoder = WOEEncoder().fit(frame, target)
        payload = json.loads(json.dumps(encoder.to_dict(), allow_nan=False))
        restored = WOEEncoder.from_dict(payload)
        pd.testing.assert_frame_equal(encoder.transform(frame), restored.transform(frame))
        pd.testing.assert_frame_equal(encoder.bin_ids(frame), restored.bin_ids(frame))
        for original, loaded in zip(encoder.tables(), restored.tables()):
            pd.testing.assert_frame_equal(original, loaded)

    def test_all_missing_numeric_and_rare_categories(self):
        frame = pd.DataFrame({"numeric": [np.nan] * 100,
                              "category": ["A"] * 80 + ["B"] * 19 + ["RARE"]})
        target = np.arange(100) % 2
        encoder = WOEEncoder(min_bad=1, min_good=1).fit(frame, target)
        holdout = pd.DataFrame({"numeric": [10, np.nan], "category": ["RARE", "NEVER"]})
        encoded = encoder.transform(holdout)
        self.assertTrue(np.isfinite(encoded.to_numpy()).all())
        self.assertEqual(encoder.bin_ids(holdout).loc[0, "numeric"], "UNSEEN")
        self.assertEqual(encoded.loc[0, "numeric"], 0)
        self.assertEqual(encoder.summary().set_index("feature").loc["category", "rare_category_count"], 1)

    def test_infeasible_constraints_are_audited_and_invalid_targets_rejected(self):
        encoder = WOEEncoder(min_bad=20, min_good=20).fit(pd.DataFrame({"x": [0, 1, 2, 3]}), [0, 0, 0, 1])
        self.assertGreater(encoder.summary().iloc[0].constraint_violations, 0)
        with self.assertRaises(ValueError):
            WOEEncoder().fit(pd.DataFrame({"x": [0, 1]}), [1, 1])
        with self.assertRaises(ValueError):
            WOEEncoder().fit(pd.DataFrame({"x": [0, 1]}), [0, np.nan])

    def test_tied_numeric_values_have_contiguous_exported_intervals(self):
        frame = pd.DataFrame({"x": [0] * 100 + [1] * 100})
        target = [1] * 20 + [0] * 80 + [1] * 80 + [0] * 20
        encoder = WOEEncoder().fit(frame, target)
        _, coarse = encoder.tables()
        regular = coarse[coarse.bin_id.str.startswith("B")]
        self.assertEqual(len(regular), 2)
        self.assertEqual(regular.iloc[0].upper, regular.iloc[1].lower)
        ids = encoder.bin_ids(pd.DataFrame({"x": [-1, 0, .5, 1, 2]}))["x"].tolist()
        self.assertEqual(ids, ["B00", "B00", "B01", "B01", "B01"])


if __name__ == "__main__":
    unittest.main()
