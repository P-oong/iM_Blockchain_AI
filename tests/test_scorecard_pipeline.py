"""Time-leakage boundaries and portable score/probability consistency."""

import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.models.config import ScorecardConfig
from src.models.artifacts import Reports
from src.models.scorecard import ScorecardModel
from src.models.train_scorecard import assign_splits, load_source
from src.models.woe import WOEEncoder


class SourceValidationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.source = self.directory / "source.csv"
        self.metadata = self.directory / "labeling.metadata.json"
        self.metadata.write_text(json.dumps({"feature_allowlist": [
            "업력_연수", "구", "읍면동", "업종",
        ]}, ensure_ascii=False), encoding="utf-8")
        self.reports = Reports(self.directory / "reports")

    @staticmethod
    def business(identifier, survived):
        return {"관리번호": identifier, "t0": "2019-03-31", "기준분기": "2019Q1",
                "구": "중구", "읍면동": "성내1동", "업종": "일반음식점",
                "crisis_flag": 1, "episode_start": 1, "업력_연수": 3.5,
                "Y_12M": survived, "horizon_end_12M": "2020-03-31",
                "label_status_12M": "observed"}

    def load(self, rows, **overrides):
        pd.DataFrame(rows).to_csv(self.source, index=False, encoding="utf-8-sig")
        config = ScorecardConfig(source=str(self.source), metadata=str(self.metadata), **overrides)
        return load_source(config, self.reports)

    def test_exact_duplicates_removed_before_model_bad_rate(self):
        bad = self.business("001", 0)
        good = self.business("002", 1)
        frame, candidates, audit = self.load([bad, bad.copy(), bad.copy(), good])
        self.assertEqual(audit["source_rows"], 4)
        self.assertEqual(audit["exact_duplicate_rows_removed"], 2)
        self.assertEqual(audit["unique_businesses"], 2)
        self.assertAlmostEqual(audit["raw_weighted_bad_rate"], .75)
        self.assertAlmostEqual(audit["deduplicated_bad_rate"], .5)
        self.assertEqual(frame["관리번호"].tolist(), ["001", "002"])
        self.assertEqual(frame["BAD"].tolist(), [1, 0])
        self.assertTrue(frame["eligible"].all())
        self.assertNotIn("읍면동", candidates)

    def test_conflicting_same_id_rejected_even_in_nonmodel_column(self):
        original = dict(self.business("001", 0), source_note="first")
        conflicting = dict(original, source_note="different")
        with self.assertRaisesRegex(ValueError, "Conflicting or non-identical"):
            self.load([original, conflicting])

    def test_missing_requested_horizon_is_not_inferred(self):
        with self.assertRaisesRegex(ValueError, "Missing required columns.*Y_24M"):
            self.load([self.business("001", 1)], horizon_months=24)

    def test_enrichment_published_after_baseline_is_rejected(self):
        future = dict(self.business("001", 1), 주민인구=10000, 인구_공표일="2019-04-01")
        with self.assertRaisesRegex(ValueError, "availability must be known and <= t0"):
            self.load([future], extra_feature_availability={"주민인구": "인구_공표일"})


class ChronologicalSplitTests(unittest.TestCase):
    def test_purges_outcomes_at_or_after_next_period_boundary(self):
        frame = pd.DataFrame({
            "t0": pd.to_datetime([
                "2020-12-31", "2021-01-01", "2021-06-30", "2022-01-01",
                "2022-12-31", "2022-12-31", "2023-01-01", "2024-01-01",
                "2024-12-31", "2025-01-01", "2020-12-31",
            ]),
            "horizon_end": pd.to_datetime([
                "2021-12-31", "2022-01-01", "2022-06-30", "2023-01-01",
                "2023-12-31", "2024-01-01", "2024-01-01", "2025-01-01",
                "2025-12-31", "2026-01-01", "2021-12-31",
            ]),
            "eligible": [True] * 9 + [False, False],
            "Y_12M": [1, 0, 1, 0, 1, 0, 1, 0, 1, None, None],
        }, index=[f"business_{index}" for index in range(11)])
        result = assign_splits(frame, ScorecardConfig())
        self.assertEqual(result.tolist(), [
            "train", "embargo", "embargo", "validation", "validation",
            "embargo", "embargo", "oot", "oot", "ineligible", "ineligible",
        ])
        self.assertTrue(result.index.equals(frame.index))
        self.assertTrue((frame.loc[result == "train", "horizon_end"] < pd.Timestamp("2022-01-01")).all())
        self.assertTrue((frame.loc[result == "validation", "horizon_end"] < pd.Timestamp("2024-01-01")).all())

    def test_split_assignment_does_not_depend_on_outcome(self):
        frame = pd.DataFrame({
            "t0": pd.to_datetime(["2020-01-01", "2022-01-01", "2024-01-01"]),
            "horizon_end": pd.to_datetime(["2021-01-01", "2023-01-01", "2025-01-01"]),
            "eligible": [True, True, True], "Y_12M": [0, 0, 0],
        })
        before = assign_splits(frame, ScorecardConfig())
        frame["Y_12M"] = 1
        pd.testing.assert_series_equal(before, assign_splits(frame, ScorecardConfig()))


class PortableScorecardTests(unittest.TestCase):
    def make_model(self, *, slope=.8, calibration_intercept=-.4, intercept=-.3):
        train = pd.DataFrame({"x": [0] * 100 + [1] * 100,
                              "category": ["A", "B"] * 100})
        bad = [1] * 20 + [0] * 80 + [1] * 80 + [0] * 20
        encoder = WOEEncoder().fit(train, bad, categorical=["category"])
        return ScorecardModel(
            encoder=encoder, features=["x", "category"], coefficients=[1.2, .4],
            intercept=intercept,
            calibration={"slope": slope, "intercept": calibration_intercept, "method": "platt"},
            grade_definition={"probability_cuts": [.1, .4], "grades": ["R1", "R2", "R3"]},
            scaling={"base_score": 600., "base_odds": 20., "pdo": 50.},
            horizon_months=12,
        )

    def holdout(self):
        return pd.DataFrame({"x": [0, 1, 100, np.nan],
                             "category": ["A", "A", "NEVER_SEEN", None]},
                            index=[11, 21, 31, 41])

    def test_calibration_and_scores_reconstruct_independently(self):
        model = self.make_model()
        frame = self.holdout()
        predictions = model.predict(frame)
        # WOE values come from a separate equivalent fit, avoiding use of the
        # model's probability or contribution calculations in the expectation.
        train = pd.DataFrame({"x": [0] * 100 + [1] * 100,
                              "category": ["A", "B"] * 100})
        encoder = WOEEncoder().fit(train, [1] * 20 + [0] * 80 + [1] * 80 + [0] * 20,
                                   categorical=["category"])
        raw_logit = encoder.transform(frame).to_numpy() @ np.array([1.2, .4]) - .3
        calibrated_logit = .8 * raw_logit - .4
        expected_bad = 1 / (1 + np.exp(-calibrated_logit))
        np.testing.assert_allclose(predictions["p_bad_raw"], 1 / (1 + np.exp(-raw_logit)))
        np.testing.assert_allclose(predictions["p_bad"], expected_bad)
        np.testing.assert_allclose(predictions["p_survival"], 1 - expected_bad)
        np.testing.assert_allclose(predictions["recovery_score"], 100 * (1 - expected_bad))
        expected_score = 600 + 50 / math.log(2) * (np.log((1 - expected_bad) / expected_bad) - math.log(20))
        np.testing.assert_allclose(predictions["score"], expected_score)
        contributions = model.contributions(frame)
        np.testing.assert_allclose(contributions.sum(axis=1), predictions["score"])
        self.assertLess(predictions.loc[11, "p_bad"], predictions.loc[21, "p_bad"])
        self.assertGreater(predictions.loc[11, "score"], predictions.loc[21, "score"])
        self.assertTrue(predictions.index.equals(frame.index))

    def test_portable_json_model_round_trip_including_unseen_and_missing(self):
        model = self.make_model()
        frame = self.holdout()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scorecard.json"
            model.save(path)
            restored = ScorecardModel.load(path)
            pd.testing.assert_frame_equal(model.predict(frame), restored.predict(frame))
            pd.testing.assert_frame_equal(model.contributions(frame), restored.contributions(frame))
        self.assertTrue(np.isfinite(model.predict(frame).select_dtypes(include=[np.number])).all().all())

    def test_doubling_good_bad_odds_increases_score_by_pdo(self):
        frame = self.holdout()
        initial = self.make_model(slope=1, calibration_intercept=0, intercept=0).predict(frame)
        doubled = self.make_model(slope=1, calibration_intercept=0, intercept=-math.log(2)).predict(frame)
        np.testing.assert_allclose(doubled["score"] - initial["score"], 50.)
        initial_odds = (1 - initial["p_bad"]) / initial["p_bad"]
        doubled_odds = (1 - doubled["p_bad"]) / doubled["p_bad"]
        np.testing.assert_allclose(doubled_odds, initial_odds * 2)


if __name__ == "__main__":
    unittest.main()
