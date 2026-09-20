"""Boundary, direction and support tests for frozen risk-grade evaluation."""

import json
import unittest

import numpy as np

from src.models.evaluation import (
    assign_grades, calibration_table, classification_metrics,
    contribution_variance, episode_variance, fit_grades,
    population_stability, score_points,
)


class EvaluationTests(unittest.TestCase):
    def test_perfect_and_reversed_probability_direction(self):
        good = classification_metrics([0, 0, 1, 1], [.1, .2, .8, .9])
        self.assertEqual(good["roc_auc"], 1)
        self.assertEqual(good["ks"], 1)
        self.assertEqual(good["pr_auc"], 1)
        self.assertEqual(good["average_precision"], 1)
        self.assertAlmostEqual(good["brier"], .025)
        reversed_metrics = classification_metrics([0, 0, 1, 1], [.9, .8, .2, .1])
        self.assertEqual(reversed_metrics["roc_auc"], 0)
        self.assertEqual(reversed_metrics["ks"], 1)  # KS is unsigned; AUC gives direction.

    def test_ties_count_together_and_pr_auc_is_not_ap(self):
        metrics = classification_metrics([0, 1, 0, 1], [.5] * 4)
        self.assertEqual(metrics["roc_auc"], .5)
        self.assertEqual(metrics["ks"], 0)
        self.assertEqual(metrics["average_precision"], .5)
        self.assertEqual(metrics["pr_auc"], .75)

    def test_one_class_and_empty_are_json_serializable(self):
        for y, p in [([0, 0], [.1, .2]), ([1], [.9]), ([], [])]:
            result = classification_metrics(y, p)
            self.assertIsNone(result["roc_auc"])
            self.assertIsNone(result["ks"])
            json.dumps(result, allow_nan=False)

    def test_score_odds_and_pdo(self):
        values = score_points([1 / 41, 1 / 21, 1 / 11])
        np.testing.assert_allclose(values, [650, 600, 550])
        self.assertTrue(np.isfinite(score_points([0, 1])).all())

    def test_calibration_preserves_ties_and_totals(self):
        table = calibration_table([0, 1, 0, 1], [.2, .2, .8, .8], 10)
        self.assertEqual(len(table), 2)
        self.assertEqual(table["count"].sum(), 4)
        self.assertEqual(table["bad_count"].sum(), 2)
        self.assertEqual(len(calibration_table([0, 1], [.1, .1])), 1)

    @staticmethod
    def validation():
        probabilities, outcomes = [], []
        for count_bad, probability in [(10, .05), (20, .1), (40, .2), (80, .4), (120, .6)]:
            probabilities.extend([probability] * 200)
            outcomes.extend([1] * count_bad + [0] * (200 - count_bad))
        return np.array(outcomes), np.array(probabilities)

    def test_grades_have_increasing_bad_rates_and_match_assignment(self):
        y, p = self.validation()
        definition = fit_grades(y, p)
        self.assertEqual(len(definition["grades"]), 5)
        rates = [row["actual_bad_rate"] for row in definition["validation_table"]]
        self.assertTrue((np.diff(rates) > 0).all())
        assigned = assign_grades(p, definition)
        for row in definition["validation_table"]:
            mask = assigned == row["grade"]
            self.assertEqual(mask.sum(), row["count"])
            self.assertEqual(y[mask].sum(), row["bad_count"])
            self.assertTrue(row["support_ok"])
        json.dumps(definition, allow_nan=False)

    def test_grade_boundary_goes_to_riskier_grade_and_support_reduces_grades(self):
        y, p = self.validation()
        definition = fit_grades(y, p, min_bad=50)
        self.assertLess(len(definition["grades"]), 5)
        assigned = assign_grades([0, *definition["probability_cuts"], 1], definition)
        self.assertEqual(list(assigned), [*definition["grades"], definition["grades"][-1]])
        self.assertTrue(all(row["bad_count"] >= 50 for row in definition["validation_table"]))

    def test_reversed_risk_evidence_and_constant_predictions_do_not_force_five_grades(self):
        y, p = self.validation()
        reversed_definition = fit_grades(1 - y, p)
        self.assertEqual(reversed_definition["grades"], ["R1"])
        constant = fit_grades([0, 1, 0, 1], [.5] * 4)
        self.assertEqual(constant["grades"], ["R1"])
        self.assertTrue(constant["warnings"])
        self.assertFalse(constant["validation_table"][0]["support_ok"])
        one_class = fit_grades([0] * 20, np.linspace(0, 1, 20))
        self.assertEqual(one_class["grades"], ["R1"])
        self.assertTrue(any("one outcome" in warning for warning in one_class["warnings"]))

    def test_psi_includes_new_and_missing_bins(self):
        total, table = population_stability(["A", "A", None], ["A", "B", None])
        self.assertGreater(total, 0)
        self.assertEqual(set(table["bin"]), {"value:A", "value:B", "missing:"})
        self.assertAlmostEqual(table["reference_share"].sum(), 1)
        self.assertAlmostEqual(table["current_share"].sum(), 1)
        self.assertEqual(population_stability(["A", "B"], ["A", "B"])[0], 0)

    def test_decompositions_reconcile_and_expose_covariance(self):
        result = episode_variance([1, 3, 5, 7], ["A", "A", "B", "B"])
        self.assertAlmostEqual(result["total_variance"], 5)
        self.assertAlmostEqual(result["within_episode_variance"], 1)
        self.assertAlmostEqual(result["between_episode_variance"], 4)
        shares = contribution_variance([1, 2, 3], [2, 4, 6])
        self.assertAlmostEqual(shares["total_variance"], shares["individual_variance"] +
                               shares["market_variance"] + shares["twice_covariance"])
        self.assertAlmostEqual(shares["individual_allocated_share"] + shares["market_allocated_share"], 1)

    def test_invalid_input_is_rejected(self):
        for y, p in [([0, None], [.1, .2]), ([0], [1.1]), ([0], [.1, .2])]:
            with self.assertRaises(ValueError):
                classification_metrics(y, p)
        with self.assertRaises(ValueError):
            fit_grades([], [])
        with self.assertRaises(ValueError):
            population_stability([], ["A"])


if __name__ == "__main__":
    unittest.main()
