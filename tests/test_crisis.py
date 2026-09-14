"""Outcome-independent crisis detection and incomplete-observation behavior."""

import csv
import math
import tempfile
import unittest
from pathlib import Path

from src.features.crisis import CrisisConfig, SALES_COLUMNS, load_sales


class CrisisTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / "sales.csv"

    def load(self, rows, config=None):
        with self.path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(SALES_COLUMNS)
            for quarter, area, sales in rows:
                writer.writerow([quarter, "중구", area, "일반음식점", sales, 1])
        return load_sales(self.path, config or CrisisConfig())

    @staticmethod
    def market(rows, area="성내1동"):
        return [row for row in rows if row.group[1] == area]

    def test_peer_growth_uses_matched_sales_totals_not_mean_growth(self):
        rows = self.load([
            ("2023Q1", "성내1동", 100), ("2024Q1", "성내1동", 70),
            ("2023Q1", "성내2동", 900), ("2024Q1", "성내2동", 900),
            ("2024Q1", "성내3동", 100000),  # No prior-year coverage; excluded from both totals.
            ("2023Q1", "동인동", 100000),  # No current-year coverage; likewise excluded.
        ])
        row = self.market(rows)[-1]
        self.assertAlmostEqual(row.peer_yoy, -.03)
        self.assertAlmostEqual(row.sales_yoy, -.30)
        self.assertAlmostEqual(row.peer_gap, -.27)
        self.assertTrue(row.crisis)
        self.assertTrue(row.episode_start)

    def test_inclusive_thresholds_handle_floating_point_boundary(self):
        rows = self.load([
            ("2023Q1", "성내1동", 100), ("2024Q1", "성내1동", 80),
            ("2023Q1", "성내2동", 100), ("2024Q1", "성내2동", 100),
        ])
        row = self.market(rows)[-1]
        self.assertAlmostEqual(row.sales_yoy, -.20)
        self.assertAlmostEqual(row.peer_gap, -.10)
        self.assertTrue(row.crisis)

    def test_rule_and_or_and_yoy_only_are_adjustable(self):
        observations = [
            ("2023Q1", "성내1동", 100), ("2024Q1", "성내1동", 75),
            ("2023Q1", "성내2동", 100), ("2024Q1", "성내2동", 75),
        ]
        self.assertFalse(self.market(self.load(observations))[-1].crisis)
        for rule in ("or", "yoy-only"):
            self.assertTrue(self.market(self.load(observations, CrisisConfig(rule=rule)))[-1].crisis)
        stricter = CrisisConfig(yoy_threshold=-.30, rule="yoy-only")
        self.assertFalse(self.market(self.load(observations, stricter))[-1].crisis)

    def test_yoy_requires_exact_four_quarter_lag(self):
        rows = self.load([
            ("2022Q4", "성내1동", 100), ("2023Q2", "성내1동", 100),
            ("2023Q3", "성내1동", 100), ("2023Q4", "성내1동", 100),
            ("2024Q1", "성내1동", 50),
        ], CrisisConfig(rule="yoy-only"))
        self.assertIsNone(rows[-1].sales_yoy)
        self.assertIsNone(rows[-1].crisis)
        self.assertFalse(rows[-1].episode_start)

    def test_missing_nonfinite_and_zero_lag_remain_unknown(self):
        for value in ("", "NaN", "Infinity", "-Infinity", "null", 0):
            with self.subTest(value=value):
                rows = self.load([
                    ("2023Q1", "성내1동", value), ("2024Q1", "성내1동", 50),
                ])
                self.assertIsNone(rows[-1].sales_yoy)
                self.assertIsNone(rows[-1].crisis)
                self.assertFalse(rows[-1].episode_start)

    def test_zero_current_sales_is_a_valid_total_loss(self):
        rows = self.load([
            ("2023Q1", "성내1동", 100), ("2024Q1", "성내1동", 0),
        ], CrisisConfig(rule="yoy-only"))
        self.assertEqual(rows[-1].sales_yoy, -1)
        self.assertTrue(rows[-1].crisis)

    def test_negative_and_duplicate_market_quarter_are_rejected(self):
        for observations in (
            [("2023Q1", "성내1동", -1)],
            [("2023Q1", "성내1동", 100), ("2023Q1", "성내1동", 100)],
        ):
            with self.subTest(observations=observations), self.assertRaises(ValueError):
                self.load(observations)

    def test_consecutive_rule_emits_entry_at_confirmation_once(self):
        observations = [(f"2023Q{q}", "성내1동", 100) for q in range(1, 5)]
        observations += [(f"2024Q{q}", "성내1동", 50) for q in range(1, 5)]
        rows = self.load(observations, CrisisConfig(rule="yoy-only", min_consecutive_quarters=2))[-4:]
        self.assertEqual([row.crisis for row in rows], [False, True, True, True])
        self.assertEqual([row.episode_start for row in rows], [False, True, False, False])

    def test_gap_does_not_manufacture_episode_after_known_history(self):
        observations = [(f"2023Q{q}", "성내1동", 100) for q in range(1, 5)]
        observations += [
            ("2024Q1", "성내1동", 100),
            ("2024Q3", "성내1동", 50), ("2024Q4", "성내1동", 100),
            ("2025Q1", "성내1동", 50),
        ]
        rows = self.load(observations, CrisisConfig(rule="yoy-only"))
        self.assertTrue(rows[-3].crisis)
        self.assertFalse(rows[-3].episode_start)
        self.assertTrue(rows[-1].episode_start)

    def test_unknown_quarter_resets_consecutive_run_and_entry_evidence(self):
        observations = [(f"2023Q{q}", "성내1동", 100) for q in range(1, 5)]
        observations += [
            ("2024Q1", "성내1동", 50), ("2024Q2", "성내1동", "NaN"),
            ("2024Q3", "성내1동", 50), ("2024Q4", "성내1동", 50),
        ]
        rows = self.load(observations, CrisisConfig(rule="yoy-only", min_consecutive_quarters=2))[-4:]
        self.assertEqual([row.crisis for row in rows], [False, None, False, True])
        self.assertEqual([row.episode_start for row in rows], [False, False, False, False])

    def test_known_normal_allows_another_episode(self):
        observations = [(f"2023Q{q}", "성내1동", 100) for q in range(1, 5)]
        observations += [
            ("2024Q1", "성내1동", 50), ("2024Q2", "성내1동", 50),
            ("2024Q3", "성내1동", 100), ("2024Q4", "성내1동", 50),
        ]
        rows = self.load(observations, CrisisConfig(rule="yoy-only"))[-4:]
        self.assertEqual([row.episode_start for row in rows], [True, False, False, True])

    def test_config_rejects_nonfinite_thresholds_and_invalid_run_lengths(self):
        for kwargs in (
            {"yoy_threshold": math.nan}, {"peer_gap_threshold": math.inf},
            {"yoy_threshold": "-0.2"}, {"rule": "xor"},
            {"min_consecutive_quarters": 0}, {"min_consecutive_quarters": 1.5},
            {"min_consecutive_quarters": True},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                CrisisConfig(**kwargs)


if __name__ == "__main__":
    unittest.main()
