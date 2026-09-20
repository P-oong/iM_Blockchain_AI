"""Check observable survival outcomes, cohort entry, and source data safeguards."""

import csv
import tempfile
import unittest
from datetime import date
from pathlib import Path

from src.data.businesses import Business, load_businesses
from src.data.labeling import (
    LabelConfig,
    add_months,
    build_cohort,
    label_distribution,
    label_outcome,
)
from src.features.crisis import MarketQuarter


GROUP = ("중구", "동인동", "일반음식점")


def business(business_id="A", *, opened=date(2020, 1, 1), closed=None, group=GROUP):
    return Business(business_id=business_id, group=group, opened=opened, closed=closed)


def market(year, quarter, *, crisis=True, episode_start=True, group=GROUP):
    return MarketQuarter(
        group=group,
        quarter=year * 4 + quarter - 1,
        sales=70,
        transactions=10,
        sales_yoy=-0.3,
        peer_yoy=-0.1,
        peer_gap=-0.2,
        crisis=crisis,
        episode_start=episode_start,
    )


class OutcomeTests(unittest.TestCase):
    def setUp(self):
        self.opened = date(2020, 1, 1)
        self.t0 = date(2024, 6, 30)
        self.observation_end = date(2025, 12, 31)

    def outcome(self, closed, *, months=6, observation_end=None):
        return label_outcome(
            self.opened,
            closed,
            self.t0,
            months,
            observation_end or self.observation_end,
        )

    def test_quarter_end_horizons_preserve_month_end(self):
        examples = [
            (date(2024, 6, 30), 6, date(2024, 12, 31)),
            (date(2024, 9, 30), 6, date(2025, 3, 31)),
            (date(2024, 2, 29), 12, date(2025, 2, 28)),
            (date(2023, 2, 28), 12, date(2024, 2, 29)),
            (date(2024, 8, 30), 6, date(2025, 2, 28)),
            (date(2024, 6, 15), 6, date(2024, 12, 15)),
        ]
        for start, months, expected in examples:
            with self.subTest(start=start, months=months):
                self.assertEqual(add_months(start, months), expected)

    def test_closure_on_horizon_endpoint_is_failure(self):
        self.assertEqual(self.outcome(date(2024, 12, 31)), 0)

    def test_closure_after_horizon_endpoint_is_survival(self):
        self.assertEqual(self.outcome(date(2025, 1, 1)), 1)
        self.assertEqual(self.outcome(None), 1)

    def test_same_business_can_survive_six_but_fail_twelve_months(self):
        closed = date(2025, 3, 10)
        self.assertEqual(self.outcome(closed, months=6), 1)
        self.assertEqual(self.outcome(closed, months=12), 0)

    def test_closed_or_unopened_at_baseline_is_not_in_population(self):
        for opened, closed in (
            (self.opened, self.t0),
            (self.opened, date(2024, 6, 29)),
            (date(2024, 7, 1), None),
        ):
            with self.subTest(opened=opened, closed=closed), self.assertRaises(ValueError):
                label_outcome(opened, closed, self.t0, 6, self.observation_end)

    def test_business_opening_on_baseline_is_eligible(self):
        self.assertEqual(
            label_outcome(self.t0, None, self.t0, 6, self.observation_end), 1
        )

    def test_incomplete_horizon_is_censored_even_with_early_known_closure(self):
        for closed in (None, date(2024, 10, 1)):
            with self.subTest(closed=closed):
                self.assertIsNone(
                    self.outcome(closed, observation_end=date(2024, 12, 30))
                )

    def test_observation_end_equal_to_horizon_is_sufficient(self):
        self.assertEqual(
            self.outcome(None, observation_end=date(2024, 12, 31)), 1
        )


class SourceBusinessTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)

    @staticmethod
    def row(**changes):
        result = {
            "관리번호": "A",
            "구": GROUP[0],
            "읍면동": GROUP[1],
            "업종": GROUP[2],
            "인허가일자": "2020-01-01",
            "폐업일자": "",
            "기준연도": "2024",
            "기준연도말_폐업상태": "0",
        }
        result.update(changes)
        return result

    def source(self, rows):
        path = self.directory / "source.csv"
        fields = list(dict.fromkeys(key for row in rows for key in row))
        with path.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        return load_businesses(path)

    def test_annual_duplicates_produce_one_business_and_one_cohort_row(self):
        loaded = self.source([
            self.row(기준연도="2023"),
            self.row(기준연도="2024"),
            self.row(기준연도="2025"),
        ])
        self.assertEqual(len(loaded.businesses), 1)
        self.assertEqual(loaded.businesses[0].business_id, "A")
        rows, _ = build_cohort(
            loaded.businesses,
            [market(2024, 2)],
            LabelConfig(observation_end=date(2025, 12, 31)),
        )
        self.assertEqual(len(rows), 1)

    def test_closed_flag_without_closure_date_is_excluded(self):
        loaded = self.source([self.row(기준연도말_폐업상태="1")])
        self.assertEqual(loaded.businesses, [])
        self.assertTrue(loaded.exclusions)

    def test_later_snapshot_can_supply_a_previously_blank_closure_date(self):
        rows = [
            self.row(기준연도="2023"),
            self.row(
                기준연도="2024", 폐업일자="2024-10-01", 기준연도말_폐업상태="1"
            ),
        ]
        first = self.source(rows)
        reversed_source = self.source(list(reversed(rows)))
        self.assertEqual(len(first.businesses), 1)
        self.assertEqual(first.businesses[0].closed, date(2024, 10, 1))
        self.assertEqual(first.businesses, reversed_source.businesses)
        self.assertEqual(first.exclusions, [])

    def test_missing_or_invalid_dates_are_excluded(self):
        cases = [
            {"인허가일자": ""},
            {"인허가일자": "2020-02-30"},
            {"폐업일자": "2024-02-30"},
            {"폐업일자": "2019-12-31"},
        ]
        for changes in cases:
            with self.subTest(changes=changes):
                loaded = self.source([self.row(**changes)])
                self.assertEqual(loaded.businesses, [])
                self.assertTrue(loaded.exclusions)

    def test_numeric_building_dong_is_excluded_without_guessing_an_area(self):
        loaded = self.source([self.row(읍면동="103동")])
        self.assertEqual(loaded.businesses, [])
        self.assertTrue(loaded.exclusions)

    def test_conflicting_repeated_business_records_are_excluded(self):
        cases = [
            (self.row(), self.row(인허가일자="2021-01-01")),
            (self.row(), self.row(읍면동="삼덕동")),
            (
                self.row(폐업일자="2024-10-01", 기준연도말_폐업상태="1"),
                self.row(폐업일자="2024-11-01", 기준연도말_폐업상태="1"),
            ),
        ]
        for rows in cases:
            with self.subTest(rows=rows):
                loaded = self.source(rows)
                self.assertEqual(loaded.businesses, [])
                self.assertTrue(loaded.exclusions)


class CohortTests(unittest.TestCase):
    def setUp(self):
        self.config = LabelConfig(observation_end=date(2025, 12, 31), horizons=(6, 12))

    def test_defaults_are_12_18_24_with_no_six_month_label(self):
        config = LabelConfig(observation_end=date(2025, 12, 31))
        self.assertEqual(config.horizons, (12, 18, 24))
        rows, _ = build_cohort([business(closed=date(2025, 3, 10))], [market(2024, 2)], config)
        self.assertNotIn("Y_6M", rows[0])
        self.assertEqual(rows[0]["Y_12M"], 0)
        self.assertEqual(rows[0]["Y_18M"], 0)
        self.assertIsNone(rows[0]["Y_24M"])

    def test_first_eligible_episode_only_and_annual_horizons(self):
        markets = [
            market(2025, 1),
            market(2024, 3, episode_start=False),
            market(2024, 2),
        ]
        rows, _ = build_cohort(
            [business(closed=date(2025, 3, 10))], markets, self.config
        )
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["기준분기"], "2024Q2")
        self.assertEqual(str(rows[0]["t0"]), "2024-06-30")
        self.assertEqual(rows[0]["Y_6M"], 1)
        self.assertEqual(rows[0]["Y_12M"], 0)

    def test_business_opened_during_ongoing_crisis_waits_for_next_episode(self):
        markets = [
            market(2024, 2),
            market(2024, 3, episode_start=False),
            market(2024, 4, crisis=False, episode_start=False),
            market(2025, 1),
        ]
        rows, _ = build_cohort(
            [business(opened=date(2024, 7, 1))], markets, self.config
        )
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["기준분기"], "2025Q1")
        self.assertEqual(rows[0]["Y_6M"], 1)
        self.assertIsNone(rows[0]["Y_12M"])

    def test_first_exposure_mode_accepts_an_ongoing_episode(self):
        config = LabelConfig(
            observation_end=date(2025, 12, 31), entry_mode="first-exposure"
        )
        markets = [market(2024, 2), market(2024, 3, episode_start=False)]
        rows, _ = build_cohort(
            [business(opened=date(2024, 7, 1))], markets, config
        )
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["기준분기"], "2024Q3")

    def test_unmatched_market_and_inactive_businesses_are_excluded(self):
        businesses = [
            business("closed", closed=date(2024, 6, 30)),
            business("future", opened=date(2024, 7, 1)),
            business("unmatched", group=("중구", "103동", "일반음식점")),
        ]
        rows, excluded = build_cohort(businesses, [market(2024, 2)], self.config)
        self.assertEqual(rows, [])
        self.assertEqual(len(excluded), len(businesses))

    def test_baseline_after_observation_cutoff_is_not_a_cohort_row(self):
        rows, excluded = build_cohort(
            [business()], [market(2026, 1)], self.config
        )
        self.assertEqual(rows, [])
        self.assertEqual(len(excluded), 1)

    def test_no_crisis_does_not_create_negative_or_survival_samples(self):
        rows, excluded = build_cohort(
            [business()],
            [market(2024, 2, crisis=False, episode_start=False)],
            self.config,
        )
        self.assertEqual(rows, [])
        self.assertEqual(len(excluded), 1)

    def test_custom_horizons_change_only_requested_label_columns(self):
        config = LabelConfig(
            observation_end=date(2026, 12, 31), horizons=(3, 24)
        )
        rows, _ = build_cohort(
            [business(closed=date(2024, 11, 1))], [market(2024, 2)], config
        )
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["Y_3M"], 1)
        self.assertEqual(rows[0]["Y_24M"], 0)
        self.assertNotIn("Y_6M", rows[0])
        self.assertNotIn("Y_12M", rows[0])

    def test_distribution_uses_only_observed_labels_in_outcome_ratios(self):
        rows = [
            {"Y_6M": 1}, {"Y_6M": 1}, {"Y_6M": 0}, {"Y_6M": None}
        ]
        result = label_distribution(rows, 6)
        self.assertEqual(result["total_cohort"], 4)
        self.assertEqual(result["labeled"], 3)
        self.assertEqual(result["survived_1"], 2)
        self.assertEqual(result["closed_0"], 1)
        self.assertEqual(result["unobserved_na"], 1)
        self.assertAlmostEqual(result["survival_pct_among_labeled"], 200 / 3)
        self.assertAlmostEqual(result["closure_pct_among_labeled"], 100 / 3)
        self.assertEqual(result["na_pct_among_cohort"], 25)
        empty = label_distribution([{"Y_6M": None}], 6)
        self.assertIsNone(empty["survival_pct_among_labeled"])
        self.assertIsNone(empty["closure_pct_among_labeled"])


if __name__ == "__main__":
    unittest.main()
