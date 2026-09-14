"""Synthetic sales must stay reproducible and independent of outcome labels."""

import csv
import tempfile
import unittest
from pathlib import Path

from src.data.generate_synthetic_sales import (
    Config,
    generate_sales,
    load_source,
    summarize,
)


class SyntheticSalesTests(unittest.TestCase):
    def setUp(self):
        self.temp_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_directory.cleanup)
        self.directory = Path(self.temp_directory.name)
        self.groups = [
            {"구": "중구", "읍면동": "성내1동", "업종": "일반음식점"},
            {"구": "동구", "읍면동": "신암1동", "업종": "일반음식점"},
            {"구": "북구", "읍면동": "산격1동", "업종": "미용업"},
        ]

    def source(self, rows, *, name="source.csv", use_competition=False):
        path = self.directory / name
        fields = list(dict.fromkeys(key for row in rows for key in row))
        with path.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        return load_source(path, use_competition=use_competition)

    def test_row_order_and_duplicate_businesses_do_not_change_sales(self):
        source = self.source(self.groups)
        reordered = self.source(
            list(reversed(self.groups)) + [dict(self.groups[0])] * 3,
            name="reordered.csv",
        )
        self.assertEqual(
            generate_sales(source, Config()),
            generate_sales(reordered, Config()),
        )

    def test_closure_dates_and_labels_are_not_generation_inputs(self):
        first = self.source(
            [dict(group, 폐업일="2016-03-01", Y6M="0", Y12M="0")
             for group in self.groups]
        )
        second = self.source(
            [dict(group, 폐업일="", Y6M="1", Y12M="1")
             for group in self.groups],
            name="different_outcomes.csv",
        )
        self.assertEqual(
            generate_sales(first, Config()),
            generate_sales(second, Config()),
        )

    def test_default_panel_has_every_quarter_and_positive_integer_values(self):
        rows = generate_sales(self.source(self.groups), Config())
        expected_quarters = {
            f"{year}Q{quarter}"
            for year in range(2015, 2026)
            for quarter in range(1, 5)
        }
        actual = {}
        for row in rows:
            self.assertEqual(
                set(row), {"기준분기", "구", "읍면동", "업종", "매출액", "이용건수"}
            )
            key = (row["구"], row["읍면동"], row["업종"])
            quarters = actual.setdefault(key, set())
            self.assertNotIn(row["기준분기"], quarters)
            quarters.add(row["기준분기"])
            for column in ("매출액", "이용건수"):
                self.assertIs(type(row[column]), int)
                self.assertGreater(row[column], 0)
        self.assertEqual(len(rows), 3 * 44)
        self.assertEqual(
            set(actual),
            {(group["구"], group["읍면동"], group["업종"]) for group in self.groups},
        )
        for quarters in actual.values():
            self.assertEqual(quarters, expected_quarters)

    def test_changing_seed_changes_generated_sales(self):
        source = self.source(self.groups)
        first = generate_sales(source, Config(seed=42))
        second = generate_sales(source, Config(seed=43))
        self.assertNotEqual(
            [row["매출액"] for row in first],
            [row["매출액"] for row in second],
        )

    def test_changing_output_window_preserves_overlapping_sales(self):
        source = self.source(self.groups)
        full = generate_sales(source, Config())
        subset = generate_sales(source, Config(start_quarter="2020Q1", end_quarter="2023Q4"))
        self.assertEqual(subset, [row for row in full if "2020Q1" <= row["기준분기"] <= "2023Q4"])

    def test_numbered_dong_with_je_prefix_is_retained(self):
        source = self.source([{"구": "동구", "읍면동": "신암제1동", "업종": "일반음식점"}])
        self.assertEqual(source.groups, [("동구", "신암제1동", "일반음식점")])

    def test_building_names_and_numeric_building_tokens_are_not_area_keys(self):
        invalid_rows = [
            {"구": "중구", "읍면동": area, "업종": "일반음식점"}
            for area in ("100동", "상가동", "관리동")
        ]
        source = self.source(self.groups + invalid_rows)
        self.assertEqual(
            set(source.groups),
            {(group["구"], group["읍면동"], group["업종"]) for group in self.groups},
        )
        self.assertEqual(source.audit["excluded_rows"], len(invalid_rows))

    def test_groups_observed_only_after_source_cutoff_are_excluded(self):
        known = [dict(group, 기준연도="2025") for group in self.groups]
        future_only = {
            "구": "남구", "읍면동": "대명1동", "업종": "일반음식점", "기준연도": "2026"
        }
        source = self.source(known + [future_only])
        self.assertEqual(
            set(source.groups),
            {(group["구"], group["읍면동"], group["업종"]) for group in self.groups},
        )
        self.assertEqual(source.audit["source_cutoff_year"], 2025)
        self.assertEqual(source.audit["excluded_rows_by_reason"]["after_source_cutoff_year"], 1)

    def test_ambiguous_cross_district_competition_metrics_are_rejected(self):
        rows = [
            {
                "구": district, "읍면동": "동산동", "업종": "일반음식점",
                "기준연도": "2020", "동_동일업종수": "20", "동_동일업종_신규진입률": "0.1",
            }
            for district in ("중구", "동구")
        ]
        # The market keys remain distinct when ambiguous competition is unused.
        self.assertEqual(len(self.source(rows).groups), 2)
        with self.assertRaises(ValueError):
            self.source(rows, name="ambiguous_competition.csv", use_competition=True)

    def test_future_competition_snapshots_do_not_change_earlier_sales(self):
        past = [
            dict(group, 기준연도="2014", 동_동일업종수="20", 동_동일업종_신규진입률="0.1")
            for group in self.groups
        ]
        future = [
            dict(group, 기준연도="2022", 동_동일업종수="30", 동_동일업종_신규진입률="0.2")
            for group in self.groups
        ]
        changed_future = [
            dict(group, 기준연도="2022", 동_동일업종수="900", 동_동일업종_신규진입률="0.8")
            for group in self.groups
        ]
        first = self.source(past + future, use_competition=True)
        second = self.source(
            past + changed_future, name="changed_future.csv", use_competition=True
        )
        config = Config(use_competition=True)
        first_rows = generate_sales(first, config)
        second_rows = generate_sales(second, config)
        self.assertEqual(
            [row for row in first_rows if row["기준분기"] < "2023Q1"],
            [row for row in second_rows if row["기준분기"] < "2023Q1"],
        )
        self.assertNotEqual(
            [row for row in first_rows if row["기준분기"] >= "2023Q1"],
            [row for row in second_rows if row["기준분기"] >= "2023Q1"],
        )

    def test_invalid_quarter_ranges_and_probabilities_are_rejected(self):
        invalid_options = [
            {"start_quarter": "2025Q4", "end_quarter": "2025Q1"},
            {"start_quarter": "2020Q0"},
            {"end_quarter": "2025Q5"},
            {"start_quarter": "2020-01-01"},
            {"shock_probability": -0.01},
            {"shock_probability": 1.01},
            {"shock_probability": float("nan")},
        ]
        for options in invalid_options:
            with self.subTest(options=options), self.assertRaises(ValueError):
                Config(**options)

    def test_summary_uses_pooled_sales_within_each_industry(self):
        # Restaurant YoY is 922 / 1000 - 1 = -7.8%, not the -15% mean.
        # The small restaurant group therefore has a -16.2 percentage point gap.
        rows = []
        for quarter in ("2015Q1", "2015Q2", "2015Q3", "2015Q4", "2016Q1"):
            current = quarter == "2016Q1"
            values = (76, 846, 40) if current else (100, 900, 100)
            for group, sales in zip(self.groups, values):
                rows.append(dict(group, 기준분기=quarter, 매출액=sales, 이용건수=1))
        summary = summarize(rows)
        self.assertEqual(summary["rows"], 15)
        self.assertEqual(summary["groups"], 3)
        self.assertEqual(summary["yoy_eligible_rows"], 3)
        self.assertEqual(summary["crisis_rows"], 1)
        initial = summarize([row for row in rows if row["기준분기"] != "2016Q1"])
        self.assertEqual(initial["yoy_eligible_rows"], 0)
        self.assertEqual(initial["crisis_rows"], 0)

    def test_crisis_thresholds_include_exact_boundary(self):
        # Equal-sized groups: -20% and 0% produce city YoY -10%, gap -10%p.
        rows = []
        for quarter in ("2015Q1", "2015Q2", "2015Q3", "2015Q4", "2016Q1"):
            for index, group in enumerate(self.groups[:2]):
                sales = 80 if quarter == "2016Q1" and index == 0 else 100
                rows.append(dict(group, 기준분기=quarter, 매출액=sales, 이용건수=1))
        self.assertEqual(summarize(rows)["crisis_rows"], 1)


if __name__ == "__main__":
    unittest.main()
