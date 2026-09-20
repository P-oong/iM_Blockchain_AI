import csv
import tempfile
import unittest
from datetime import date
from pathlib import Path

from src.data.businesses import load_businesses
from src.data.build_labeled_dataset import run_pipeline
from src.data.generate_synthetic_sales import Config, load_source, save_generation
from src.data.labeling import LabelConfig
from src.data.provenance import find_input_csv
from src.features.crisis import CrisisConfig


class CorrectedSourceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.input = self.root / "raw.csv"
        self.rows = [{"관리번호": str(i), "기준연도": "2025", "구": "서구", "행정동": "비산7동",
                      "행정동코드": "2717061000.0", "업종": "음식점", "인허가일자": "2020-01-01",
                      "폐업일자": ""} for i in range(21)]

    def write(self):
        with self.input.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(self.rows[0]))
            writer.writeheader()
            writer.writerows(self.rows)

    def test_admin_columns_replace_legacy_area_and_decimal_codes_stay_strings(self):
        self.write()
        businesses = load_businesses(self.input)
        source = load_source(self.input)
        self.assertEqual(businesses.audit["geography"]["area_column"], "행정동")
        self.assertEqual(businesses.businesses[0].area_code, "2717061000")
        self.assertEqual(source.area_codes[("서구", "비산7동", "음식점")], "2717061000")

    def test_minor_district_error_is_normalized_consistently_and_recorded(self):
        self.rows[-1]["구"] = "북구"
        self.write()
        businesses = load_businesses(self.input)
        source = load_source(self.input)
        self.assertEqual({b.group for b in businesses.businesses}, {("서구", "비산7동", "음식점")})
        self.assertEqual(source.groups, [("서구", "비산7동", "음식점")])
        self.assertEqual(businesses.audit["geography"]["corrections"][0]["records"], 1)

    def test_ambiguous_code_does_not_trigger_guessing(self):
        for row in self.rows[:10]:
            row["구"] = "북구"
        self.write()
        source = load_businesses(self.input)
        self.assertEqual(source.businesses, [])
        self.assertEqual(len(source.exclusions), 21)

    def test_source_change_blocks_stale_synthetic_sales(self):
        self.write()
        sales = self.root / "sales.csv"
        save_generation(load_source(self.input), Config(), sales, self.input)
        self.rows[0]["인허가일자"] = "2021-01-01"
        self.write()
        with self.assertRaisesRegex(ValueError, "regenerate-sales"):
            run_pipeline(self.input, sales, self.root / "output", CrisisConfig(), LabelConfig(date(2025, 12, 31)))

    def test_excel_owner_file_is_not_an_input_csv(self):
        self.write()
        (self.root / "~$raw.csv").write_bytes(b"Excel lock")
        self.assertEqual(find_input_csv(self.root), self.input)

    def test_short_administrative_name_padong_is_preserved(self):
        for row in self.rows:
            row.update({"구": "수성구", "행정동": "파동", "행정동코드": "2726063000.0"})
        self.write()
        self.assertEqual(load_source(self.input).groups, [("수성구", "파동", "음식점")])
        self.assertEqual(len(load_businesses(self.input).businesses), 21)

    def test_end_to_end_outputs_single_target_files_and_historical_features(self):
        self.rows = self.rows[:2]
        self.rows[0]["폐업일자"] = "2024-06-01"
        self.rows[1].update({"구": "중구", "행정동": "성내1동", "행정동코드": "2711051700.0"})
        self.write()
        sales = self.root / "sales.csv"
        with sales.open("w", encoding="utf-8-sig", newline="") as stream:
            columns = ["기준분기", "구", "읍면동", "업종", "매출액", "이용건수", "지역코드"]
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            for quarter in ("2022Q1", "2022Q2", "2022Q3", "2022Q4", "2023Q1"):
                for i, raw in enumerate(self.rows):
                    amount = (70 if i == 0 else 110) if quarter == "2023Q1" else 100
                    writer.writerow(dict(zip(columns, (quarter, raw["구"], raw["행정동"], raw["업종"], amount, 10,
                                                       raw["행정동코드"].removesuffix(".0")))))
        out = self.root / "out"
        meta = run_pipeline(self.input, sales, out, CrisisConfig(), LabelConfig(date(2025, 12, 31)))
        self.assertEqual(meta["cohort_audit"]["cohort_rows"], 1)
        self.assertEqual([s["closed_0"] for s in meta["label_distributions"]], [0, 1, 1])
        self.assertTrue((out / "label_distribution_common_cohort.csv").exists())
        self.assertFalse((out / "labeled_6m.csv").exists())
        for months in (12, 18, 24):
            with (out / f"labeled_{months}m.csv").open(encoding="utf-8-sig", newline="") as stream:
                reader = csv.DictReader(stream)
                self.assertEqual([c for c in reader.fieldnames if c.startswith("Y_")], [f"Y_{months}M"])
                self.assertTrue(set(meta["feature_allowlist"]).issubset(reader.fieldnames))
                row = next(reader)
                self.assertEqual(row["동_동일업종수_lag1q"], "1")
                self.assertNotIn("폐업일자", row)


if __name__ == "__main__":
    unittest.main()
