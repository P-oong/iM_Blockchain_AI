import unittest
from dataclasses import replace
from datetime import date

from src.data.businesses import Business
from src.features.crisis import MarketQuarter
from src.features.independent import FeatureConfig, add_independent_features


GROUP = ("중구", "성내1동", "식품_일반음식점")


class IndependentFeatureTests(unittest.TestCase):
    def setUp(self):
        self.businesses = [
            Business("A", GROUP, date(2020, 1, 1), date(2024, 12, 31)),
            Business("B", GROUP, date(2020, 1, 1), date(2023, 12, 31)),
            Business("C", GROUP, date(2023, 10, 1), None),
            Business("D", GROUP, date(2024, 5, 1), None),
        ]
        self.markets = [MarketQuarter(GROUP, 2023 * 4 + i, 100 * 1.1 ** i, 10,
                                     sales_yoy=1.1 ** 4 - 1 if i >= 4 else None) for i in range(6)]
        self.rows = [{"관리번호": "A", "기준분기": "2024Q2", "t0": "2024-06-30"}]

    def output(self, businesses=None, markets=None, config=None):
        return add_independent_features(self.rows, businesses or self.businesses,
                                        markets or self.markets, config or FeatureConfig())[0]

    def test_counts_and_rates_use_previous_quarter_only(self):
        row = self.output()
        self.assertEqual(row["동_동일업종수_lag1q"], 2)
        self.assertEqual(row["동_동일업종_신규수_4q_lag1q"], 1)
        self.assertEqual(row["동_동일업종_폐업수_4q_lag1q"], 1)
        self.assertAlmostEqual(row["동_동일업종_신규진입률_4q_lag1q"], 0.5)
        self.assertAlmostEqual(row["동_동일업종_폐업률_4q_lag1q"], 1 / 3)
        self.assertAlmostEqual(row["동_동일업종_순증감률_4q_lag1q"], 0)

    def test_future_closure_opening_and_sales_do_not_change_past_features(self):
        businesses = [replace(b, closed=date(2026, 1, 1)) if b.business_id == "A" else b for b in self.businesses]
        businesses[-1] = replace(businesses[-1], opened=date(2026, 2, 1))
        markets = self.markets + [MarketQuarter(GROUP, 2024 * 4 + 2, 1, 1, sales_yoy=-0.99)]
        self.assertEqual(self.output(), self.output(businesses, markets))

    def test_sales_features_match_constant_geometric_growth(self):
        row = self.output()
        self.assertAlmostEqual(row["sales_qoq"], 0.1)
        self.assertAlmostEqual(row["sales_trend_4q"], 0.1)
        self.assertAlmostEqual(row["sales_vol_4q"], 0.0)
        self.assertEqual(row["sales_drawdown_4q"], 0)

    def test_history_shortfall_is_not_imputed_as_zero(self):
        row = self.output(config=FeatureConfig(history_start=date(2024, 1, 1)))
        self.assertIsNone(row["동_동일업종_폐업률_4q_lag1q"])
        self.assertIsNone(row["동_동일업종_신규수_4q_lag1q"])
        self.assertEqual(row["동_동일업종수_lag1q"], 2)

    def test_new_business_has_no_past_age_ratio(self):
        self.rows[0]["관리번호"] = "D"
        row = self.output()
        self.assertIsNone(row["지역대비_업력비율_lag1q"])
        self.assertEqual(row["최근1년_신규여부"], 1)

    def test_same_dong_name_in_another_district_is_separate(self):
        other = Business("E", ("서구", GROUP[1], GROUP[2]), date(2020, 1, 1), None)
        self.assertEqual(self.output(), self.output(self.businesses + [other]))

    def test_window_option_changes_names_and_missing_sales_remain_unknown(self):
        row = self.output(markets=self.markets[2:], config=FeatureConfig(sales_window_quarters=5))
        self.assertIn("sales_trend_5q", row)
        self.assertNotIn("sales_trend_4q", row)
        self.assertIsNone(row["sales_trend_5q"])
        self.assertIsNone(row["sales_vol_5q"])


if __name__ == "__main__":
    unittest.main()
