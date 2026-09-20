"""Features recomputed strictly from information at/before each cohort baseline."""

from __future__ import annotations

import math
import statistics
from bisect import bisect_right
from collections import defaultdict
from dataclasses import dataclass
from datetime import date
from itertools import accumulate

from src.data.businesses import Business
from src.data.labeling import quarter_end
from .crisis import MarketQuarter


@dataclass(frozen=True)
class FeatureConfig:
    business_lag_quarters: int = 1
    business_window_quarters: int = 4
    sales_window_quarters: int = 4
    history_start: date = date(2015, 1, 1)

    def __post_init__(self):
        for name in ("business_lag_quarters", "business_window_quarters", "sales_window_quarters"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.sales_window_quarters < 2:
            raise ValueError("sales_window_quarters must be at least 2")
        if not isinstance(self.history_start, date):
            raise ValueError("history_start must be a date")


class _Events:
    """Sorted event indexes make snapshots independent of future outcomes."""

    def __init__(self, businesses):
        self.opens = sorted(b.opened.toordinal() for b in businesses)
        closures = sorted((b.closed.toordinal(), b.opened.toordinal()) for b in businesses if b.closed)
        self.closes = [value[0] for value in closures]
        self.open_sum = [0, *accumulate(self.opens)]
        self.closed_open_sum = [0, *accumulate(value[1] for value in closures)]

    def count(self, when: date) -> int:
        day = when.toordinal()
        return bisect_right(self.opens, day) - bisect_right(self.closes, day)

    def events(self, start: date, end: date) -> tuple[int, int]:
        left, right = start.toordinal(), end.toordinal()
        return (bisect_right(self.opens, right) - bisect_right(self.opens, left),
                bisect_right(self.closes, right) - bisect_right(self.closes, left))

    def mean_age_years(self, when: date) -> float | None:
        day = when.toordinal()
        opened, closed = bisect_right(self.opens, day), bisect_right(self.closes, day)
        active = opened - closed
        if not active:
            return None
        active_open_days = self.open_sum[opened] - self.closed_open_sum[closed]
        return (active * day - active_open_days) / active / 365.25


def _ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def feature_definitions(config: FeatureConfig) -> dict[str, str]:
    lag, window, sales_window = config.business_lag_quarters, config.business_window_quarters, config.sales_window_quarters
    return {
        "업력_연수": "t0와 인허가일의 차이 / 365.25",
        "업력_개월수": "t0와 인허가일의 차이 / 365.25 * 12",
        "최근1년_신규여부": "t0의 4분기 전 분기말보다 늦게 개업했으면 1",
        "최근3년_신규여부": "t0의 12분기 전 분기말보다 늦게 개업했으면 1",
        "quarter_of_year": "t0의 분기 번호 1~4",
        f"구_전체사업장수_lag{lag}q": "과거 스냅샷에 영업 중인 같은 구의 전체 사업자 수",
        f"구_동일업종수_lag{lag}q": "과거 스냅샷에 영업 중인 같은 구·업종 사업자 수",
        f"동_전체사업장수_lag{lag}q": "과거 스냅샷에 영업 중인 같은 구·행정동 전체 사업자 수",
        f"동_동일업종수_lag{lag}q": "과거 스냅샷에 영업 중인 같은 구·행정동·업종 사업자 수",
        f"구_동일업종비율_lag{lag}q": "구 동일업종수 / 구 전체사업장수",
        f"동_동일업종비율_lag{lag}q": "동 동일업종수 / 동 전체사업장수",
        f"동_동일업종_신규수_{window}q_lag{lag}q": "과거 스냅샷까지의 직전 window 분기 중 신규 사업자 수",
        f"동_동일업종_폐업수_{window}q_lag{lag}q": "과거 스냅샷까지의 직전 window 분기 중 폐업 사업자 수",
        f"동_동일업종_신규진입률_{window}q_lag{lag}q": "위 신규수 / 창 시작 시점 영업 사업자 수 (1 초과 가능)",
        f"동_동일업종_폐업률_{window}q_lag{lag}q": "위 폐업수 / (창 시작 영업수 + 창 내 신규수)",
        f"동_동일업종_순증감률_{window}q_lag{lag}q": "(창 종료 영업수 - 창 시작 영업수) / 창 시작 영업수",
        f"동_동일업종_평균업력_lag{lag}q": "과거 스냅샷에 영업 중인 동·업종 사업자 평균 업력(연)",
        f"지역대비_업력비율_lag{lag}q": "사업자의 과거 스냅샷 업력 / 해당 동·업종 평균 업력; 당시 미개업이면 NA",
        "sales_qoq": "현재 매출 / 직전 분기 매출 - 1; 직전 매출 0이면 NA",
        f"sales_trend_{sales_window}q": "최근 window개 분기 log(매출)의 OLS 기울기를 exp(slope)-1로 변환한 분기 추세",
        f"sales_vol_{sales_window}q": "최근 window개 분기 QoQ의 모집단 표준편차 (window+1개 매출 필요)",
        f"sales_drawdown_{sales_window}q": "현재 매출 / 최근 window개 분기 최대 매출 - 1",
        "sales_decline_duration": "현재까지 YoY<0이 연속된 분기 수; 관측 누락 시 재시작",
        "sales_rank_pct": "현재 동일 업종의 유효 YoY 중 평균 순위 / 시장 수",
        "average_ticket": "현재 매출액 / 이용건수; 이용건수 0이면 NA",
    }


def _sales_features(markets: list[MarketQuarter], config: FeatureConfig) -> dict:
    lookup = {(row.group, row.quarter): row for row in markets}
    ranks = {}
    peers = defaultdict(list)
    for row in markets:
        if row.sales_yoy is not None:
            peers[(row.group[2], row.quarter)].append((row.sales_yoy, row.group))
    for (_, quarter), values in peers.items():
        values.sort()
        left = 0
        while left < len(values):
            right = left + 1
            while right < len(values) and values[right][0] == values[left][0]:
                right += 1
            percentile = ((left + 1 + right) / 2) / len(values)
            for _, group in values[left:right]:
                ranks[(group, quarter)] = percentile
            left = right
    result = {}
    durations = {}
    w = config.sales_window_quarters
    for row in sorted(markets, key=lambda item: (item.group, item.quarter)):
        key = (row.group, row.quarter)
        previous = lookup.get((row.group, row.quarter - 1))
        qoq = row.sales / previous.sales - 1 if row.sales is not None and previous and previous.sales else None
        series = [lookup.get((row.group, quarter)) for quarter in range(row.quarter - w, row.quarter + 1)]
        values = [item.sales if item else None for item in series]
        trend = volatility = drawdown = None
        recent = values[1:]
        if all(value is not None and value > 0 for value in recent):
            logs = [math.log(value) for value in recent]
            center = (w - 1) / 2
            slope = sum((i - center) * value for i, value in enumerate(logs)) / sum((i - center) ** 2 for i in range(w))
            trend = math.expm1(slope)
        if all(value is not None and value >= 0 for value in recent) and max(recent) > 0:
            drawdown = row.sales / max(recent) - 1
        if all(value is not None for value in values) and all(value > 0 for value in values[:-1]):
            volatility = statistics.pstdev([values[i] / values[i - 1] - 1 for i in range(1, len(values))])
        duration = durations.get((row.group, row.quarter - 1), 0) + 1 if row.sales_yoy is not None and row.sales_yoy < 0 else 0
        durations[key] = duration
        result[key] = {"sales_qoq": qoq, f"sales_trend_{w}q": trend, f"sales_vol_{w}q": volatility,
                       f"sales_drawdown_{w}q": drawdown, "sales_decline_duration": duration,
                       "sales_rank_pct": ranks.get(key), "average_ticket": _ratio(row.sales, row.transactions) if row.sales is not None else None}
    return result


def add_independent_features(cohort: list[dict], businesses: list[Business], markets: list[MarketQuarter],
                             config: FeatureConfig) -> list[dict]:
    """Return new rows. Raw annual aggregates and future-label columns are not read."""
    by_id = {business.business_id: business for business in businesses}
    pools = defaultdict(list)
    for business in businesses:
        district, area, industry = business.group
        for key in (("district", district), ("district_industry", district, industry),
                    ("area", district, area), ("market", *business.group)):
            pools[key].append(business)
    indexes = {key: _Events(values) for key, values in pools.items()}
    sales = _sales_features(markets, config)
    lag, window = config.business_lag_quarters, config.business_window_quarters
    snapshots = {}
    output = []
    for row in cohort:
        business = by_id[row["관리번호"]]
        quarter = int(row["기준분기"][:4]) * 4 + int(row["기준분기"][-1]) - 1
        baseline = date.fromisoformat(row["t0"])
        asof, start = quarter_end(quarter - lag), quarter_end(quarter - lag - window)
        cache_key = (business.group, quarter)
        if cache_key not in snapshots:
            district, area, industry = business.group
            d = indexes[("district", district)].count(asof)
            di = indexes[("district_industry", district, industry)].count(asof)
            a = indexes[("area", district, area)].count(asof)
            market = indexes[("market", *business.group)]
            n, initial = market.count(asof), market.count(start)
            entries, closures = market.events(start, asof)
            complete = start.toordinal() + 1 >= config.history_start.toordinal()
            snapshots[cache_key] = {
                f"구_전체사업장수_lag{lag}q": d, f"구_동일업종수_lag{lag}q": di,
                f"동_전체사업장수_lag{lag}q": a, f"동_동일업종수_lag{lag}q": n,
                f"구_동일업종비율_lag{lag}q": _ratio(di, d), f"동_동일업종비율_lag{lag}q": _ratio(n, a),
                f"동_동일업종_신규수_{window}q_lag{lag}q": entries if complete else None,
                f"동_동일업종_폐업수_{window}q_lag{lag}q": closures if complete else None,
                f"동_동일업종_신규진입률_{window}q_lag{lag}q": _ratio(entries, initial) if complete else None,
                f"동_동일업종_폐업률_{window}q_lag{lag}q": _ratio(closures, initial + entries) if complete else None,
                f"동_동일업종_순증감률_{window}q_lag{lag}q": _ratio(n - initial, initial) if complete else None,
                f"동_동일업종_평균업력_lag{lag}q": market.mean_age_years(asof),
            }
        aggregate = snapshots[cache_key]
        age = (baseline - business.opened).days
        own_lag_age = (asof - business.opened).days / 365.25 if business.opened <= asof else None
        mean_age = aggregate[f"동_동일업종_평균업력_lag{lag}q"]
        output.append({**row, **aggregate, **sales[cache_key],
                       "행정동코드": business.area_code,
                       "업력_연수": age / 365.25, "업력_개월수": age / 365.25 * 12,
                       "최근1년_신규여부": int(business.opened > quarter_end(quarter - 4)),
                       "최근3년_신규여부": int(business.opened > quarter_end(quarter - 12)),
                       "quarter_of_year": quarter % 4 + 1,
                       f"지역대비_업력비율_lag{lag}q": _ratio(own_lag_age, mean_age) if own_lag_age is not None else None})
    return output
