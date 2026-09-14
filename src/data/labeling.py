"""Configurable first-crisis cohorts and fixed-horizon business survival labels."""

from __future__ import annotations

import calendar
from collections import defaultdict
from dataclasses import dataclass
from datetime import date

from src.features.crisis import MarketQuarter

from .businesses import Business, exclusion_row


@dataclass(frozen=True)
class LabelConfig:
    observation_end: date
    horizons: tuple[int, ...] = (6, 12)
    entry_mode: str = "episode-start"

    def __post_init__(self) -> None:
        if not isinstance(self.observation_end, date):
            raise ValueError("observation_end는 날짜여야 합니다.")
        if (not self.horizons or any(type(months) is not int or months <= 0 for months in self.horizons)
                or len(self.horizons) != len(set(self.horizons))):
            raise ValueError("관측기간은 중복 없는 양의 정수 개월이어야 합니다.")
        if self.entry_mode not in ("episode-start", "first-exposure"):
            raise ValueError("entry_mode는 episode-start 또는 first-exposure여야 합니다.")


def quarter_end(quarter: int) -> date:
    year, quarter_index = divmod(quarter, 4)
    month = (quarter_index + 1) * 3
    return date(year, month, calendar.monthrange(year, month)[1])


def add_months(value: date, months: int) -> date:
    """Preserve month-end: 2024-06-30 + 6 months = 2024-12-31."""
    if type(months) is not int or months < 0:
        raise ValueError("months는 0 이상의 정수여야 합니다.")
    year, month_index = divmod(value.year * 12 + value.month - 1 + months, 12)
    month = month_index + 1
    last_day = calendar.monthrange(year, month)[1]
    is_month_end = value.day == calendar.monthrange(value.year, value.month)[1]
    return date(year, month, last_day if is_month_end else min(value.day, last_day))


def label_outcome(
    opened: date, closed: date | None, t0: date, horizon_months: int, observation_end: date
) -> int | None:
    """Label only mature windows; closure on t0 is outside the active risk set."""
    if opened > t0 or (closed is not None and closed <= t0):
        raise ValueError("기준일에 영업 중인 사업자만 라벨링할 수 있습니다.")
    if type(horizon_months) is not int or horizon_months <= 0:
        raise ValueError("관측기간은 양의 정수 개월이어야 합니다.")
    endpoint = add_months(t0, horizon_months)
    if endpoint > observation_end:
        return None
    return 0 if closed is not None and t0 < closed <= endpoint else 1


def cohort_columns(config: LabelConfig) -> list[str]:
    columns = ["관리번호", "구", "읍면동", "업종", "기준분기", "t0", "업력_일수",
               "매출액", "이용건수", "sales_yoy", "peer_yoy", "peer_gap", "crisis_flag", "episode_start"]
    for months in config.horizons:
        columns.extend([f"horizon_end_{months}M", f"Y_{months}M", f"label_status_{months}M"])
    return columns


def build_cohort(
    businesses: list[Business], markets: list[MarketQuarter], config: LabelConfig
) -> tuple[list[dict], list[dict]]:
    """Select at most one eligible t0 per ID before checking label maturity.

    Thus censoring never causes a business to be reassigned to another crisis.
    episode-start excludes new openings during an already ongoing market crisis;
    first-exposure allows their first eligible quarter in that ongoing crisis.
    """
    groups = {market.group for market in markets}
    candidates: dict[tuple, list[MarketQuarter]] = defaultdict(list)
    for market in sorted(markets, key=lambda item: (item.group, item.quarter)):
        if (market.crisis is True and quarter_end(market.quarter) <= config.observation_end
                and (config.entry_mode == "first-exposure" or market.episode_start)):
            candidates[market.group].append(market)
    rows, exclusions = [], []
    seen_ids = set()
    for business in sorted(businesses, key=lambda item: item.business_id):
        if business.business_id in seen_ids:
            raise ValueError("build_cohort에는 사업자별로 중복 제거된 자료가 필요합니다.")
        seen_ids.add(business.business_id)
        selected = None
        reason = "no_eligible_crisis_while_open"
        if business.group not in groups:
            reason = "market_not_in_sales"
        elif business.opened > config.observation_end:
            reason = "opened_after_observation_end"
        else:
            for candidate in candidates[business.group]:
                t0 = quarter_end(candidate.quarter)
                if business.opened <= t0 and (business.closed is None or business.closed > t0):
                    selected = candidate
                    break
        if selected is None:
            exclusions.append(exclusion_row(business.business_id, business.group, reason, "cohort"))
            continue
        t0 = quarter_end(selected.quarter)
        row = dict(zip(cohort_columns(config)[:14], (
            business.business_id, *business.group,
            f"{selected.quarter // 4:04d}Q{selected.quarter % 4 + 1}", t0.isoformat(),
            (t0 - business.opened).days, selected.sales, selected.transactions,
            selected.sales_yoy, selected.peer_yoy, selected.peer_gap, 1, int(selected.episode_start),
        )))
        for months in config.horizons:
            label = label_outcome(business.opened, business.closed, t0, months, config.observation_end)
            row[f"horizon_end_{months}M"] = add_months(t0, months).isoformat()
            row[f"Y_{months}M"] = label
            row[f"label_status_{months}M"] = "insufficient_followup" if label is None else "observed"
        rows.append(row)
    return rows, exclusions


def label_distribution(rows: list[dict], months: int) -> dict:
    values = [row[f"Y_{months}M"] for row in rows]
    survival, closure, unknown = values.count(1), values.count(0), values.count(None)
    labeled = survival + closure
    return {
        "horizon_months": months, "total_cohort": len(values), "labeled": labeled,
        "survived_1": survival, "closed_0": closure, "unobserved_na": unknown,
        "survival_pct_among_labeled": 100 * survival / labeled if labeled else None,
        "closure_pct_among_labeled": 100 * closure / labeled if labeled else None,
        "na_pct_among_cohort": 100 * unknown / len(values) if values else None,
    }
