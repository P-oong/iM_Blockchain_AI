"""Quarterly market crisis detection, independent of business outcomes.

The benchmark is growth of matched-market industry sales totals, including the
subject market. It is never an unweighted mean of market growth rates. Invalid
numeric sales (blank, NaN or infinity) remain unknown; negative sales are errors.

An episode starts when its required consecutive-quarter run is confirmed. The
first evaluable run after initial YoY warmup is eligible (a left-boundary
assumption). After any evaluable history, gaps/unknowns cannot establish a new
entry: a known false raw condition must precede the next eligible run. This
avoids inventing a normal-to-crisis transition across missing observations.
"""

from __future__ import annotations

import csv
import math
import re
from collections import defaultdict
from dataclasses import dataclass, replace
from numbers import Real
from pathlib import Path

from src.data.market_keys import market_key, normalize_text


Group = tuple[str, str, str]
SALES_COLUMNS = ("기준분기", "구", "읍면동", "업종", "매출액", "이용건수")


@dataclass(frozen=True)
class CrisisConfig:
    """Thresholds are decimal changes: -0.20 means -20%, -0.10 means -10 pp."""

    yoy_threshold: float = -0.20
    peer_gap_threshold: float = -0.10
    rule: str = "and"
    min_consecutive_quarters: int = 1

    def __post_init__(self) -> None:
        for name in ("yoy_threshold", "peer_gap_threshold"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
                raise ValueError(f"{name} must be a finite number.")
        if self.rule not in ("and", "or", "yoy-only"):
            raise ValueError("rule must be 'and', 'or', or 'yoy-only'.")
        if type(self.min_consecutive_quarters) is not int or self.min_consecutive_quarters < 1:
            raise ValueError("min_consecutive_quarters must be a positive integer.")


@dataclass(frozen=True)
class MarketQuarter:
    group: Group
    quarter: int
    sales: float | None
    transactions: int | None = 0
    sales_yoy: float | None = None
    peer_yoy: float | None = None
    peer_gap: float | None = None
    crisis: bool | None = None
    episode_start: bool = False


def _quarter_number(value: str) -> int:
    if not re.fullmatch(r"[0-9]{4}Q[1-4]", value):
        raise ValueError(f"Invalid quarter {value!r}; expected YYYYQ1 through YYYYQ4.")
    return int(value[:4]) * 4 + int(value[-1]) - 1


def _number(value: str, column: str, line: int) -> float | None:
    value = normalize_text(value)
    if value.casefold() in ("", "na", "n/a", "nan", "none", "null"):
        return None
    try:
        number = float(value)
    except ValueError as error:
        raise ValueError(f"Sales CSV row {line}: invalid {column}: {value!r}.") from error
    if not math.isfinite(number):
        return None
    if number < 0:
        raise ValueError(f"Sales CSV row {line}: {column} must not be negative.")
    return number


def _read_sales(path: Path, encoding: str) -> dict[tuple[Group, int], MarketQuarter]:
    observations: dict[tuple[Group, int], MarketQuarter] = {}
    with Path(path).open(encoding=encoding, newline="") as handle:
        reader = csv.reader(handle)
        header = [normalize_text(name) for name in next(reader, [])]
        if len(header) != len(set(header)):
            raise ValueError("Sales CSV contains duplicate column names.")
        missing = set(SALES_COLUMNS) - set(header)
        if missing:
            raise ValueError(f"Sales CSV is missing columns: {', '.join(sorted(missing))}.")
        positions = [header.index(name) for name in SALES_COLUMNS]
        for line, row in enumerate(reader, start=2):
            if len(row) != len(header):
                raise ValueError(f"Sales CSV row {line}: column count does not match header.")
            period, district, area, industry, sales, transactions = [
                normalize_text(row[index]) for index in positions
            ]
            group = market_key(district, area, industry)
            if group is None:
                raise ValueError(f"Sales CSV row {line}: missing or invalid market key.")
            quarter = _quarter_number(period)
            key = (group, quarter)
            if key in observations:
                raise ValueError(f"Sales CSV row {line}: duplicate market-quarter.")
            amount = _number(sales, "매출액", line)
            count = _number(transactions, "이용건수", line)
            if count is not None and not count.is_integer():
                raise ValueError(f"Sales CSV row {line}: 이용건수 must be an integer.")
            observations[key] = MarketQuarter(
                group, quarter, amount, None if count is None else int(count)
            )
    if not observations:
        raise ValueError("Sales CSV contains no observations.")
    return observations


def _at_most(value: float, threshold: float) -> bool:
    # A mathematically exact -20% may be represented as -0.19999999999999996.
    return value <= threshold or math.isclose(value, threshold, rel_tol=1e-12, abs_tol=1e-12)


def _raw_crisis(row: MarketQuarter, config: CrisisConfig) -> bool | None:
    if row.sales_yoy is None:
        return None
    absolute = _at_most(row.sales_yoy, config.yoy_threshold)
    if config.rule == "yoy-only":
        return absolute
    if row.peer_gap is None:
        return None
    relative = _at_most(row.peer_gap, config.peer_gap_threshold)
    return absolute and relative if config.rule == "and" else absolute or relative


def load_sales(
    path: Path, config: CrisisConfig, encoding: str = "utf-8-sig"
) -> list[MarketQuarter]:
    """Load and detect crisis rows, sorted by (district, area, industry, quarter).

    YoY requires the exact t-4 observation, rather than the fourth previous row.
    Peer totals use only markets with finite sales at both t and t-4, preventing
    entry/exit of CSV coverage from creating artificial industry growth. Zero
    total lag sales leave the benchmark unknown; zero individual lag sales leave
    that market's YoY unknown. Missing values are retained, never imputed.
    """
    observations = _read_sales(path, encoding)
    paired_totals: dict[tuple[str, int], tuple[list[float], list[float]]] = defaultdict(
        lambda: ([], [])
    )
    for (group, quarter), row in observations.items():
        lag = observations.get((group, quarter - 4))
        if row.sales is None or lag is None or lag.sales is None:
            continue
        current_values, lag_values = paired_totals[(group[2], quarter)]
        current_values.append(row.sales)
        lag_values.append(lag.sales)

    benchmarks: dict[tuple[str, int], float | None] = {}
    for key, (current_values, lag_values) in paired_totals.items():
        current_total = math.fsum(current_values)
        lag_total = math.fsum(lag_values)
        benchmark = current_total / lag_total - 1 if lag_total > 0 else None
        benchmarks[key] = benchmark if benchmark is None or math.isfinite(benchmark) else None

    output: list[MarketQuarter] = []
    previous_group: Group | None = None
    previous_quarter: int | None = None
    consecutive = 0
    seen_evaluable = False
    onset_allowed = True
    previous_crisis = False
    for (group, quarter), row in sorted(observations.items()):
        if group != previous_group:
            previous_group = group
            previous_quarter = None
            consecutive = 0
            seen_evaluable = False
            onset_allowed = True
            previous_crisis = False
        elif previous_quarter is not None and quarter != previous_quarter + 1:
            consecutive = 0
            previous_crisis = False
            if seen_evaluable:
                onset_allowed = False

        lag = observations.get((group, quarter - 4))
        yoy = None
        if row.sales is not None and lag is not None and lag.sales is not None and lag.sales > 0:
            value = row.sales / lag.sales - 1
            if math.isfinite(value):
                yoy = value
        peer = benchmarks.get((group[2], quarter))
        gap = yoy - peer if yoy is not None and peer is not None else None
        row = replace(row, sales_yoy=yoy, peer_yoy=peer, peer_gap=gap)
        raw = _raw_crisis(row, config)
        episode_start = False
        if raw is None:
            consecutive = 0
            crisis = None
            if seen_evaluable:
                onset_allowed = False
        elif not raw:
            seen_evaluable = True
            consecutive = 0
            crisis = False
            onset_allowed = True
        else:
            seen_evaluable = True
            consecutive += 1
            crisis = consecutive >= config.min_consecutive_quarters
            episode_start = crisis and not previous_crisis and onset_allowed
        output.append(replace(row, crisis=crisis, episode_start=episode_start))
        previous_crisis = crisis is True
        previous_quarter = quarter
    return output
