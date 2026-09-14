"""Reduce annual public rows to one validated business record for labeling."""

from __future__ import annotations

import csv
from collections import Counter
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

from .market_keys import Group, MISSING_VALUES, market_key, normalize_text


REQUIRED_COLUMNS = ("관리번호", "구", "읍면동", "업종", "인허가일자", "폐업일자")
CLOSED_FLAGS = ("기준연도말_폐업상태", "해당연도_폐업여부")


@dataclass(frozen=True)
class Business:
    business_id: str
    group: Group
    opened: date
    closed: date | None


@dataclass
class BusinessSource:
    businesses: list[Business]
    exclusions: list[dict]
    audit: dict


@dataclass
class _Record:
    group: Group
    opened: date | None = None
    closed: date | None = None
    closed_flag: bool = False
    errors: set[str] = field(default_factory=set)


def parse_date(value: str) -> date | None:
    """Accept a date, never reinterpret an invalid nonempty value as survival."""
    value = normalize_text(value)
    if value.lower() in MISSING_VALUES:
        return None
    return date.fromisoformat(value)


def exclusion_row(business_id: str, group: Group, reason: str, stage: str) -> dict:
    return dict(zip(("관리번호", "구", "읍면동", "업종", "stage", "reason"),
                    (business_id, *group, stage, reason)))


def load_businesses(path: Path, encoding: str = "utf-8-sig") -> BusinessSource:
    """Read only identity, market, permit/closure dates and closure consistency flags.

    A historical blank closure may be completed by a later record. Two distinct
    nonempty dates or market keys are ambiguous and exclude the entire business.
    No annual precomputed age, competition or future-derived feature is copied.
    """
    records: dict[str, _Record] = {}
    row_count = missing_id_rows = 0
    with Path(path).open(encoding=encoding, newline="") as stream:
        reader = csv.reader(stream)
        header = [normalize_text(value) for value in next(reader, [])]
        missing = set(REQUIRED_COLUMNS) - set(header)
        if missing or len(header) != len(set(header)):
            raise ValueError(f"사업자 CSV 필수 열 누락 또는 중복: {sorted(missing)}")
        positions = [header.index(name) for name in REQUIRED_COLUMNS]
        flags = [header.index(name) for name in CLOSED_FLAGS if name in header]
        for line_number, row in enumerate(reader, start=2):
            row_count += 1
            if len(row) != len(header):
                raise ValueError(f"사업자 CSV {line_number}행의 열 개수가 맞지 않습니다.")
            business_id, district, area, industry, opened_raw, closed_raw = (
                normalize_text(row[position]) for position in positions
            )
            if business_id.lower() in MISSING_VALUES:
                missing_id_rows += 1
                continue
            group = (district, area, industry)
            record = records.get(business_id)
            if record is None:
                record = records[business_id] = _Record(group)
            if group != record.group:
                record.errors.add("conflicting_market_key")
            if market_key(*group) is None:
                record.errors.add("invalid_or_missing_market_key")
            for attr, raw in (("opened", opened_raw), ("closed", closed_raw)):
                try:
                    parsed = parse_date(raw)
                except ValueError:
                    record.errors.add(f"invalid_{attr}_date")
                    continue
                previous = getattr(record, attr)
                if parsed is not None:
                    if previous is not None and previous != parsed:
                        record.errors.add(f"conflicting_{attr}_date")
                    else:
                        setattr(record, attr, parsed)
            record.closed_flag |= any(normalize_text(row[index]) in ("1", "1.0", "True", "true", "폐업") for index in flags)
    businesses, exclusions = [], []
    reasons: Counter = Counter()
    for business_id, record in sorted(records.items()):
        if record.opened is None:
            record.errors.add("missing_opened_date")
        if record.closed is not None and record.opened is not None and record.closed < record.opened:
            record.errors.add("closure_before_opening")
        if record.closed is None and record.closed_flag:
            record.errors.add("closed_status_without_closure_date")
        if record.errors:
            reasons.update(record.errors)
            exclusions.append(exclusion_row(business_id, record.group, "|".join(sorted(record.errors)), "source"))
        else:
            businesses.append(Business(business_id, record.group, record.opened, record.closed))
    if not records:
        raise ValueError("관리번호가 있는 사업자 행이 없습니다.")
    return BusinessSource(businesses, exclusions, {
        "source_file": Path(path).name,
        "source_rows": row_count,
        "rows_without_business_id": missing_id_rows,
        "unique_business_ids": len(records),
        "valid_businesses": len(businesses),
        "excluded_businesses": len(exclusions),
        "exclusion_reasons_nonexclusive": dict(sorted(reasons.items())),
        "businesses_with_closure_date": sum(b.closed is not None for b in businesses),
        "businesses_without_closure_date": sum(b.closed is None for b in businesses),
        "used_columns": list(REQUIRED_COLUMNS) + [header[index] for index in flags],
    })
