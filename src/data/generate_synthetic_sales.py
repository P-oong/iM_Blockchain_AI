"""Generate quarterly synthetic market sales without consulting business outcomes.

Python standard library only. Run from the repository root with --help.
The six-column CSV is a provisional DIP-like contract, not a verified DIP schema.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

if __package__:
    from .market_keys import (AREA_PATTERN, BUILDING_PATTERN, normalize_text, geography_columns,
                             normalize_area_code, canonical_geography, market_key)
    from .provenance import fingerprint, find_input_csv
else:
    from market_keys import (AREA_PATTERN, BUILDING_PATTERN, normalize_text, geography_columns,
                            normalize_area_code, canonical_geography, market_key)
    from provenance import fingerprint, find_input_csv


KEY_COLUMNS = ("구", "읍면동", "업종")
COMPETITION_COLUMNS = ("기준연도", "동_동일업종수", "동_동일업종_신규진입률")
SALES_COLUMNS = ("기준분기", *KEY_COLUMNS, "매출액", "이용건수")
Group = tuple[str, str, str]
REPO_ROOT = Path(__file__).resolve().parents[2]


def quarter_number(value: str) -> int:
    if not re.fullmatch(r"[0-9]{4}Q[1-4]", value):
        raise ValueError(f"분기는 YYYYQ1~YYYYQ4 형식이어야 합니다: {value}")
    return int(value[:4]) * 4 + int(value[-1]) - 1


def quarter_label(number: int) -> str:
    return f"{number // 4:04d}Q{number % 4 + 1}"


@dataclass(frozen=True)
class Config:
    start_quarter: str = "2015Q1"
    end_quarter: str = "2025Q4"
    seed: int = 42
    shock_probability: float = 0.08
    use_competition: bool = False

    def __post_init__(self) -> None:
        if quarter_number(self.start_quarter) > quarter_number(self.end_quarter):
            raise ValueError("시작분기는 종료분기보다 늦을 수 없습니다.")
        if not math.isfinite(self.shock_probability) or not 0 <= self.shock_probability <= 1:
            raise ValueError("shock_probability는 0~1이어야 합니다.")


@dataclass
class SourceData:
    groups: list[Group]
    competition: dict[tuple[Group, int], tuple[float, float]]
    audit: dict
    area_codes: dict = field(default_factory=dict)


def _clean(value: str | None) -> str:
    return normalize_text(value)


def load_source(
    path: Path, encoding: str = "utf-8-sig", use_competition: bool = False,
    through_year: int = 2025, area_column: str = "auto",
) -> SourceData:
    """Stream the annual file, retaining only an explicit input allowlist.

    No business IDs, opening/closure dates, survival flags or labels are used.
    Repeated business-year rows contribute only one market key / annual metric.
    """
    with Path(path).open(encoding=encoding, newline="") as stream:
        header = [normalize_text(value) for value in next(csv.reader(stream), [])]
    chosen_area, code_column = geography_columns(header, area_column)
    if chosen_area != "읍면동" or code_column:
        if use_competition:
            raise ValueError("수정 원본의 연간 경쟁지표는 재사용하지 않습니다. 라벨링 단계에서 과거 날짜로 재산출하세요.")
        return _load_coded_catalog(path, encoding, through_year, chosen_area, code_column)
    groups: set[Group] = set()
    competition: dict[tuple[Group, int], tuple[float, float]] = {}
    reasons: Counter = Counter()
    invalid_areas: Counter = Counter()
    total = 0
    used_columns = KEY_COLUMNS + (COMPETITION_COLUMNS if use_competition else ())
    with Path(path).open(encoding=encoding, newline="") as handle:
        reader = csv.reader(handle)
        header = [_clean(name) for name in next(reader, [])]
        missing = set(used_columns) - set(header)
        if missing:
            raise ValueError(f"필수 열이 없습니다: {', '.join(sorted(missing))}")
        if len(header) != len(set(header)):
            raise ValueError("중복된 CSV 열 이름이 있습니다.")
        year_index = header.index("기준연도") if "기준연도" in header else None
        positions = [header.index(name) for name in used_columns]
        for line, row in enumerate(reader, start=2):
            total += 1
            if len(row) != len(header):
                raise ValueError(f"CSV {line}행의 열 개수가 헤더와 다릅니다.")
            if year_index is not None:
                try:
                    year = int(_clean(row[year_index]))
                except ValueError as error:
                    raise ValueError(f"CSV {line}행의 기준연도가 정수가 아닙니다.") from error
                if year > through_year:
                    reasons["after_source_cutoff_year"] += 1
                    continue
            values = [_clean(row[index]) for index in positions]
            group = tuple(values[:3])
            if any(value.lower() in ("", "nan", "none", "null", "미상", "없음") for value in group):
                reasons["missing_market_key"] += 1
                continue
            if not AREA_PATTERN.fullmatch(group[1]) or BUILDING_PATTERN.search(group[1]):
                reasons["invalid_area_syntax"] += 1
                invalid_areas[group[1]] += 1
                continue
            groups.add(group)
            if use_competition:
                try:
                    year_value, count, entry_rate = map(float, values[3:])
                    if (not all(math.isfinite(value) for value in (year_value, count, entry_rate))
                            or not year_value.is_integer() or not 1900 <= year_value <= 2200
                            or not count.is_integer() or count < 0 or not 0 <= entry_rate <= 1):
                        raise ValueError
                    key = (group, int(year_value))
                    value = (count, entry_rate)
                    if key in competition and competition[key] != value:
                        raise ValueError(f"동일 지역·업종·연도의 경쟁지표가 다릅니다: {key}")
                    competition[key] = value
                except ValueError as error:
                    raise ValueError(f"CSV {line}행 경쟁지표 오류: {error}") from error
    if not groups:
        raise ValueError("사용 가능한 구·읍면동·업종 조합이 없습니다.")
    if use_competition:
        districts: dict[tuple[str, str], set[str]] = defaultdict(set)
        for district, area, industry in groups:
            districts[(area, industry)].add(district)
        if any(len(values) > 1 for values in districts.values()):
            raise ValueError("구가 다른 동명 읍면동·업종이 있습니다. 원본 경쟁지표의 구별 집계를 검증·수정한 뒤 사용하세요.")
    return SourceData(sorted(groups), competition, {
        "source_file": Path(path).name,
        "source_rows": total,
        "used_columns": sorted(set(used_columns) | ({"기준연도"} if year_index is not None else set())),
        "source_cutoff_year": through_year if year_index is not None else None,
        "excluded_rows": sum(reasons.values()),
        "excluded_rows_by_reason": dict(reasons),
        "invalid_area_values": dict(sorted(invalid_areas.items())),
        "market_groups": len(groups),
        "annual_competition_snapshots": len(competition),
    })


def _load_coded_catalog(path, encoding, through_year, area_column, code_column):
    groups = set()
    counts = defaultdict(Counter)
    excluded = Counter()
    total = 0
    used = ["구", area_column, "업종"] + ([code_column] if code_column else [])
    with Path(path).open(encoding=encoding, newline="") as stream:
        reader = csv.reader(stream)
        header = [normalize_text(value) for value in next(reader, [])]
        if len(header) != len(set(header)) or set(used) - set(header):
            raise ValueError("지역·업종 입력 열이 누락되거나 중복되었습니다.")
        indices = [header.index(name) for name in used]
        year_index = header.index("기준연도") if "기준연도" in header else None
        for row in reader:
            total += 1
            if len(row) != len(header):
                raise ValueError("원본 CSV 열 개수가 일치하지 않습니다.")
            if year_index is not None and int(row[year_index]) > through_year:
                excluded["after_source_cutoff_year"] += 1
                continue
            values = [normalize_text(row[index]) for index in indices]
            group = market_key(*values[:3])
            if group is None:
                excluded["invalid_market_key"] += 1
                continue
            try:
                code = normalize_area_code(values[3]) if code_column else None
            except ValueError:
                excluded["invalid_area_code"] += 1
                continue
            groups.add((group, code))
            if code:
                counts[code][group[:2]] += 1
    mapping, corrections = canonical_geography(counts)
    normalized, codes = set(), {}
    for group, code in groups:
        if code and code not in mapping:
            continue
        canonical = (*mapping[code], group[2]) if code else group
        normalized.add(canonical)
        if code:
            codes[canonical] = code
    if not normalized:
        raise ValueError("사용할 수 있는 지역·업종 조합이 없습니다.")
    return SourceData(sorted(normalized), {}, {
        "source_file": Path(path).name, "source_rows": total,
        "used_columns": used + (["기준연도"] if year_index is not None else []),
        "source_cutoff_year": through_year, "excluded_rows": sum(excluded.values()),
        "excluded_rows_by_reason": dict(excluded), "market_groups": len(normalized),
        "geography": {"area_column": area_column, "code_column": code_column,
                      "codes": len(mapping), "corrections": corrections},
    }, codes)


def _rng(seed: int, *parts: object) -> random.Random:
    payload = json.dumps([seed, *parts], ensure_ascii=False, separators=(",", ":"))
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return random.Random(int.from_bytes(digest[:16], "big"))


def _industry_profile(seed: int, industry: str) -> tuple:
    """Illustrative PoC assumptions in KRW, not estimated market parameters."""
    rng = _rng(seed, "industry", industry)
    if "휴게음식" in industry:
        base, ticket, season = 350_000_000, 8_000, (0.88, 1.08, 1.12, 0.92)
    elif "음식" in industry:
        base, ticket, season = 700_000_000, 25_000, (0.91, 0.99, 1.00, 1.10)
    elif "미용" in industry or "이용업" in industry:
        base, ticket, season = 220_000_000, 30_000, (0.97, 1.02, 0.98, 1.03)
    elif "축산" in industry:
        base, ticket, season = 500_000_000, 45_000, (1.07, 0.91, 1.08, 0.94)
    else:
        base, ticket, season = rng.uniform(200_000_000, 900_000_000), rng.uniform(12_000, 60_000), (0.96, 1.01, 1.02, 1.01)
    return base, ticket, season, rng.uniform(-0.005, 0.025), rng.uniform(0, 2 * math.pi)


def _local_shocks(group: Group, config: Config, start: int, end: int) -> dict[int, float]:
    """Independent local shocks, a low-sales phase, then gradual recovery.

    Events use calendar-keyed random streams; changing the requested output
    window, group order or future data cannot change an earlier event.
    """
    factors = {quarter: 1.0 for quarter in range(start, end + 1)}
    for onset in range(start - 9, end + 1):
        # Keep the initial reference year free of artificial local shocks.
        if onset < quarter_number("2016Q1"):
            continue
        rng = _rng(config.seed, "shock", *group, onset)
        if rng.random() >= config.shock_probability:
            continue
        depth = rng.uniform(0.28, 0.52)
        hold, recovery = rng.randint(2, 4), rng.randint(2, 5)
        for age in range(hold + recovery):
            quarter = onset + age
            if start <= quarter <= end:
                remaining = 1.0 if age < hold else 1 - (age - hold + 1) / recovery
                factors[quarter] = min(factors[quarter], 1 - depth * remaining)
    return factors


def _competition_factor(source: SourceData, group: Group, year: int) -> float:
    """Only t-1 and t-2 calendar-year aggregates, never t or future values."""
    previous = source.competition.get((group, year - 1))
    older = source.competition.get((group, year - 2))
    if previous is None:
        return 1.0
    count, entry_rate = previous
    growth = 0.0 if older is None or older[0] <= 0 else max(0.0, count / older[0] - 1)
    return math.exp(-0.15 * min(growth, 1.0) - 0.10 * entry_rate)


def generate_sales(source: SourceData, config: Config) -> list[dict]:
    """Return a balanced market-quarter panel, without any business-level Y."""
    start, end = quarter_number(config.start_quarter), quarter_number(config.end_quarter)
    rows = []
    for group in sorted(set(source.groups)):
        district, area, industry = group
        base, ticket, seasons, growth, phase = _industry_profile(config.seed, industry)
        area_scale = _rng(config.seed, "area", district, area).uniform(0.55, 1.80)
        market_scale = _rng(config.seed, "market", *group).uniform(0.75, 1.35)
        shocks = _local_shocks(group, config, start, end)
        for quarter in range(start, end + 1):
            years = (quarter - quarter_number("2015Q1")) / 4
            industry_cycle = math.exp(0.04 * math.sin(2 * math.pi * years / 5 + phase))
            # Common hypothetical downturn for all markets, unrelated to closures.
            city_cycle = math.exp(0.08 * math.sin(2 * math.pi * years / 7))
            current = _rng(config.seed, "noise", *group, quarter).gauss(0, 0.035)
            previous = _rng(config.seed, "noise", *group, quarter - 1).gauss(0, 0.035)
            noise = math.exp(0.8 * current + 0.6 * previous)
            competition = _competition_factor(source, group, quarter // 4) if config.use_competition else 1.0
            sales = max(1, round(base * area_scale * market_scale * seasons[quarter % 4]
                                 * math.exp(growth * years) * industry_cycle * city_cycle
                                 * competition * shocks[quarter] * noise))
            price_noise = _rng(config.seed, "ticket", *group, quarter).gauss(0, 0.025)
            average_ticket = ticket * math.exp(0.015 * years + price_noise)
            rows.append(dict(zip(SALES_COLUMNS, (
                quarter_label(quarter), district, area, industry, sales,
                max(1, round(sales / average_ticket)),
            ))))
            if source.area_codes:
                rows[-1]["지역코드"] = source.area_codes[group]
    return sorted(rows, key=lambda row: tuple(row[column] for column in SALES_COLUMNS[:4]))


def summarize(rows: list[dict]) -> dict:
    """Check the intended Crisis rule; no eligibility cohorts or Y are created.

    The industry benchmark is YoY of summed sales, not mean district YoY.
    'Daegu' here covers only retained input market keys, not official city totals.
    """
    sales_by_key = {}
    totals: dict[tuple[str, int], int] = defaultdict(int)
    for row in rows:
        quarter = quarter_number(row["기준분기"])
        group = tuple(row[column] for column in KEY_COLUMNS)
        key = (group, quarter)
        if key in sales_by_key:
            raise ValueError(f"중복 매출 키: {key}")
        if row["매출액"] <= 0 or row["이용건수"] <= 0:
            raise ValueError("매출액과 이용건수는 양수여야 합니다.")
        sales_by_key[key] = row["매출액"]
        totals[(group[2], quarter)] += row["매출액"]
    eligible = crisis = 0
    crisis_by_quarter: Counter = Counter()
    for (group, quarter), sales in sales_by_key.items():
        prior = sales_by_key.get((group, quarter - 4))
        prior_total = totals.get((group[2], quarter - 4))
        if prior is None or not prior_total:
            continue
        eligible += 1
        yoy = sales / prior - 1
        peer_yoy = totals[(group[2], quarter)] / prior_total - 1
        if yoy <= -0.20 + 1e-12 and yoy - peer_yoy <= -0.10 + 1e-12:
            crisis += 1
            crisis_by_quarter[quarter_label(quarter)] += 1
    return {
        "rows": len(rows),
        "groups": len({group for group, _ in sales_by_key}),
        "yoy_eligible_rows": eligible,
        "crisis_rows": crisis,
        "crisis_share_of_yoy_eligible": crisis / eligible if eligible else None,
        "crisis_rows_by_quarter": dict(sorted(crisis_by_quarter.items())),
    }


def save_generation(source: SourceData, config: Config, output: Path, input_path: Path) -> dict:
    output = Path(output).resolve()
    if output == Path(input_path).resolve() or output.is_relative_to((REPO_ROOT / "data/raw").resolve()):
        raise ValueError("입력 파일 또는 data/raw에 출력할 수 없습니다.")
    source_fingerprint = fingerprint(input_path)
    rows = generate_sales(source, config)
    diagnostics = summarize(rows)
    metadata = {
        "is_synthetic": True, "generator_version": 2, "config": asdict(config),
        "source_fingerprint": source_fingerprint,
        "schema": {"key": list(SALES_COLUMNS[:4]), "sales_unit": "KRW", "transactions_unit": "count"},
        "source": source.audit, "validation": diagnostics,
        "assumptions": [
            "읍면동 출력 열은 입력에서 선택한 지역 열의 호환 별칭입니다. geography.area_column에서 실제 기준을 확인하세요.",
            "새 원본에서는 행정동을 우선하며 지역코드는 문자열로 보존합니다. 구명은 동일 코드의 원본 내 95% 이상 일치 매핑으로 정규화합니다.",
            "전체 기간에 고정된 지역 목록을 쓰며 역사적 행정구역 변경을 복원하지 않습니다.",
            "매출 규모·계절성·추세·충격은 PoC 가정이며 실제 DIP 명세·매출의 검증된 복제가 아닙니다.",
            "폐업일·폐업라벨은 생성에 사용하지 않습니다. 경쟁효과는 기본적으로 중립입니다.",
            "업종 비교치는 입력 시장들의 합계이며 공식 대구 전체 통계가 아닙니다.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_suffix(".csv.tmp")
    with temp.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(SALES_COLUMNS) + (["지역코드"] if source.area_codes else []))
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(output)
    output.with_suffix(".metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="원본 사업체 CSV. 생략 시 data/raw의 유일한 CSV")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "data/synthetic/synthetic_dip_sales.csv")
    parser.add_argument("--start-quarter", default="2015Q1")
    parser.add_argument("--end-quarter", default="2025Q4")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shock-probability", type=float, default=0.08)
    parser.add_argument("--use-competition", action="store_true", help="산출 기준을 확인한 경쟁지표를 전년도부터 사용")
    parser.add_argument("--area-column", default="auto", help="auto는 행정동을 우선 선택")
    parser.add_argument("--encoding", default="utf-8-sig", help="입력 인코딩. 출력은 항상 utf-8-sig")
    args = parser.parse_args()
    try:
        config = Config(args.start_quarter, args.end_quarter, args.seed,
                        args.shock_probability, args.use_competition)
        if args.input is None:
            args.input = find_input_csv(REPO_ROOT / "data/raw")
        output = args.output.resolve()
        if output.suffix.lower() != ".csv":
            raise ValueError("출력 파일 확장자는 .csv여야 합니다.")
        if output == args.input.resolve() or output.is_relative_to((REPO_ROOT / "data/raw").resolve()):
            raise ValueError("입력 파일 또는 data/raw에 출력할 수 없습니다.")
        source = load_source(args.input, args.encoding, config.use_competition, int(config.end_quarter[:4]), args.area_column)
        metadata = save_generation(source, config, output, args.input)
        diagnostics = metadata["validation"]
        metadata_path = output.with_suffix(".metadata.json")
    except (ValueError, OSError, UnicodeError) as error:
        parser.error(str(error))
    print(f"Sales CSV: {output}")
    print(f"Metadata: {metadata_path}")
    print(f"Groups: {diagnostics['groups']:,}; rows: {diagnostics['rows']:,}; crisis rows: {diagnostics['crisis_rows']:,}")
    print(f"Excluded source rows: {source.audit['excluded_rows']:,} (see metadata)")


if __name__ == "__main__":
    main()
