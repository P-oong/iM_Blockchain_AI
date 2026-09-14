"""Build reproducible business survival cohorts from sales and public records.

Run from the repository root: python -m src.data.build_labeled_dataset --help
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from dataclasses import asdict
from datetime import date
from pathlib import Path

from src.features.crisis import CrisisConfig, load_sales

from .businesses import load_businesses
from .labeling import LabelConfig, build_cohort, cohort_columns, label_distribution, quarter_end


REPO_ROOT = Path(__file__).resolve().parents[2]
FEATURE_COLUMNS = ["구", "읍면동", "업종", "업력_일수", "매출액", "이용건수",
                   "sales_yoy", "peer_yoy", "peer_gap"]


def write_csv(path: Path, rows, columns: list[str]) -> None:
    """Write completed UTF-8 BOM files atomically; None is explicitly NA."""
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: "NA" if row[column] is None else row[column] for column in columns})
    temp_path.replace(path)


def run_pipeline(
    business_path: Path, sales_path: Path, output_dir: Path,
    crisis_config: CrisisConfig, label_config: LabelConfig,
    encoding: str = "utf-8-sig", sales_encoding: str = "utf-8-sig",
    observation_note: str = "관측 종료일까지 수집이 완전하다고 가정. 원본만으로 완전성은 확인되지 않음.",
) -> dict:
    output_dir = output_dir.resolve()
    protected = (REPO_ROOT / "data/raw", REPO_ROOT / "data/synthetic")
    if any(output_dir == path or output_dir.is_relative_to(path) for path in protected):
        raise ValueError("원본 또는 매출 입력 폴더 안에는 라벨링 결과를 쓸 수 없습니다.")
    output_names = ["business_crisis_cohort.csv", "label_distribution.csv", "label_distribution_by_year.csv",
                    "excluded_businesses.csv", "market_crisis.csv", "labeling.metadata.json"]
    output_names += [f"labeled_{months}m.csv" for months in label_config.horizons]
    inputs = {business_path.resolve(), sales_path.resolve()}
    if any((output_dir / name).resolve() in inputs for name in output_names):
        raise ValueError("출력 파일이 입력 파일과 겹칩니다.")
    print("Loading market sales and detecting crisis...", flush=True)
    markets = load_sales(sales_path, crisis_config, sales_encoding)
    print("Reading and deduplicating public business records...", flush=True)
    source = load_businesses(business_path, encoding)
    cohort, cohort_exclusions = build_cohort(source.businesses, markets, label_config)
    distributions = [label_distribution(cohort, months) for months in label_config.horizons]
    output_dir.mkdir(parents=True, exist_ok=True)
    output_files = []

    def save(name: str, rows, columns: list[str]) -> None:
        write_csv(output_dir / name, rows, columns)
        output_files.append(name)

    all_columns = cohort_columns(label_config)
    save("business_crisis_cohort.csv", cohort, all_columns)
    for months in label_config.horizons:
        columns = all_columns[:14] + [f"horizon_end_{months}M", f"Y_{months}M", f"label_status_{months}M"]
        save(f"labeled_{months}m.csv", (row for row in cohort if row[f"Y_{months}M"] is not None), columns)

    distribution_columns = list(distributions[0])
    save("label_distribution.csv", distributions, distribution_columns)
    by_year = []
    for year in sorted({row["t0"][:4] for row in cohort}):
        year_rows = [row for row in cohort if row["t0"].startswith(year)]
        for months in label_config.horizons:
            by_year.append({"cohort_year": year, **label_distribution(year_rows, months)})
    save("label_distribution_by_year.csv", by_year, ["cohort_year", *distribution_columns])

    exclusions = source.exclusions + cohort_exclusions
    save("excluded_businesses.csv", exclusions, ["관리번호", "구", "읍면동", "업종", "stage", "reason"])
    market_rows = ({
        "구": row.group[0], "읍면동": row.group[1], "업종": row.group[2],
        "기준분기": f"{row.quarter // 4:04d}Q{row.quarter % 4 + 1}",
        "t0": quarter_end(row.quarter).isoformat(), "매출액": row.sales,
        "sales_yoy": row.sales_yoy, "peer_yoy": row.peer_yoy, "peer_gap": row.peer_gap,
        "crisis_flag": None if row.crisis is None else int(row.crisis),
        "episode_start": int(row.episode_start),
    } for row in markets)
    save("market_crisis.csv", market_rows,
         ["구", "읍면동", "업종", "기준분기", "t0", "매출액", "sales_yoy", "peer_yoy", "peer_gap", "crisis_flag", "episode_start"])

    metadata = {
        "pipeline_version": 1,
        "inputs": {"businesses": str(business_path.resolve()), "sales": str(sales_path.resolve())},
        "crisis_config": asdict(crisis_config),
        "label_config": {**asdict(label_config), "observation_end": label_config.observation_end.isoformat()},
        "observation_note": observation_note,
        "source_audit": source.audit,
        "market_audit": {
            "market_quarter_rows": len(markets), "groups": len({row.group for row in markets}),
            "unknown_crisis_rows": sum(row.crisis is None for row in markets),
            "crisis_rows": sum(row.crisis is True for row in markets),
            "episode_start_rows": sum(row.episode_start for row in markets),
        },
        "cohort_audit": {
            "cohort_rows": len(cohort), "unique_businesses": len({row["관리번호"] for row in cohort}),
            "excluded_after_source_cleaning": len(cohort_exclusions),
            "cohort_exclusion_reasons": dict(sorted(Counter(row["reason"] for row in cohort_exclusions).items())),
            "accounted_business_ids": len(cohort) + len(exclusions),
        },
        "label_distributions": distributions,
        "feature_allowlist": FEATURE_COLUMNS,
        "output_files": output_files,
        "assumptions": [
            "Y=1은 위기 이후 영업지속이며 개별 점포 매출의 회복을 뜻하지 않습니다.",
            "매출은 합성이고 폐업 결과는 공공데이터에서 가져옵니다. 실제 성능 검증용이 아닙니다.",
            "분기 매출이 분기말에 이용 가능하다고 가정합니다. 실제 공표 지연은 반영하지 않았습니다.",
            "기준일은 분기말이며 n개월 뒤 날짜도 월말을 보존합니다. 폐업 판정 구간은 (t0, horizon_end]입니다.",
            "관측기간 전체가 관측 종료일 이내인 경우만 Y를 부여합니다. 미성숙 기간은 조기 폐업을 알더라도 NA입니다.",
            "첫 위기 선택 후 관측기간을 판단합니다. 사업자별 하나의 기준일을 6개월·12개월 등 모든 라벨에 공통 사용합니다.",
            "최초 계산 가능한 위기 구간은 관측 범위 내 진입으로 허용합니다. 관측 범위 이전의 위기는 알 수 없습니다.",
            "읍면동은 원본 문자열의 형식·건물명 필터만 적용하며 행정동으로 검증하거나 추정 변환하지 않습니다.",
            "지역·업종은 원본에 반복된 고정값을 사용합니다. 과거 이전·업종변경·행정구역 변경은 복원하지 않았습니다.",
            "동명이 같아도 구가 다르면 별도 시장입니다. 입력 매출과 같은 구·읍면동·업종만 연결합니다.",
            "업종 비교치는 t와 t-4 모두 관측된 시장들의 매출 합계 비율이며 공식 대구 전체 통계가 아닙니다.",
            "폐업일·인허가일 원문과 미래 파생변수는 학습용 CSV에 복사하지 않습니다. feature_allowlist만 X 후보로 사용하세요.",
            "기간별 CSV는 해당 기간의 단일 Y만 포함합니다. 전체 cohort의 다른 기간 Y와 목표 종료일·관측상태는 X로 사용하지 마세요.",
            "매출 생성 시 사용하지 않은 원본 경쟁지표도 라벨링 결과에는 복사하지 않습니다.",
        ],
    }
    if metadata["cohort_audit"]["accounted_business_ids"] != source.audit["unique_business_ids"]:
        raise AssertionError("사업자별 포함·제외 집계가 원본 관리번호 수와 일치하지 않습니다.")
    metadata_path = output_dir / "labeling.metadata.json"
    temp_metadata = metadata_path.with_suffix(".json.tmp")
    temp_metadata.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp_metadata.replace(metadata_path)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--businesses", type=Path, help="원본 CSV. 생략 시 data/raw의 유일한 CSV")
    parser.add_argument("--sales", type=Path, default=REPO_ROOT / "data/synthetic/synthetic_dip_sales.csv")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "data/processed/labeling")
    parser.add_argument("--observation-end", default="2025-12-31", help="완전 관측 종료일 가정, YYYY-MM-DD")
    parser.add_argument("--observation-note", default="2025-12-31 기본값은 사용자와 합의한 가정. 지정 종료일까지의 수집 완전성은 원본만으로 확인되지 않음.")
    parser.add_argument("--horizons", nargs="+", type=int, default=[6, 12], help="생존 관측기간(개월)")
    parser.add_argument("--yoy-threshold", type=float, default=-0.20)
    parser.add_argument("--peer-gap-threshold", type=float, default=-0.10)
    parser.add_argument("--crisis-rule", choices=["and", "or", "yoy-only"], default="and")
    parser.add_argument("--min-consecutive-quarters", type=int, default=1)
    parser.add_argument("--entry-mode", choices=["episode-start", "first-exposure"], default="episode-start")
    parser.add_argument("--encoding", default="utf-8-sig", help="원본 사업자 CSV 인코딩")
    parser.add_argument("--sales-encoding", default="utf-8-sig")
    args = parser.parse_args()
    try:
        crisis_config = CrisisConfig(args.yoy_threshold, args.peer_gap_threshold,
                                     args.crisis_rule, args.min_consecutive_quarters)
        label_config = LabelConfig(date.fromisoformat(args.observation_end), tuple(args.horizons), args.entry_mode)
        if args.businesses is None:
            candidates = sorted((REPO_ROOT / "data/raw").glob("*.csv"))
            if len(candidates) != 1:
                raise ValueError("data/raw에 CSV가 하나가 아닙니다. --businesses를 지정하세요.")
            args.businesses = candidates[0]
        metadata = run_pipeline(args.businesses, args.sales, args.output_dir, crisis_config,
                                label_config, args.encoding, args.sales_encoding, args.observation_note)
    except (ValueError, OSError, UnicodeError, OverflowError) as error:
        parser.error(str(error))
    print(f"Observation end (assumed): {args.observation_end}")
    print(f"Unique businesses: {metadata['source_audit']['unique_business_ids']:,}; cohort: {metadata['cohort_audit']['cohort_rows']:,}")
    for summary in metadata["label_distributions"]:
        survival_pct = summary["survival_pct_among_labeled"]
        closure_pct = summary["closure_pct_among_labeled"]
        percentages = "NA" if survival_pct is None else f"Y=1 {survival_pct:.2f}%, Y=0 {closure_pct:.2f}%"
        print(f"{summary['horizon_months']}M: labeled={summary['labeled']:,}, Y=1={summary['survived_1']:,}, "
              f"Y=0={summary['closed_0']:,}, NA={summary['unobserved_na']:,}; {percentages}")
    print(f"Output: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
