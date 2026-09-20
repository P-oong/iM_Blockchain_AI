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
from src.features.independent import FeatureConfig, add_independent_features, feature_definitions

from .businesses import load_businesses
from .labeling import LabelConfig, build_cohort, cohort_columns, label_distribution, quarter_end
from .provenance import fingerprint, find_input_csv


REPO_ROOT = Path(__file__).resolve().parents[2]
FEATURE_COLUMNS = ["구", "읍면동", "업종", "업력_일수", "매출액", "이용건수",
                   "sales_yoy", "peer_yoy", "peer_gap"]
BASE_FEATURE_DEFINITIONS = {
    "구": "지역코드 기준 정규화한 구·군", "읍면동": "선택한 지역 열의 호환 별칭; 수정 원본은 행정동",
    "업종": "원본 업종 분류", "업력_일수": "t0와 인허가일의 차이",
    "매출액": "t0 분기의 지역·업종 합성 매출(원)", "이용건수": "t0 분기의 합성 거래 수",
    "sales_yoy": "전년 동분기 대비 매출 증감률", "peer_yoy": "비교 가능한 동일 업종 시장의 매출 합계 YoY",
    "peer_gap": "sales_yoy - peer_yoy",
}


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
    feature_config: FeatureConfig | None = None, area_column: str = "auto",
) -> dict:
    feature_config = feature_config or FeatureConfig()
    business_fingerprint = fingerprint(business_path)
    sales_metadata_path = sales_path.with_suffix(".metadata.json")
    if sales_metadata_path.exists():
        sales_metadata = json.loads(sales_metadata_path.read_text(encoding="utf-8"))
        original_fingerprint = sales_metadata.get("source_fingerprint", {})
        if original_fingerprint.get("sha256") != business_fingerprint["sha256"]:
            raise ValueError("매출 생성 원본과 현재 사업자 CSV가 다릅니다. --regenerate-sales로 합성 매출부터 다시 생성하세요.")
    output_dir = output_dir.resolve()
    protected = (REPO_ROOT / "data/raw", REPO_ROOT / "data/synthetic")
    if any(output_dir == path or output_dir.is_relative_to(path) for path in protected):
        raise ValueError("원본 또는 매출 입력 폴더 안에는 라벨링 결과를 쓸 수 없습니다.")
    output_names = ["business_crisis_cohort.csv", "label_distribution.csv", "label_distribution_by_year.csv",
                    "excluded_businesses.csv", "market_crisis.csv", "labeling.metadata.json",
                    "feature_dictionary.csv", "feature_missingness.csv", "label_distribution_common_cohort.csv"]
    output_names += [f"labeled_{months}m.csv" for months in label_config.horizons]
    inputs = {business_path.resolve(), sales_path.resolve()}
    if any((output_dir / name).resolve() in inputs for name in output_names):
        raise ValueError("출력 파일이 입력 파일과 겹칩니다.")
    print("Loading market sales and detecting crisis...", flush=True)
    markets = load_sales(sales_path, crisis_config, sales_encoding)
    print("Reading and deduplicating public business records...", flush=True)
    source = load_businesses(business_path, encoding, area_column)
    cohort, cohort_exclusions = build_cohort(source.businesses, markets, label_config)
    print("Recomputing independent variables from historical snapshots...", flush=True)
    cohort = add_independent_features(cohort, source.businesses, markets, feature_config)
    distributions = [label_distribution(cohort, months) for months in label_config.horizons]
    output_dir.mkdir(parents=True, exist_ok=True)
    output_files = []

    def save(name: str, rows, columns: list[str]) -> None:
        write_csv(output_dir / name, rows, columns)
        output_files.append(name)

    added_definitions = feature_definitions(feature_config)
    all_definitions = {**BASE_FEATURE_DEFINITIONS, **added_definitions}
    added_columns = ["행정동코드", *added_definitions]
    base_columns = cohort_columns(label_config)[:14] + added_columns
    all_columns = base_columns + cohort_columns(label_config)[14:]
    save("business_crisis_cohort.csv", cohort, all_columns)
    for months in label_config.horizons:
        columns = base_columns + [f"horizon_end_{months}M", f"Y_{months}M", f"label_status_{months}M"]
        save(f"labeled_{months}m.csv", (row for row in cohort if row[f"Y_{months}M"] is not None), columns)

    distribution_columns = list(distributions[0])
    save("label_distribution.csv", distributions, distribution_columns)
    common_rows = [row for row in cohort if all(row[f"Y_{months}M"] is not None for months in label_config.horizons)]
    common_distributions = [label_distribution(common_rows, months) for months in label_config.horizons]
    save("label_distribution_common_cohort.csv", common_distributions, distribution_columns)
    by_year = []
    for year in sorted({row["t0"][:4] for row in cohort}):
        year_rows = [row for row in cohort if row["t0"].startswith(year)]
        for months in label_config.horizons:
            by_year.append({"cohort_year": year, **label_distribution(year_rows, months)})
    save("label_distribution_by_year.csv", by_year, ["cohort_year", *distribution_columns])
    feature_allowlist = FEATURE_COLUMNS + list(added_definitions)
    save("feature_dictionary.csv", ({"feature": name, "definition": definition} for name, definition in all_definitions.items()),
         ["feature", "definition"])
    save("feature_missingness.csv", ({"feature": name, "missing": sum(row.get(name) is None for row in cohort),
                                      "total": len(cohort)} for name in feature_allowlist), ["feature", "missing", "total"])

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
        "pipeline_version": 2,
        "inputs": {"businesses": str(business_path.resolve()), "sales": str(sales_path.resolve())},
        "input_fingerprints": {"businesses": business_fingerprint, "sales": fingerprint(sales_path)},
        "crisis_config": asdict(crisis_config),
        "label_config": {**asdict(label_config), "observation_end": label_config.observation_end.isoformat()},
        "observation_note": observation_note,
        "feature_config": {**asdict(feature_config), "history_start": feature_config.history_start.isoformat()},
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
        "label_distributions_common_cohort": common_distributions,
        "feature_allowlist": feature_allowlist,
        "feature_definitions": all_definitions,
        "output_files": output_files,
        "assumptions": [
            "Y=1은 위기 이후 영업지속이며 개별 점포 매출의 회복을 뜻하지 않습니다.",
            "매출은 합성이고 폐업 결과는 공공데이터에서 가져옵니다. 실제 성능 검증용이 아닙니다.",
            "분기 매출이 분기말에 이용 가능하다고 가정합니다. 실제 공표 지연은 반영하지 않았습니다.",
            "기준일은 분기말이며 n개월 뒤 날짜도 월말을 보존합니다. 폐업 판정 구간은 (t0, horizon_end]입니다.",
            "관측기간 전체가 관측 종료일 이내인 경우만 Y를 부여합니다. 미성숙 기간은 조기 폐업을 알더라도 NA입니다.",
            "첫 위기 선택 후 관측기간을 판단합니다. 사업자별 하나의 기준일을 12개월·18개월·24개월 등 모든 라벨에 공통 사용합니다.",
            "최초 계산 가능한 위기 구간은 관측 범위 내 진입으로 허용합니다. 관측 범위 이전의 위기는 알 수 없습니다.",
            "읍면동은 선택한 지역 열의 호환 별칭입니다. 새 원본은 행정동과 행정동코드를 사용하고 원본 내부 코드 매핑을 정규화합니다.",
            "지역·업종은 원본에 반복된 고정값을 사용합니다. 과거 이전·업종변경·행정구역 변경은 복원하지 않았습니다.",
            "동명이 같아도 구가 다르면 별도 시장입니다. 입력 매출과 같은 구·읍면동·업종만 연결합니다.",
            "업종 비교치는 t와 t-4 모두 관측된 시장들의 매출 합계 비율이며 공식 대구 전체 통계가 아닙니다.",
            "폐업일·인허가일 원문과 미래 파생변수는 학습용 CSV에 복사하지 않습니다. feature_allowlist만 X 후보로 사용하세요.",
            "기간별 CSV는 해당 기간의 단일 Y만 포함합니다. 전체 cohort의 다른 기간 Y와 목표 종료일·관측상태는 X로 사용하지 마세요.",
            "원본 연간 경쟁지표는 복사하지 않습니다. 과거 개·폐업 사건으로 직전 분기 영업수와 후행 창의 진입·폐업률을 새로 계산합니다.",
            "과거 폐업일은 해당 스냅샷 날짜까지의 사건만 집계합니다. 창이 history_start보다 앞서면 진입·폐업률 등은 NA입니다.",
            "지역 사업체 수는 확보한 업종·정제된 사업자 범위의 수이며 대구의 모든 업종을 포괄하는 공식 수치가 아닙니다.",
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
    parser.add_argument("--horizons", nargs="+", type=int, default=[12, 18, 24], help="생존 관측기간(개월)")
    parser.add_argument("--yoy-threshold", type=float, default=-0.20)
    parser.add_argument("--peer-gap-threshold", type=float, default=-0.10)
    parser.add_argument("--crisis-rule", choices=["and", "or", "yoy-only"], default="and")
    parser.add_argument("--min-consecutive-quarters", type=int, default=1)
    parser.add_argument("--entry-mode", choices=["episode-start", "first-exposure"], default="episode-start")
    parser.add_argument("--encoding", default="utf-8-sig", help="원본 사업자 CSV 인코딩")
    parser.add_argument("--sales-encoding", default="utf-8-sig")
    parser.add_argument("--area-column", default="auto", help="auto는 행정동 우선, 없으면 읍면동")
    parser.add_argument("--regenerate-sales", action="store_true", help="동일 원본으로 합성 매출부터 재생성")
    parser.add_argument("--sales-start-quarter", default="2015Q1")
    parser.add_argument("--sales-end-quarter", default="2025Q4")
    parser.add_argument("--sales-seed", type=int, default=42)
    parser.add_argument("--sales-shock-probability", type=float, default=0.08)
    parser.add_argument("--business-lag-quarters", type=int, default=1)
    parser.add_argument("--business-window-quarters", type=int, default=4)
    parser.add_argument("--sales-window-quarters", type=int, default=4)
    parser.add_argument("--history-start", default="2015-01-01")
    args = parser.parse_args()
    try:
        crisis_config = CrisisConfig(args.yoy_threshold, args.peer_gap_threshold,
                                     args.crisis_rule, args.min_consecutive_quarters)
        label_config = LabelConfig(date.fromisoformat(args.observation_end), tuple(args.horizons), args.entry_mode)
        feature_config = FeatureConfig(args.business_lag_quarters, args.business_window_quarters,
                                       args.sales_window_quarters, date.fromisoformat(args.history_start))
        if args.businesses is None:
            args.businesses = find_input_csv(REPO_ROOT / "data/raw")
        if args.regenerate_sales:
            from .generate_synthetic_sales import Config, load_source, save_generation
            print("Regenerating synthetic sales from the selected source...", flush=True)
            sales_config = Config(args.sales_start_quarter, args.sales_end_quarter, args.sales_seed, args.sales_shock_probability)
            catalog = load_source(args.businesses, args.encoding, through_year=int(args.sales_end_quarter[:4]), area_column=args.area_column)
            save_generation(catalog, sales_config, args.sales, args.businesses)
        metadata = run_pipeline(args.businesses, args.sales, args.output_dir, crisis_config,
                                label_config, args.encoding, args.sales_encoding, args.observation_note,
                                feature_config, args.area_column)
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
