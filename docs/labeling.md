# 위기 노출 사업자의 생존 라벨링

합성 매출로 위기 대상자를 선정하고 공공데이터의 인허가일·폐업일로 영업지속 여부를 만듭니다. `Crisis`는 진입 조건이고 `Y=1`은 미래 영업지속, `Y=0`은 미래 폐업입니다. 매출 회복을 직접 측정하는 라벨은 아닙니다.

## 실행과 설정 변경

Python 3.10 이상, 추가 패키지 없이 저장소 루트에서 실행합니다.

```powershell
python -m src.data.build_labeled_dataset
```

설정을 바꾼 결과는 별도 폴더에 보관하면 비교하기 쉽습니다.

```powershell
python -m src.data.build_labeled_dataset --horizons 6 12 --yoy-threshold -0.15 --peer-gap-threshold -0.10 --output-dir data/processed/labeling_yoy15
python -m src.data.build_labeled_dataset --horizons 12 --crisis-rule yoy-only --yoy-threshold -0.25 --min-consecutive-quarters 2 --output-dir data/processed/labeling_yoy25_2q
```

| 옵션 | 기본값 | 의미 |
| --- | --- | --- |
| `--businesses` | `data/raw/`의 유일한 CSV | 공공 사업자 원본 |
| `--sales` | `data/synthetic/synthetic_dip_sales.csv` | 위기 탐지에 쓸 매출 |
| `--output-dir` | `data/processed/labeling` | 결과 폴더; 같은 파일은 재실행 시 갱신 |
| `--horizons` | `6 12` | 개월 단위 관측기간. `3 6 12` 등 지정 가능 |
| `--observation-end` | `2025-12-31` | 완전 관측 종료일 **가정** |
| `--observation-note` | 수집 완전성 미확인 안내 | 관측 종료일의 근거·가정 기록 |
| `--yoy-threshold` | `-0.20` | 전년 동분기 매출 증감률 기준. `-0.20`은 -20% |
| `--peer-gap-threshold` | `-0.10` | 동일 업종 대비 격차. `-0.10`은 -10%p |
| `--crisis-rule` | `and` | 두 기준 모두 충족. `or`, `yoy-only`도 지원 |
| `--min-consecutive-quarters` | `1` | 연속 충족 분기 수. 2이면 두 번째 분기에 위기 확정 |
| `--entry-mode` | `episode-start` | 위기 구간 시작 시점에 영업 중인 사업자 중 최초 진입 |
| `--encoding`, `--sales-encoding` | `utf-8-sig` | 각각 사업자·매출 CSV 입력 인코딩 |

관측 종료일을 늘리려면 해당 날짜까지 폐업 이력이 수집되었는지 먼저 확인해야 합니다. 원본의 `기준일자=2026-12-31`과 마지막 폐업일은 수집 완전성의 근거가 아닙니다. 현재 `2025-12-31`도 검증된 수집 종료일이 아닌 사용자와 합의한 가정입니다.

## 모집단과 Y 정의

1. `관리번호`별로 연도 반복 행을 한 사업자로 합칩니다. 인허가·폐업일의 비결측 값이나 지역·업종이 충돌하면 제외합니다. 날짜 오류와 폐업상태만 있고 폐업일이 없는 경우도 제외합니다.
2. 합성 매출과 동일한 지역 필터로 `103동`, `상가동` 등의 명백한 오류를 제외합니다. 남은 원본 `구 + 읍면동 + 업종`이 매출 키와 정확히 일치해야 합니다. 임의의 행정동 추정이나 다른 구의 동명 연결은 하지 않습니다.
3. 전년 동분기 매출 변화율과 동일 업종 비교치를 계산합니다. 비교치는 현재와 전년 동분기 모두 매출이 있는 시장의 합계를 사용하며, 지역별 성장률의 단순평균이 아닙니다. 전년 매출이 없거나 0이면 YoY는 미확인입니다.
4. 위기 구간 시작 분기말을 후보 `t0`로 삼습니다. `인허가일 ≤ t0`, `폐업일 > t0 또는 폐업일 없음`인 사업자만 포함합니다. 각 사업자는 영업 중에 맞은 최초 후보 한 건만 선택합니다.
5. `t0`에 n개월을 더하되 월말을 보존합니다. 예: `2024-06-30 + 6개월 = 2024-12-31`.
6. 목표 종료일이 관측 종료일보다 늦으면 **Y=NA**입니다. 관측기간 전체가 확보된 경우에만 `(t0, 목표 종료일]` 폐업을 `0`, 목표 종료일까지 미폐업을 `1`로 표시합니다.

단기 폐업이 이미 확인되었더라도 목표기간 전체가 관측되지 않으면 NA로 둡니다. 이는 기간 전체가 관측된 코호트만 비교하기 위한 정책입니다. 6개월과 12개월은 **같은 t0**에서 계산하며 관측기간별 성숙 여부만 달라집니다. 폐업일이 t0와 같으면 애초에 영업 중 모집단이 아닙니다.

`episode-start`에서는 위기가 진행되는 중간에 개업한 사업자는 다음 위기 구간 시작까지 기다립니다. 이런 사업자도 최초로 노출된 분기에 포함하려면 `--entry-mode first-exposure`를 사용합니다. 두 방식 모두 사업자별 최대 한 행입니다.

처음 계산 가능한 분기가 위기면 관측 범위 내 최초 위기 진입으로 허용합니다. 그 이전부터 위기였는지는 알 수 없습니다. 이후 데이터 누락으로 정상→위기 전환이 확인되지 않는 경우에는 새 진입을 임의로 만들지 않습니다. 연속 분기 조건을 높여도 시작점을 과거로 소급하지 않습니다.

## 결과 파일

| 파일 | 내용 |
| --- | --- |
| `business_crisis_cohort.csv` | 사업자별 최초 위기 1행, 모든 요청 기간의 Y와 NA 포함 |
| `labeled_6m.csv`, `labeled_12m.csv` | 해당 기간 Y가 확정된 행만 포함; 해당 기간의 Y 열만 출력 |
| `label_distribution.csv` | 기간별 생존·폐업·NA 건수와 비율 |
| `label_distribution_by_year.csv` | 위기 진입연도별 같은 집계 |
| `market_crisis.csv` | 지역·업종·분기별 YoY, 비교치, Crisis, 진입 여부 |
| `excluded_businesses.csv` | 제외된 관리번호와 단계·사유 |
| `labeling.metadata.json` | 실행 설정, 관측 가정, 입력·제외 집계, X 허용 열, 이번 실행 결과 목록 |

CSV는 UTF-8 BOM이며 결측은 `NA`입니다. **생존·폐업 비율의 분모는 Y가 확정된 행만**, NA 비율의 분모는 전체 최초 위기 코호트입니다. 6개월·12개월은 관측 가능한 표본 수가 달라질 수 있습니다.

설정이 다른 실행은 별도 `--output-dir`를 권장합니다. 다른 기간으로 재실행해도 기존 파일을 임의로 삭제하지 않으므로, 같은 폴더를 재사용할 때는 `labeling.metadata.json`의 `output_files`에 있는 파일이 이번 결과입니다.

## 코드에서 조정할 위치

- `src/data/market_keys.py`: 공통 지역 문자열 정리·형식 필터
- `src/data/businesses.py`: 원본 읽기, 사업자 중복 제거, 날짜·지역 오류 처리
- `src/features/crisis.py`: `CrisisConfig`, 매출 변화율, 위기 조건과 구간 시작
- `src/data/labeling.py`: `LabelConfig`, 최초 노출 선택, 월말 관측기간, Y와 비율
- `src/data/build_labeled_dataset.py`: CLI, 파일 저장과 실행 메타데이터

학습 X 후보는 메타데이터의 `feature_allowlist`에 명시합니다. 원본의 폐업일·미래 파생변수는 결과에 복사하지 않습니다. 다른 기간의 Y, 목표 종료일, 관측상태도 X로 사용하면 안 됩니다.

지금 결과는 지역 문자열이 완전히 검증되지 않은 합성 매출 기반 PoC입니다. 과거 주소·업종 변경, 행정구역 변화와 실제 매출 공표 지연도 아직 반영하지 않았습니다. 실제 적용 전에 지역 매핑과 관측 종료일을 검증해야 합니다.

```powershell
python -B -m unittest discover -s tests -v
```
