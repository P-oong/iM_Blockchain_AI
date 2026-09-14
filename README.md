# SOHO Resilience Rating

대구 지역 소상공인 중 매출 충격이 발생한 상권·업종에 속한 사업자의 향후 영업 지속 가능성을 평가하는 회복탄력성 평점모형 프로젝트입니다. LOCALDATA 등 공공데이터의 실제 사업자 인허가·폐업 이력과 상권·경쟁 정보를 활용하며, 초기 PoC에서는 향후 활용 예정인 DIP 카드매출 데이터 구조와 유사한 합성 매출 데이터를 결합합니다.

## 핵심 목표

- 위기 노출 사업자의 향후 6개월 생존확률을 예측하고 Recovery Score 및 R1~R5 등의 등급을 산출합니다.
- 실제 DIP 카드매출 데이터 확보 시 합성데이터를 교체할 수 있는 데이터·모델링 파이프라인을 구축합니다.
- 향후 iM뱅크와 대구시의 소상공인 금융·정책지원 대상 선별에 활용합니다.

## 전체 분석 흐름

행정동 × 업종 × 시점별 매출 변화로 위기(Crisis) 탐지 → 위기 노출 실제 사업자 선정 → 공공데이터의 기준시점 이후 6개월 이내 폐업 여부로 라벨링(생존 = 1, 폐업 = 0) → 6개월 생존확률 예측 → Recovery Score 및 등급 산출 → 금융·정책지원 대상 선별

## 현재 단계

- Repository initialization: README 및 최소 폴더 구조 구성
- Public business data collected: 공공 사업자 데이터 수집 완료
- Synthetic sales data: 분기별 합성 매출 생성 코드 작성
- Survival labeling: 사업자 중복·지역 오류 처리와 6·12개월 생존 라벨 생성
- Modeling: 모델링 예정

현재 실제 DIP 카드매출 데이터는 확보 전입니다. 초기 PoC는 모델 성능을 과장하기보다 교체 가능한 파이프라인 구축에 초점을 둡니다. `main`은 개발의 기준점으로 유지하고, 실제 구현은 기능별 브랜치에서 진행합니다.

## 프로젝트 폴더 구조

```text
.
├── README.md
├── data/
│   ├── raw/          # 원본 공공데이터
│   ├── synthetic/    # 합성 DIP 매출 데이터
│   └── processed/    # 모델링용 최종 데이터
├── notebooks/        # 탐색 및 실험
├── src/
│   ├── data/         # 데이터 구축 및 전처리
│   ├── features/     # 파생변수 및 Crisis 정의
│   └── models/       # 평점모형
├── docs/             # 기획 및 데이터 정의서
└── tests/            # 검증
```

빈 폴더는 `.gitkeep`으로 추적합니다.

## 합성 매출 생성

Python 3.10 이상에서 추가 패키지 없이 실행합니다. 저장소 루트에서:

```powershell
python src/data/generate_synthetic_sales.py
```

`data/raw/`의 CSV를 읽어 기본적으로 2015Q1~2025Q4의 `구 × 읍면동 × 업종 × 분기` 매출액·이용건수를 `data/synthetic/synthetic_dip_sales.csv`에 저장합니다. 같은 시드(기본값 42)와 입력으로 재현할 수 있으며, 생성 가정·제외 건수·Crisis 점검 결과는 함께 생성되는 `.metadata.json`에서 확인합니다.

원본 `읍면동`은 아직 행정동으로 검증되지 않았습니다. 현재 출력은 PoC용이며, 실제 지역 매핑은 후속 작업입니다. [생성 기준과 실행 옵션](docs/synthetic_sales.md)을 참고하세요.

## 생존 라벨 데이터셋 생성

```powershell
python -m src.data.build_labeled_dataset
```

기본 위기 기준은 `YoY ≤ -20% AND PeerGap ≤ -10%p`이며, 당시 영업 중인 사업자의 최초 위기 진입 1건에 6·12개월 생존 라벨을 붙입니다. **관측 종료일은 사용자와 합의한 `2025-12-31` 가정**입니다. 관측기간이 부족하면 `NA`로 남기고 기간별 라벨 완료 CSV에서 제외합니다.

결과와 종속변수 비율은 `data/processed/labeling/`에 저장됩니다. 기간, 임계값, 위기 조건과 진입 방식은 [라벨링 기준 및 실행 옵션](docs/labeling.md)에서 확인하세요.
