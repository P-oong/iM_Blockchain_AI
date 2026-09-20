# SOHO 영업지속 평점모형

`data/synthetic/dmdrlt4.csv`로 WOE 로지스틱 모형을 학습하고, 사업자별 영업지속 확률·점수·등급을 산출한다. 종속변수는 **12개월 영업지속 여부**이며, 내부 학습용 `BAD = 1 - Y_12M`은 기간 내 폐업을 뜻한다. 대출 연체·부도 확률이나 개별 점포의 매출 회복 확률을 추정하는 모형은 아니다.

## 실행

프로젝트 루트에서 모델 전용 환경을 준비한다. 이미 설치했다면 설치 단계는 생략한다.

```powershell
py -3 -m venv .venv-model
.\.venv-model\Scripts\python.exe -m pip install -r requirements-models.txt
.\.venv-model\Scripts\python.exe -m src.models.train_scorecard
```

[개발 노트북](../notebooks/최종스코어파이개발_현풍최종_end.ipynb)은 `RUN_TRAINING = False`일 때 마지막 실행 결과를 조회한다. `True`로 바꾸면 `ScorecardConfig`의 설정으로 다시 학습한다. 원본 노트북은 [archive](../notebooks/archive/최종스코어파이개발_현풍최종_end_legacy.ipynb)에 보존했다.

## 데이터와 분할

- 원본 252,424행 중 완전 중복 219,459행을 제거하면 **32,965개 사업자**가 남는다. 중복 상태의 12개월 폐업률은 5.68%지만, 사업자 단위로는 **11.14%**다.
- 12개월·18개월 Y가 있으며 **24개월 Y는 없다**. 기간을 바꿀 때 라벨 존재 여부와 학습·검증의 관측기간을 함께 확인해야 한다.
- 관측 종료일 `2025-12-31`은 사용자가 승인한 **가정**이다. 실제 수집 완전성이 확인된 날짜는 아니다.
- 지하철·서비스인구·주민인구 등 추가 변수는 기준일과 공개 시점을 확인할 수 없어 기본 학습에서 제외한다. 기존 파이프라인의 X 후보 34개 중 행정동 이름을 제외한 33개를 검증 대상으로 사용한다.
- 신규 데이터는 같은 기준일 범위의 기존 코호트 35,088개 중 2,123개를 포함하지 않는다. 제외 이유는 추가 확인이 필요하다.

| 구분 | 기준일 연도 | 사업자 수 | 12개월 폐업 수 | 용도 |
|---|---|---:|---:|---|
| 학습 | 2019–2020 | 14,164 | 1,377 | 구간화·WOE·변수 선택·계수 추정 |
| 검증 | 2022 | 5,558 | 652 | 정규화 강도·확률 보정·등급 결정 |
| OOT | 2024 | 2,821 | 396 | 고정한 모형과 등급의 최종 평가 |
| 간격 확보 | 2021, 2023 | 10,422 | 1,247 | 이후 평가기간으로 라벨 관측기간이 넘어가지 않도록 분할에서 제외 |

분할은 연도 문자열만으로 정하지 않고 기준일과 라벨 관측 종료일을 검사한다. 같은 사업자가 중복으로 학습·검증에 들어가지 않는다. 같은 상권의 서로 다른 사업자는 시점별 집합에 존재할 수 있으므로 OOT 결과는 신규 지역 일반화 성능과 구분해 해석한다.

## 자동 처리 단계

| 단계 | 자동 처리 및 기본 기준 |
|---|---|
| 데이터 검증 | 완전 중복 제거, 관리번호별 값 충돌·타깃·관측기간 확인, 사용 가능 변수 목록 기록 |
| 구간화 | 학습 데이터에서 fine/coarse bin과 WOE 산출, 검증·OOT에는 고정된 구간 적용 |
| 변수 사전 선정 | IV 0.02, 결측률 0.95, 최빈값 비율 0.95 및 상관계수 절댓값 0.7 기준 적용 |
| 로지스틱 추정 | L1 정규화 사용, C 후보 `0.01, 0.05, 0.1, 0.3, 0.5, 1, 2`를 검증 데이터로 비교 |
| 확률 보정 | 검증기간 앞부분에서 양의 기울기 Platt 보정을 적합하고 뒷부분의 Brier·log loss를 비교하여 채택 여부 결정 |
| 평점 | 기준점수 600점, good:bad 기준 odds 20:1, PDO 50점으로 산출; 높은 점수는 낮은 폐업 위험을 의미 |
| 등급 | 검증 데이터의 경험적 폐업률이 순서대로 증가하도록 최대 5개 등급을 만들고 OOT에 고정 적용 |
| 모형 검증 | AUC·KS·확률오차·보정도·연도별 결과·등급별 분포·PSI와 변수별 점수 기여도 저장 |

검증 데이터를 선택·보정·등급화에 사용하므로 최종 성능 판단은 별도 OOT 결과를 기준으로 한다. 임계값은 통계적 선별을 위한 초기 설정이며, 금융지원 의사결정의 승인 기준을 뜻하지 않는다.

## 산출물

모형 파일은 `src/models` 아래 실행별 모델 디렉터리에, 보고서는 `data/processed/scorecard` 아래 실행별 보고서 디렉터리에 저장한다. `data/processed/scorecard/latest_run.json`에서 마지막 실행 경로를 확인한다. 각 단계의 표는 CSV와 JSON으로 함께 저장한다. 중간 단계 그래프는 해당 실행의 `figures/`에 PNG로 자동 저장하며, `figure_manifest.csv/json`에서 이미지와 근거 통계 파일의 대응을 확인한다.

| 파일명 접두사 | 내용 |
|---|---|
| `data_audit`, `feature_inventory`, `split_summary` | 데이터 품질·사용/제외 변수·분할 규모 |
| `feature_screening`, `woe_fine`, `woe_coarse`, `woe_summary` | 변수별 통계 및 구간별 WOE·IV |
| `correlation_pairs`, `selection_log`, `c_tuning`, `coefficients` | 상관성 제거·선정 이력·정규화 비교·최종 계수 |
| `calibration_comparison`, `calibration_curve` | 확률 보정 채택 근거 및 보정도 |
| `metrics`, `metrics_by_year` | 집합별·연도별 성능 |
| `grade_definition`, `grade_performance` | 등급 경계와 등급별 관측 결과 |
| `psi_summary`, `psi_detail` | 변수·점수 분포 변화 |
| `scorecard`, `scored_businesses`, `score_contributions`, `score_variance` | 구간별 배점·사업자별 확률/점수/등급·기여도·점수 분산 | 
| `human_review`, `run_summary` | 사람의 검토가 필요한 항목 및 실행 요약 |
| `figure_manifest`, `figures/*.png` | Fine/Coarse 구간·IV·상관성·선정·ROC/KS/PR·확률보정·PSI·등급·연도 안정성 그래프 |

## 사람이 결정할 부분

1. **라벨의 신뢰성**: 폐업 이력의 완전한 관측 종료일, 관리번호의 사업자 단위 적절성, 휴업·이전·재개업의 처리.
2. **추가 변수 사용 승인**: 지하철·인구 변수의 원천, 집계기간, 공개일, 기준일 이전 사용 가능 여부와 2,123개 사업자 제외 이유.
3. **변수의 업무적 타당성**: 높은 IV가 누수 때문인지, WOE와 계수 방향이 납득 가능한지, 지역·업종의 대리효과가 적절한지.
4. **서비스 정책**: 12·18·24개월 중 실제 의사결정 기간, 등급 수, 지원 대상 점수 기준, 오선정 비용과 지원 규모.
5. **적용 승인**: 실제 매출데이터로 교체한 뒤의 재검증, OOT·등급 순서·분포 안정성 및 운영 중 재학습 기준. 합성 매출을 사용한 현재 결과는 PoC다.

통계적 통과만으로 운영 사용이 승인되지는 않는다. 자동화 결과와 검토 항목을 함께 확인한 후 적용 여부를 결정한다.

구현 참고: [scikit-learn LogisticRegression](https://scikit-learn.org/1.7/modules/generated/sklearn.linear_model.LogisticRegression.html), [확률 보정](https://scikit-learn.org/1.8/modules/calibration.html).

## 설정 변경과 저장 모델 재사용

`src/models/scorecard_config.json`에서 변수선정·등급·점수 기준을 변경할 수 있다.

```powershell
.\.venv-model\Scripts\python.exe -m src.models.train_scorecard --config src/models/scorecard_config.json
```

추가 변수는 `extra_feature_availability`에 `변수명: 사용가능일_컬럼명`을 지정해야 한다. 해당 날짜가 모든 행의 t0 이하인지 검사하며, 근거 없는 일괄 날짜를 만들어 채워 넣으면 안 된다.

```python
from src.models.scorecard import ScorecardModel
model = ScorecardModel.load("src/models/artifacts/<실행명>/scorecard_model.json")
predictions = model.predict(new_data)
```

`score`는 600점·PDO 50의 로그오즈 평점이며 범위를 0~100으로 제한하지 않는다. `recovery_score`는 `100 × (1 - p_bad)`인 별도 0~100점 표시값이다. 등급은 반올림 전 확률로 결정한다. `grade_cutoffs`의 빈 점수 경계는 무한한 상·하한을 뜻한다.

PR-AUC는 선형 보간 면적이며, AP(`average_precision`)를 함께 제공한다. 동일 확률이 많은 경우 두 지표가 크게 달라질 수 있으므로 폐업률 기준선과 비교할 때는 AP도 확인한다.
