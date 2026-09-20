# ============================================================
# TREE MODEL
# 12M / 18M
# 불균형 처리 비교 + Feature Selection + Final Test
#
# 필요 패키지
# pip install pandas numpy scikit-learn imbalanced-learn
# pip install xgboost lightgbm openpyxl joblib
# ============================================================


# ============================================================
# 1. IMPORT
# ============================================================

from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import joblib
from tqdm.auto import tqdm
import time

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    brier_score_loss
)

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

warnings.filterwarnings("ignore")


# ============================================================
# 2. CONFIG
# ============================================================

RANDOM_STATE = 42

df = pd.read_csv(r'C:\Users\DC\2026\IMBANK\iM_Blockchain_AI\data\processed\variable\dmdrlt5.csv')

RESULT_DIR = Path("./tree_model_results")
RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# ------------------------------------------------------------
# 검증 범위
# ------------------------------------------------------------

FIRST_VALID_QUARTER = "2020Q1"
LAST_VALID_QUARTER = "2024Q1"

# ------------------------------------------------------------
# Final Test
# ------------------------------------------------------------

FINAL_TEST_QUARTER = "2024Q2"


# ------------------------------------------------------------
# 분석 대상
# ------------------------------------------------------------

TARGET_CONFIG = {

    "12M": {
        "target": "Y_12M",
        "status": "label_status_12M"
    },

    "18M": {
        "target": "Y_18M",
        "status": "label_status_18M"
    }
}


# ------------------------------------------------------------
# Feature Selection 설정
# ------------------------------------------------------------
#
# 중요도 누적합 90%에 해당하는 변수 선택
#
# 너무 적게 선택되는 것을 방지하기 위해
# 최소 15개
#
# 너무 많이 선택되는 것을 방지하기 위해
# 최대 40개
# ------------------------------------------------------------

IMPORTANCE_CUM_THRESHOLD = 0.90

MIN_FEATURES = 15
MAX_FEATURES = 40


# ============================================================
# 3. 제외 변수
# ============================================================

DROP_COLS = [

    # ID
    "관리번호",

    # 시간
    "기준분기",
    "t0",

    # 선정 조건
    "crisis_flag",
    "episode_start",

    # 행정코드
    "행정동코드",

    # -----------------------------
    # 12M
    # -----------------------------
    "horizon_end_12M",
    "Y_12M",
    "label_status_12M",

    # -----------------------------
    # 18M
    # -----------------------------
    "horizon_end_18M",
    "Y_18M",
    "label_status_18M",

    # -----------------------------
    # 24M
    # -----------------------------
    "horizon_end_24M",
    "Y_24M",
    "label_status_24M",

    # -----------------------------
    # 중복 업력 변수
    # 업력_연수만 남김
    # -----------------------------
    "업력_일수",
    "업력_개월수",

    # -----------------------------
    # 지역 매핑용 변수
    # -----------------------------
    "서비스인구_매핑동",
]


# ============================================================
# 4. 기본 전처리
# ============================================================

df["t0"] = pd.to_datetime(
    df["t0"],
    errors="coerce"
)

df["기준분기"] = (
    df["기준분기"]
    .astype(str)
    .str.strip()
)

df["_quarter_period"] = pd.PeriodIndex(
    df["기준분기"],
    freq="Q"
)


# ============================================================
# 5. Feature 목록
# ============================================================

FEATURE_COLS = [

    col for col in df.columns

    if col not in DROP_COLS

    and col != "_quarter_period"
]


print("=" * 70)
print("전체 Feature")
print("=" * 70)

print("Feature 수:", len(FEATURE_COLS))


# ============================================================
# 6. Numeric / Categorical 분리
# ============================================================

CATEGORICAL_COLS = []
NUMERIC_COLS = []


for col in FEATURE_COLS:

    if (
        pd.api.types.is_object_dtype(df[col])
        or pd.api.types.is_bool_dtype(df[col])
        or isinstance(
            df[col].dtype,
            pd.CategoricalDtype
        )
    ):

        CATEGORICAL_COLS.append(col)

    else:

        NUMERIC_COLS.append(col)


print(
    "Numeric:",
    len(NUMERIC_COLS)
)

print(
    "Categorical:",
    len(CATEGORICAL_COLS)
)

print(
    "\nCategorical:",
    CATEGORICAL_COLS
)


# ============================================================
# 7. Preprocessor 생성 함수
# ============================================================

def make_preprocessor(feature_cols):

    numeric_cols = [

        col for col in feature_cols

        if col in NUMERIC_COLS
    ]

    categorical_cols = [

        col for col in feature_cols

        if col in CATEGORICAL_COLS
    ]


    numeric_transformer = Pipeline(

        steps=[

            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                )
            )

        ]
    )


    categorical_transformer = Pipeline(

        steps=[

            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                )
            ),

            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False
                )
            )

        ]
    )


    transformers = []


    if len(numeric_cols) > 0:

        transformers.append(

            (
                "num",
                numeric_transformer,
                numeric_cols
            )
        )


    if len(categorical_cols) > 0:

        transformers.append(

            (
                "cat",
                categorical_transformer,
                categorical_cols
            )
        )


    return ColumnTransformer(

        transformers=transformers,

        remainder="drop",

        verbose_feature_names_out=False
    )


# ============================================================
# 8. Tree Models
# ============================================================

def get_models():

    return {

        # ----------------------------------------------------
        # Random Forest
        # ----------------------------------------------------

        "RandomForest":

            RandomForestClassifier(

                n_estimators=400,

                max_depth=None,

                min_samples_leaf=2,

                n_jobs=-1,

                random_state=RANDOM_STATE
            ),


        # ----------------------------------------------------
        # XGBoost
        # ----------------------------------------------------

        "XGBoost":

            XGBClassifier(

                n_estimators=400,

                max_depth=6,

                learning_rate=0.05,

                subsample=0.8,

                colsample_bytree=0.8,

                eval_metric="logloss",

                n_jobs=-1,

                random_state=RANDOM_STATE
            ),


        # ----------------------------------------------------
        # LightGBM
        # ----------------------------------------------------

        "LightGBM":

            LGBMClassifier(

                n_estimators=400,

                learning_rate=0.05,

                num_leaves=31,

                subsample=0.8,

                colsample_bytree=0.8,

                n_jobs=-1,

                random_state=RANDOM_STATE,

                verbose=-1
            )
    }


# ============================================================
# 9. Validation Quarter
# ============================================================

VALIDATION_QUARTERS = list(

    pd.period_range(

        start=FIRST_VALID_QUARTER,

        end=LAST_VALID_QUARTER,

        freq="Q"
    )
)


print(
    "\nValidation:",
    VALIDATION_QUARTERS
)


# ============================================================
# 10. 불균형 기법
# ============================================================

IMBALANCE_METHODS = [

    "None",

    "ClassWeight",

    "RandomUnder",

    "RandomOver",

    "SMOTE"
]


# ============================================================
# 11. SMOTE 안전 처리
# ============================================================

def get_safe_sampler(
    method,
    y_train
):

    if method == "None":

        return None


    if method == "RandomUnder":

        return RandomUnderSampler(
            random_state=RANDOM_STATE
        )


    if method == "RandomOver":

        return RandomOverSampler(
            random_state=RANDOM_STATE
        )


    if method == "SMOTE":

        counts = (
            pd.Series(y_train)
            .value_counts()
        )

        if len(counts) < 2:

            return None


        minority_n = counts.min()


        # 샘플이 너무 적으면 SMOTE 대신 OverSampling
        if minority_n <= 5:

            return RandomOverSampler(
                random_state=RANDOM_STATE
            )


        k = min(
            5,
            minority_n - 1
        )


        return SMOTE(

            random_state=RANDOM_STATE,

            k_neighbors=k
        )


    return None


# ============================================================
# 12. Class Weight 적용
# ============================================================

def apply_class_weight(
    model_name,
    model,
    y_train
):

    model = clone(model)


    # --------------------------------------------------------
    # RandomForest
    # --------------------------------------------------------

    if model_name == "RandomForest":

        model.set_params(
            class_weight="balanced"
        )


    # --------------------------------------------------------
    # LightGBM
    # --------------------------------------------------------

    elif model_name == "LightGBM":

        model.set_params(
            class_weight="balanced"
        )


    # --------------------------------------------------------
    # XGBoost
    #
    # scale_pos_weight는 positive class인 Y=1에 대한 가중치.
    # Y=1이 minority일 때만 적용.
    # --------------------------------------------------------

    elif model_name == "XGBoost":

        counts = (
            pd.Series(y_train)
            .value_counts()
        )

        n0 = counts.get(0, 0)
        n1 = counts.get(1, 0)


        if n1 > 0 and n1 < n0:

            model.set_params(

                scale_pos_weight=(
                    n0 / n1
                )
            )


    return model


# ============================================================
# 13. Pipeline
# ============================================================

def build_pipeline(
    model_name,
    base_model,
    imbalance_method,
    y_train,
    feature_cols
):

    model = clone(base_model)

    preprocessor = make_preprocessor(
        feature_cols
    )


    # --------------------------------------------------------
    # Class Weight
    # --------------------------------------------------------

    if imbalance_method == "ClassWeight":

        model = apply_class_weight(

            model_name,
            model,
            y_train
        )


        return Pipeline(

            steps=[

                (
                    "preprocessor",
                    preprocessor
                ),

                (
                    "model",
                    model
                )
            ]
        )


    # --------------------------------------------------------
    # Sampling
    # --------------------------------------------------------

    sampler = get_safe_sampler(

        imbalance_method,

        y_train
    )


    if sampler is None:

        return Pipeline(

            steps=[

                (
                    "preprocessor",
                    preprocessor
                ),

                (
                    "model",
                    model
                )
            ]
        )


    return ImbPipeline(

        steps=[

            (
                "preprocessor",
                preprocessor
            ),

            (
                "sampler",
                sampler
            ),

            (
                "model",
                model
            )
        ]
    )


# ============================================================
# 14. 평가 함수
# ============================================================

def evaluate_model(
    y_true,
    y_prob,
    threshold=0.5
):

    y_true = np.asarray(
        y_true
    )

    y_prob = np.asarray(
        y_prob
    )

    y_pred = (
        y_prob >= threshold
    ).astype(int)


    result = {}


    # --------------------------------------------------------
    # AUC
    # --------------------------------------------------------

    if len(
        np.unique(y_true)
    ) == 2:

        result["ROC_AUC"] = (
            roc_auc_score(
                y_true,
                y_prob
            )
        )

        result["PR_AUC"] = (
            average_precision_score(
                y_true,
                y_prob
            )
        )

    else:

        result["ROC_AUC"] = np.nan
        result["PR_AUC"] = np.nan


    # --------------------------------------------------------
    # Classification Metrics
    # --------------------------------------------------------

    result["Accuracy"] = (
        accuracy_score(
            y_true,
            y_pred
        )
    )


    result["Balanced_Accuracy"] = (
        balanced_accuracy_score(
            y_true,
            y_pred
        )
    )


    result["Precision"] = (
        precision_score(
            y_true,
            y_pred,
            zero_division=0
        )
    )


    result["Recall"] = (
        recall_score(
            y_true,
            y_pred,
            zero_division=0
        )
    )


    result["F1"] = (
        f1_score(
            y_true,
            y_pred,
            zero_division=0
        )
    )


    result["Brier"] = (
        brier_score_loss(
            y_true,
            y_prob
        )
    )


    # --------------------------------------------------------
    # Confusion Matrix
    # --------------------------------------------------------

    tn, fp, fn, tp = (

        confusion_matrix(

            y_true,
            y_pred,

            labels=[0, 1]

        ).ravel()
    )


    result["TN"] = tn
    result["FP"] = fp
    result["FN"] = fn
    result["TP"] = tp


    result["Specificity"] = (

        tn / (tn + fp)

        if (tn + fp) > 0

        else np.nan
    )


    result["N"] = len(
        y_true
    )


    result["Survival_Rate"] = (
        np.mean(y_true)
    )


    return result


# ============================================================
# 15. 원래 변수 단위 Feature Importance
# ============================================================
#
# One-hot encoding 후
#
# 업종_음식점
# 업종_미용업
# 업종_제과점
#
# 등을
#
# 업종
#
# 하나의 중요도로 다시 합산
# ============================================================

def get_original_feature_importance(
    fitted_pipeline,
    feature_cols
):

    preprocessor = (
        fitted_pipeline
        .named_steps[
            "preprocessor"
        ]
    )

    model = (
        fitted_pipeline
        .named_steps[
            "model"
        ]
    )


    transformed_names = (
        preprocessor
        .get_feature_names_out()
    )


    importances = (
        model.feature_importances_
    )


    temp = pd.DataFrame({

        "Transformed_Feature":
            transformed_names,

        "Importance":
            importances
    })


    # --------------------------------------------------------
    # 변환된 feature가 원래 어떤 변수인지 매핑
    # --------------------------------------------------------

    def map_original_feature(
        transformed_name
    ):

        # 정확히 일치하면 numeric
        if transformed_name in feature_cols:

            return transformed_name


        # categorical
        #
        # 예:
        # 업종_일반음식점
        #
        # 원래 feature = 업종
        # ----------------------------------------------------

        candidates = [

            col for col in feature_cols

            if transformed_name.startswith(
                col + "_"
            )
        ]


        if len(candidates) > 0:

            # 변수명이 긴 것을 우선
            return max(
                candidates,
                key=len
            )


        return transformed_name


    temp[
        "Feature"
    ] = (

        temp[
            "Transformed_Feature"
        ]
        .apply(
            map_original_feature
        )
    )


    # --------------------------------------------------------
    # One-hot 중요도 합산
    # --------------------------------------------------------

    result = (

        temp
        .groupby(
            "Feature",
            as_index=False
        )["Importance"]
        .sum()
    )


    total = (
        result[
            "Importance"
        ]
        .sum()
    )


    if total > 0:

        result[
            "Importance_Normalized"
        ] = (

            result[
                "Importance"
            ]
            / total
        )

    else:

        result[
            "Importance_Normalized"
        ] = 0


    result = (

        result
        .sort_values(
            "Importance_Normalized",
            ascending=False
        )
        .reset_index(drop=True)
    )


    result[
        "Rank"
    ] = (

        np.arange(
            1,
            len(result) + 1
        )
    )


    return result


# ============================================================
# 16. Expanding Window
# ============================================================
def run_validation(
    data,
    horizon_name,
    target_col,
    status_col
):

    print()
    print("=" * 80)
    print("START:", horizon_name)
    print("=" * 80)

    # --------------------------------------------------------
    # 관측 가능한 라벨만
    # --------------------------------------------------------

    d = data[
        (
            data[status_col]
            .astype(str)
            .str.lower()
            == "observed"
        )
        &
        data[target_col].notna()
    ].copy()

    d[target_col] = (
        d[target_col]
        .astype(int)
    )

    models = get_models()

    metric_results = []
    importance_results = []

    # --------------------------------------------------------
    # 전체 작업 수
    # --------------------------------------------------------

    total_jobs = (
        len(models)
        * len(IMBALANCE_METHODS)
        * len(VALIDATION_QUARTERS)
    )

    start_time = time.time()

    # --------------------------------------------------------
    # tqdm
    # --------------------------------------------------------

    pbar = tqdm(
        total=total_jobs,
        desc=f"{horizon_name} Tree Models",
        unit="fit",
        dynamic_ncols=True
    )

    # ========================================================
    # MODEL
    # ========================================================

    for model_name, base_model in models.items():

        # ====================================================
        # IMBALANCE
        # ====================================================

        for imbalance_method in IMBALANCE_METHODS:

            # =================================================
            # FOLD
            # =================================================

            for fold_no, valid_q in enumerate(
                VALIDATION_QUARTERS,
                start=1
            ):

                # ---------------------------------------------
                # tqdm 현재 상태 표시
                # ---------------------------------------------

                pbar.set_postfix({
                    "model": model_name,
                    "imb": imbalance_method,
                    "valid": str(valid_q)
                })

                train_df = d[
                    d["_quarter_period"] < valid_q
                ].copy()

                valid_df = d[
                    d["_quarter_period"] == valid_q
                ].copy()

                # ---------------------------------------------
                # 데이터 없으면 skip
                # ---------------------------------------------

                if (
                    len(train_df) == 0
                    or len(valid_df) == 0
                ):

                    pbar.update(1)
                    continue

                X_train = train_df[
                    FEATURE_COLS
                ]

                y_train = train_df[
                    target_col
                ]

                X_valid = valid_df[
                    FEATURE_COLS
                ]

                y_valid = valid_df[
                    target_col
                ]

                # ---------------------------------------------
                # Train 한 클래스만 있으면 skip
                # ---------------------------------------------

                if y_train.nunique() < 2:

                    pbar.update(1)
                    continue

                try:

                    pipeline = build_pipeline(
                        model_name=model_name,
                        base_model=base_model,
                        imbalance_method=imbalance_method,
                        y_train=y_train,
                        feature_cols=FEATURE_COLS
                    )

                    pipeline.fit(
                        X_train,
                        y_train
                    )

                    y_prob = (
                        pipeline
                        .predict_proba(
                            X_valid
                        )[:, 1]
                    )

                    # =========================================
                    # METRICS
                    # =========================================

                    metrics = evaluate_model(
                        y_valid,
                        y_prob
                    )

                    row = {

                        "Horizon": horizon_name,

                        "Model": model_name,

                        "Imbalance": imbalance_method,

                        "Fold": fold_no,

                        "Train_End": str(
                            valid_q - 1
                        ),

                        "Valid_Quarter": str(
                            valid_q
                        ),

                        "Train_N": len(
                            train_df
                        ),

                        "Valid_N": len(
                            valid_df
                        ),

                        "Train_Y0": int(
                            (y_train == 0).sum()
                        ),

                        "Train_Y1": int(
                            (y_train == 1).sum()
                        ),

                        "Valid_Y0": int(
                            (y_valid == 0).sum()
                        ),

                        "Valid_Y1": int(
                            (y_valid == 1).sum()
                        )
                    }

                    row.update(
                        metrics
                    )

                    metric_results.append(
                        row
                    )

                    # =========================================
                    # FEATURE IMPORTANCE
                    # =========================================

                    fi = (
                        get_original_feature_importance(
                            pipeline,
                            FEATURE_COLS
                        )
                    )

                    fi["Horizon"] = horizon_name
                    fi["Model"] = model_name
                    fi["Imbalance"] = imbalance_method
                    fi["Fold"] = fold_no
                    fi["Valid_Quarter"] = str(valid_q)

                    importance_results.append(
                        fi
                    )

                except Exception as e:

                    tqdm.write(
                        f"ERROR | "
                        f"{horizon_name} | "
                        f"{model_name} | "
                        f"{imbalance_method} | "
                        f"{valid_q} | "
                        f"{e}"
                    )

                finally:

                    pbar.update(1)

    pbar.close()

    # --------------------------------------------------------
    # 총 실행 시간
    # --------------------------------------------------------

    elapsed = (
        time.time()
        - start_time
    )

    print()
    print(
        f"{horizon_name} 완료"
    )

    print(
        f"소요시간: "
        f"{elapsed / 60:.1f}분 "
        f"({elapsed / 3600:.2f}시간)"
    )

    # --------------------------------------------------------
    # DataFrame
    # --------------------------------------------------------

    metric_df = pd.DataFrame(
        metric_results
    )

    if len(importance_results) > 0:

        importance_df = pd.concat(
            importance_results,
            ignore_index=True
        )

    else:

        importance_df = pd.DataFrame()

    return (
        metric_df,
        importance_df
    )
# ============================================================
# 17. 12M + 18M 실행
# ============================================================

ALL_METRICS = []

ALL_IMPORTANCE = []


for horizon_name, config in TARGET_CONFIG.items():

    metric_df, importance_df = (

        run_validation(

            data=df,

            horizon_name=horizon_name,

            target_col=config["target"],

            status_col=config["status"]
        )
    )


    ALL_METRICS.append(
        metric_df
    )


    ALL_IMPORTANCE.append(
        importance_df
    )


# ============================================================
# 18. 통합
# ============================================================

validation_metrics = pd.concat(

    ALL_METRICS,

    ignore_index=True
)


fold_importances = pd.concat(

    ALL_IMPORTANCE,

    ignore_index=True
)


# ============================================================
# 19. Fold별 결과 저장
# ============================================================

validation_metrics.to_csv(

    RESULT_DIR /
    "01_validation_fold_results.csv",

    index=False,

    encoding="utf-8-sig"
)


fold_importances.to_csv(

    RESULT_DIR /
    "02_fold_feature_importance.csv",

    index=False,

    encoding="utf-8-sig"
)


# ============================================================
# 20. 모델 × 불균형 기법 평균 성능
# ============================================================

METRICS = [

    "ROC_AUC",

    "PR_AUC",

    "Accuracy",

    "Balanced_Accuracy",

    "Precision",

    "Recall",

    "Specificity",

    "F1",

    "Brier"
]


performance_mean = (

    validation_metrics

    .groupby(
        [
            "Horizon",
            "Model",
            "Imbalance"
        ]
    )[METRICS]

    .mean()

    .reset_index()
)


# ============================================================
# 21. 표준편차
# ============================================================

performance_std = (

    validation_metrics

    .groupby(
        [
            "Horizon",
            "Model",
            "Imbalance"
        ]
    )[METRICS]

    .std()

    .reset_index()
)


performance_std = (

    performance_std.rename(

        columns={

            col:
                f"{col}_STD"

            for col in METRICS
        }
    )
)


# ============================================================
# 22. Fold 수
# ============================================================

fold_count = (

    validation_metrics

    .groupby(
        [
            "Horizon",
            "Model",
            "Imbalance"
        ]
    )

    .size()

    .reset_index(
        name="N_Folds"
    )
)


# ============================================================
# 23. Performance Summary
# ============================================================

performance_summary = (

    performance_mean

    .merge(

        performance_std,

        on=[
            "Horizon",
            "Model",
            "Imbalance"
        ],

        how="left"
    )

    .merge(

        fold_count,

        on=[
            "Horizon",
            "Model",
            "Imbalance"
        ],

        how="left"
    )
)


performance_summary = (

    performance_summary

    .sort_values(

        [
            "Horizon",
            "PR_AUC",
            "ROC_AUC"
        ],

        ascending=[
            True,
            False,
            False
        ]
    )

    .reset_index(drop=True)
)


performance_summary.to_csv(

    RESULT_DIR /
    "03_model_imbalance_performance.csv",

    index=False,

    encoding="utf-8-sig"
)


# ============================================================
# 24. Horizon별 최적 모델 + 불균형 기법
# ============================================================
#
# 우선 PR-AUC
# 다음 ROC-AUC
#
# ============================================================

best_configs = {}


for horizon in TARGET_CONFIG.keys():

    temp = (

        performance_summary[

            performance_summary[
                "Horizon"
            ] == horizon

        ]

        .sort_values(

            [
                "PR_AUC",
                "ROC_AUC"
            ],

            ascending=False
        )
    )


    if len(temp) == 0:

        continue


    best = temp.iloc[0]


    best_configs[
        horizon
    ] = {

        "Model":
            best["Model"],

        "Imbalance":
            best["Imbalance"],

        "PR_AUC":
            best["PR_AUC"],

        "ROC_AUC":
            best["ROC_AUC"],

        "Balanced_Accuracy":
            best["Balanced_Accuracy"],

        "F1":
            best["F1"]
    }


best_config_df = (

    pd.DataFrame(
        best_configs
    )

    .T

    .reset_index()

    .rename(
        columns={
            "index":
                "Horizon"
        }
    )
)


best_config_df.to_csv(

    RESULT_DIR /
    "04_best_model_config.csv",

    index=False,

    encoding="utf-8-sig"
)


print()
print("=" * 80)
print("BEST CONFIG")
print("=" * 80)

print(
    best_config_df
)


# ============================================================
# 25. 최적 모델의 Fold Feature Importance만 추출
# ============================================================

best_importance_list = []


for horizon, config in best_configs.items():

    temp = fold_importances[

        (
            fold_importances[
                "Horizon"
            ] == horizon
        )

        &

        (
            fold_importances[
                "Model"
            ] == config["Model"]
        )

        &

        (
            fold_importances[
                "Imbalance"
            ] == config["Imbalance"]
        )

    ].copy()


    best_importance_list.append(
        temp
    )


best_fold_importance = pd.concat(

    best_importance_list,

    ignore_index=True
)


# ============================================================
# 26. Feature Importance 안정성 집계
# ============================================================
#
# Mean Importance
# Std
# Median
# 평균 Rank
# Top10 등장 비율
# Top20 등장 비율
#
# ============================================================

feature_stability = (

    best_fold_importance

    .groupby(
        [
            "Horizon",
            "Feature"
        ]
    )

    .agg(

        Mean_Importance=(
            "Importance_Normalized",
            "mean"
        ),

        STD_Importance=(
            "Importance_Normalized",
            "std"
        ),

        Median_Importance=(
            "Importance_Normalized",
            "median"
        ),

        Mean_Rank=(
            "Rank",
            "mean"
        ),

        N_Folds=(
            "Fold",
            "nunique"
        )

    )

    .reset_index()
)


# ============================================================
# 27. TOP10 / TOP20 등장 비율
# ============================================================

top10_rate = (

    best_fold_importance

    .assign(

        Is_TOP10=lambda x:
            (
                x["Rank"] <= 10
            ).astype(int)
    )

    .groupby(
        [
            "Horizon",
            "Feature"
        ]
    )["Is_TOP10"]

    .mean()

    .reset_index(
        name="TOP10_Rate"
    )
)


top20_rate = (

    best_fold_importance

    .assign(

        Is_TOP20=lambda x:
            (
                x["Rank"] <= 20
            ).astype(int)
    )

    .groupby(
        [
            "Horizon",
            "Feature"
        ]
    )["Is_TOP20"]

    .mean()

    .reset_index(
        name="TOP20_Rate"
    )
)


feature_stability = (

    feature_stability

    .merge(

        top10_rate,

        on=[
            "Horizon",
            "Feature"
        ],

        how="left"
    )

    .merge(

        top20_rate,

        on=[
            "Horizon",
            "Feature"
        ],

        how="left"
    )
)


# ============================================================
# 28. Horizon 내부 중요도 정규화
# ============================================================

feature_stability[

    "Normalized_Mean_Importance"

] = (

    feature_stability

    .groupby(
        "Horizon"
    )["Mean_Importance"]

    .transform(

        lambda x:

            x / x.sum()

            if x.sum() > 0

            else x
    )
)


# ============================================================
# 29. 누적 중요도
# ============================================================

feature_stability = (

    feature_stability

    .sort_values(

        [
            "Horizon",
            "Normalized_Mean_Importance"
        ],

        ascending=[
            True,
            False
        ]
    )
)


feature_stability[

    "Cumulative_Importance"

] = (

    feature_stability

    .groupby(
        "Horizon"
    )[

        "Normalized_Mean_Importance"

    ]

    .cumsum()
)


# ============================================================
# 30. Feature Selection
# ============================================================

selected_feature_rows = []


for horizon in TARGET_CONFIG.keys():

    temp = (

        feature_stability[

            feature_stability[
                "Horizon"
            ] == horizon

        ]

        .sort_values(

            "Normalized_Mean_Importance",

            ascending=False
        )

        .reset_index(drop=True)
    )


    if len(temp) == 0:

        continue


    # --------------------------------------------------------
    # 90% 누적 중요도까지
    # --------------------------------------------------------

    n_cumulative = (

        temp[
            "Cumulative_Importance"
        ]

        .searchsorted(
            IMPORTANCE_CUM_THRESHOLD
        )

        + 1
    )


    # --------------------------------------------------------
    # 최소 / 최대 Feature 수
    # --------------------------------------------------------

    n_selected = max(

        MIN_FEATURES,

        n_cumulative
    )


    n_selected = min(

        MAX_FEATURES,

        n_selected,

        len(temp)
    )


    temp[
        "Selected"
    ] = 0


    temp.loc[

        : n_selected - 1,

        "Selected"

    ] = 1


    temp[
        "Selection_Count"
    ] = n_selected


    selected_feature_rows.append(
        temp
    )


feature_selection_summary = pd.concat(

    selected_feature_rows,

    ignore_index=True
)


feature_selection_summary.to_csv(

    RESULT_DIR /
    "05_feature_selection_summary.csv",

    index=False,

    encoding="utf-8-sig"
)


# ============================================================
# 31. 선택 Feature 목록
# ============================================================

selected_features = {}


for horizon in TARGET_CONFIG.keys():

    selected = (

        feature_selection_summary[

            (
                feature_selection_summary[
                    "Horizon"
                ] == horizon
            )

            &

            (
                feature_selection_summary[
                    "Selected"
                ] == 1
            )

        ]["Feature"]

        .tolist()
    )


    selected_features[
        horizon
    ] = selected


    print()
    print("=" * 80)
    print(
        horizon,
        "SELECTED FEATURES"
    )
    print("=" * 80)

    print(
        "선택 변수 수:",
        len(selected)
    )

    for i, feature in enumerate(
        selected,
        start=1
    ):

        print(
            f"{i:02d}. {feature}"
        )


# ============================================================
# 32. 선택 Feature 목록 CSV
# ============================================================

selected_feature_table = []


for horizon, features in selected_features.items():

    for rank, feature in enumerate(
        features,
        start=1
    ):

        selected_feature_table.append({

            "Horizon":
                horizon,

            "Selection_Rank":
                rank,

            "Feature":
                feature
        })


selected_feature_table = pd.DataFrame(
    selected_feature_table
)


selected_feature_table.to_csv(

    RESULT_DIR /
    "06_selected_features.csv",

    index=False,

    encoding="utf-8-sig"
)


# ============================================================
# 33. 선택 변수로 FINAL TEST
# ============================================================

final_results = []

final_importance_results = []

final_predictions = []


for horizon, config in TARGET_CONFIG.items():

    if horizon not in best_configs:

        continue


    target_col = config[
        "target"
    ]

    status_col = config[
        "status"
    ]


    selected = selected_features[
        horizon
    ]


    model_name = (
        best_configs[
            horizon
        ]["Model"]
    )


    imbalance_method = (
        best_configs[
            horizon
        ]["Imbalance"]
    )


    # --------------------------------------------------------
    # observed만
    # --------------------------------------------------------

    d = df[

        (
            df[status_col]
            .astype(str)
            .str.lower()
            == "observed"
        )

        &

        df[target_col].notna()

    ].copy()


    d[target_col] = (
        d[target_col]
        .astype(int)
    )


    test_q = pd.Period(
        FINAL_TEST_QUARTER,
        freq="Q"
    )


    train_df = d[

        d["_quarter_period"]
        < test_q

    ].copy()


    test_df = d[

        d["_quarter_period"]
        == test_q

    ].copy()


    print()
    print("=" * 80)

    print(
        horizon,
        "FINAL TEST"
    )

    print(
        "Model:",
        model_name
    )

    print(
        "Imbalance:",
        imbalance_method
    )

    print(
        "Selected Features:",
        len(selected)
    )

    print(
        "Train:",
        len(train_df)
    )

    print(
        "Test:",
        len(test_df)
    )

    print("=" * 80)


    # --------------------------------------------------------
    # Test 라벨 없음
    # --------------------------------------------------------

    if len(test_df) == 0:

        print(
            horizon,
            "2024Q2 observed label 없음"
        )

        continue


    X_train = train_df[
        selected
    ]

    y_train = train_df[
        target_col
    ]


    X_test = test_df[
        selected
    ]

    y_test = test_df[
        target_col
    ]


    models = get_models()

    base_model = models[
        model_name
    ]


    pipeline = build_pipeline(

        model_name=model_name,

        base_model=base_model,

        imbalance_method=imbalance_method,

        y_train=y_train,

        feature_cols=selected
    )


    # --------------------------------------------------------
    # FIT
    # --------------------------------------------------------

    pipeline.fit(

        X_train,

        y_train
    )


    # --------------------------------------------------------
    # SAVE MODEL
    # --------------------------------------------------------

    joblib.dump(

        pipeline,

        RESULT_DIR /
        f"{horizon}_FINAL_TREE_MODEL.joblib"
    )


    # --------------------------------------------------------
    # PREDICT
    # --------------------------------------------------------

    y_prob = (

        pipeline

        .predict_proba(
            X_test
        )[:, 1]
    )


    metrics = evaluate_model(

        y_test,

        y_prob
    )


    metrics.update({

        "Horizon":
            horizon,

        "Model":
            model_name,

        "Imbalance":
            imbalance_method,

        "N_Selected_Features":
            len(selected),

        "Train_N":
            len(train_df),

        "Test_N":
            len(test_df),

        "Test_Quarter":
            FINAL_TEST_QUARTER
    })


    final_results.append(
        metrics
    )


    # ========================================================
    # Final Feature Importance
    # ========================================================

    fi = get_original_feature_importance(

        pipeline,

        selected
    )


    fi[
        "Horizon"
    ] = horizon


    fi[
        "Model"
    ] = model_name


    fi[
        "Imbalance"
    ] = imbalance_method


    final_importance_results.append(
        fi
    )


    # ========================================================
    # Prediction
    # ========================================================

    pred_df = pd.DataFrame({

        "관리번호":
            test_df[
                "관리번호"
            ].values,

        "기준분기":
            test_df[
                "기준분기"
            ].values,

        "Horizon":
            horizon,

        "Y_true":
            y_test.values,

        "Y_prob":
            y_prob,

        "Y_pred":
            (
                y_prob >= 0.5
            ).astype(int)
    })


    final_predictions.append(
        pred_df
    )


# ============================================================
# 34. FINAL 결과 통합
# ============================================================

final_performance = pd.DataFrame(
    final_results
)


if len(
    final_importance_results
) > 0:

    final_feature_importance = pd.concat(

        final_importance_results,

        ignore_index=True
    )

else:

    final_feature_importance = (
        pd.DataFrame()
    )


if len(
    final_predictions
) > 0:

    final_prediction_df = pd.concat(

        final_predictions,

        ignore_index=True
    )

else:

    final_prediction_df = (
        pd.DataFrame()
    )


# ============================================================
# 35. CSV 저장
# ============================================================

final_performance.to_csv(

    RESULT_DIR /
    "07_final_test_performance.csv",

    index=False,

    encoding="utf-8-sig"
)


final_feature_importance.to_csv(

    RESULT_DIR /
    "08_final_feature_importance.csv",

    index=False,

    encoding="utf-8-sig"
)


final_prediction_df.to_csv(

    RESULT_DIR /
    "09_final_predictions.csv",

    index=False,

    encoding="utf-8-sig"
)


# ============================================================
# 36. Excel 통합 저장
# ============================================================

EXCEL_PATH = (

    RESULT_DIR /
    "TREE_MODEL_ANALYSIS.xlsx"
)


with pd.ExcelWriter(

    EXCEL_PATH,

    engine="openpyxl"

) as writer:


    # --------------------------------------------------------
    # 가장 중요
    # --------------------------------------------------------

    performance_summary.to_excel(

        writer,

        sheet_name="01_Performance",

        index=False
    )


    # --------------------------------------------------------

    best_config_df.to_excel(

        writer,

        sheet_name="02_Best_Config",

        index=False
    )


    # --------------------------------------------------------

    feature_selection_summary.to_excel(

        writer,

        sheet_name="03_Feature_Selection",

        index=False
    )


    # --------------------------------------------------------

    selected_feature_table.to_excel(

        writer,

        sheet_name="04_Selected_Features",

        index=False
    )


    # --------------------------------------------------------

    final_performance.to_excel(

        writer,

        sheet_name="05_Final_Test",

        index=False
    )


    # --------------------------------------------------------

    final_feature_importance.to_excel(

        writer,

        sheet_name="06_Final_Importance",

        index=False
    )


    # --------------------------------------------------------

    validation_metrics.to_excel(

        writer,

        sheet_name="07_Fold_Detail",

        index=False
    )


    # --------------------------------------------------------

    best_fold_importance.to_excel(

        writer,

        sheet_name="08_Fold_Importance",

        index=False
    )


    # --------------------------------------------------------

    final_prediction_df.to_excel(

        writer,

        sheet_name="09_Predictions",

        index=False
    )


# ============================================================
# 37. Excel Formatting
# ============================================================

from openpyxl import load_workbook


wb = load_workbook(
    EXCEL_PATH
)


for ws in wb.worksheets:

    ws.freeze_panes = "A2"

    ws.auto_filter.ref = (
        ws.dimensions
    )


    for column_cells in ws.columns:

        column_letter = (
            column_cells[0]
            .column_letter
        )

        max_length = 0


        for cell in column_cells:

            try:

                max_length = max(

                    max_length,

                    len(
                        str(cell.value)
                    )
                )

            except:

                pass


        ws.column_dimensions[
            column_letter
        ].width = min(

            max_length + 2,

            35
        )


wb.save(
    EXCEL_PATH
)


# ============================================================
# 38. 최종 출력
# ============================================================

print()
print("=" * 80)
print("TREE MODEL 분석 완료")
print("=" * 80)

print(
    "\n저장 위치:"
)

print(
    RESULT_DIR.resolve()
)

print(
    "\n통합 Excel:"
)

print(
    EXCEL_PATH.resolve()
)


# ============================================================
# 39. 화면에서 바로 확인할 핵심 표
# ============================================================

print()
print("=" * 80)
print("1. 모델 + 불균형 기법 성능")
print("=" * 80)

print(

    performance_summary[
        [
            "Horizon",
            "Model",
            "Imbalance",
            "PR_AUC",
            "ROC_AUC",
            "Balanced_Accuracy",
            "Precision",
            "Recall",
            "Specificity",
            "F1",
            "Brier",
            "N_Folds"
        ]
    ]
)


print()
print("=" * 80)
print("2. 선택된 모델")
print("=" * 80)

print(
    best_config_df
)


print()
print("=" * 80)
print("3. 선택된 변수")
print("=" * 80)

print(
    selected_feature_table
)


print()
print("=" * 80)
print("4. 변수 선택 근거")
print("=" * 80)

print(

    feature_selection_summary[

        feature_selection_summary[
            "Selected"
        ] == 1

    ][
        [
            "Horizon",
            "Feature",
            "Mean_Importance",
            "STD_Importance",
            "Mean_Rank",
            "TOP10_Rate",
            "TOP20_Rate",
            "Cumulative_Importance",
            "Selected"
        ]
    ]
)


print()
print("=" * 80)
print("5. FINAL TEST")
print("=" * 80)

print(
    final_performance
)


print()
print("=" * 80)
print("6. FINAL FEATURE IMPORTANCE")
print("=" * 80)

print(
    final_feature_importance
)
# ============================================================
# 40. GPT API 기반 최종 분석 보고서 생성
# ============================================================

import os
from openai import OpenAI
from dotenv import load_dotenv


# ------------------------------------------------------------
# API KEY
# ------------------------------------------------------------

load_dotenv()

client = OpenAI(
    api_key=os.environ["OPENAI_API_KEY"]
)


# ============================================================
# 41. GPT에 전달할 결과 정리
# ============================================================

# 너무 많은 데이터를 전달하지 않고
# 실제 해석에 필요한 결과만 전달

performance_for_gpt = (
    performance_summary[
        [
            "Horizon",
            "Model",
            "Imbalance",
            "PR_AUC",
            "ROC_AUC",
            "Balanced_Accuracy",
            "Precision",
            "Recall",
            "Specificity",
            "F1",
            "Brier",
            "N_Folds"
        ]
    ]
    .round(4)
)


best_config_for_gpt = (
    best_config_df
    .round(4)
)


# ------------------------------------------------------------
# 선택된 Feature
# ------------------------------------------------------------

selected_for_gpt = (
    feature_selection_summary[
        feature_selection_summary[
            "Selected"
        ] == 1
    ][
        [
            "Horizon",
            "Feature",
            "Mean_Importance",
            "STD_Importance",
            "Mean_Rank",
            "TOP10_Rate",
            "TOP20_Rate",
            "Cumulative_Importance"
        ]
    ]
    .round(4)
)


# ------------------------------------------------------------
# Horizon별 TOP 20 변수
# ------------------------------------------------------------

top_features_for_gpt = (

    feature_selection_summary

    .sort_values(
        [
            "Horizon",
            "Mean_Importance"
        ],
        ascending=[
            True,
            False
        ]
    )

    .groupby(
        "Horizon"
    )

    .head(20)

    [
        [
            "Horizon",
            "Feature",
            "Mean_Importance",
            "STD_Importance",
            "Mean_Rank",
            "TOP10_Rate",
            "TOP20_Rate",
            "Selected"
        ]
    ]

    .round(4)
)


# ------------------------------------------------------------
# Final Test
# ------------------------------------------------------------

if len(final_performance) > 0:

    final_test_for_gpt = (
        final_performance
        .round(4)
    )

else:

    final_test_for_gpt = pd.DataFrame()


# ============================================================
# 42. Markdown 문자열로 변환
# ============================================================

performance_md = (
    performance_for_gpt
    .to_markdown(index=False)
)

best_config_md = (
    best_config_for_gpt
    .to_markdown(index=False)
)

selected_md = (
    selected_for_gpt
    .to_markdown(index=False)
)

top_features_md = (
    top_features_for_gpt
    .to_markdown(index=False)
)


if len(final_test_for_gpt) > 0:

    final_test_md = (
        final_test_for_gpt
        .to_markdown(index=False)
    )

else:

    final_test_md = (
        "Final Test 결과 없음 "
        "(해당 Horizon의 observed label 부족 가능성)"
    )


# ============================================================
# 43. Prompt
# ============================================================

prompt = f"""
당신은 머신러닝 기반 사업자 생존예측 연구의 분석가이다.

현재 분석 목적은 위기 사업자의 향후 12개월 및 18개월
영업 지속 여부를 예측하는 것이다.

종속변수 정의:
- Y=1: 해당 기간까지 영업 지속
- Y=0: 해당 기간 내 폐업

현재 단계에서는 Logistic Regression을 사용하지 않았다.

다음 3개의 Tree 기반 모델을 비교하였다.

- Random Forest
- XGBoost
- LightGBM

각 모델에 다음 불균형 처리 기법을 적용하였다.

- None
- ClassWeight
- RandomUnderSampling
- RandomOverSampling
- SMOTE

검증 방법은 Expanding Window Validation이다.

예:

2019Q1~2019Q4 -> 2020Q1 검증
2019Q1~2020Q1 -> 2020Q2 검증
2019Q1~2020Q2 -> 2020Q3 검증
...
2019Q1~2023Q4 -> 2024Q1 검증

2024Q2는 Final Test로 별도 사용한다.

주의:
Y=1은 생존이고 Y=0은 폐업이다.

따라서 Precision/Recall/F1은 기본적으로
생존(Y=1)을 positive class로 계산한 값이다.

분석의 주요 목적은 다음과 같다.

1. Tree 모델 중 어떤 모델이 상대적으로 안정적인지 분석
2. 어떤 불균형 처리 방법이 적절한지 분석
3. 12M과 18M 결과 차이 분석
4. 여러 Fold에서 안정적으로 중요한 변수를 찾기
5. 다음 단계 Logistic Regression에 사용할 변수 후보 결정

단순히 가장 높은 숫자 하나만 보고 결론내리지 말고,
평균 성능, 표준편차, PR-AUC, ROC-AUC,
Balanced Accuracy, Recall, F1을 종합적으로 검토하라.

특히 불균형 데이터이므로 Accuracy만으로
모델을 평가하지 말라.

Feature Importance는 단일 Fold가 아니라
Expanding Window의 여러 Fold에서 계산된 결과를
평균하여 얻은 것이다.

다음은 실제 분석 결과이다.


============================================================
[1] 모델 + 불균형 기법 성능
============================================================

{performance_md}


============================================================
[2] 현재 자동 선정된 최적 조합
============================================================

{best_config_md}


============================================================
[3] Feature Selection 대상 변수
============================================================

{selected_md}


============================================================
[4] 중요도 TOP 변수
============================================================

{top_features_md}


============================================================
[5] 2024Q2 Final Test
============================================================

{final_test_md}


위 결과만을 근거로 최종 분석 보고서를 작성하라.

보고서는 반드시 Markdown 형식으로 작성한다.


# 보고서 구성

# Tree Model 분석 결과

## 1. 분석 개요

분석 목적, 모델, 불균형 처리 방법,
Expanding Window Validation 방법을 간략하게 설명한다.


## 2. 12개월 생존 예측 결과

12M 모델들의 성능을 표로 정리한다.

그 다음 다음 내용을 분석한다.

- 모델별 차이
- 불균형 처리 방법별 차이
- PR-AUC
- ROC-AUC
- Balanced Accuracy
- Recall
- F1
- Fold 간 안정성

단순히 가장 높은 값 하나만으로 결론내리지 않는다.


## 3. 18개월 생존 예측 결과

12M과 동일한 방식으로 분석한다.


## 4. 불균형 처리 기법 비교

None / ClassWeight / RandomUnder /
RandomOver / SMOTE 결과를 비교한다.

어떤 방법이 현재 데이터에서 상대적으로
안정적인지 설명한다.

12M과 18M에서 결과가 다르면 그 차이도 설명한다.


## 5. Tree 모델 비교

Random Forest / XGBoost / LightGBM의
성능과 안정성을 비교한다.

각 Horizon에서 이후 분석에 사용할
모델과 불균형 처리 방법을 명확하게 제시한다.

선정 근거를 성능 지표를 이용하여 설명한다.


## 6. 변수 중요도 분석

12M과 18M을 각각 분석한다.

중요 변수 TOP 20을 표로 보여준다.

다음 값을 같이 고려한다.

- Mean Importance
- STD Importance
- Mean Rank
- TOP10 Rate
- TOP20 Rate

중요도가 높으면서 여러 Fold에서 반복적으로
상위권에 나타나는 변수를 안정적인 중요 변수로 해석한다.


## 7. Feature Selection 결과

다음 Logistic Regression 단계에서 사용할
변수 후보를 정리한다.

다음 세 그룹으로 나눈다.

### 핵심 유지 변수
여러 Fold에서 안정적으로 중요하고
반드시 유지할 가치가 높은 변수.

### 추가 검토 변수
일정 수준 중요하지만 Fold에 따라
변동성이 있거나 다른 변수와 중복 가능성이 있는 변수.

### 제외 검토 변수
중요도가 매우 낮거나 거의 선택되지 않은 변수.

단, 제공된 결과에서 확인할 수 없는 내용을
임의로 만들어내지 않는다.


## 8. 12M과 18M 공통 핵심 변수

두 Horizon 모두에서 중요하게 나타나는 변수를 찾는다.

표 형식:

| 변수 | 12M 중요도 | 18M 중요도 | 해석 |


## 9. 다음 Logistic Regression 분석 권고

다음 단계에서 사용할 데이터 구성을 제시한다.

예:

- 12M Logistic에 사용할 변수
- 18M Logistic에 사용할 변수
- 12M/18M 공통 변수
- L1/L2 Logistic에서 추가 확인할 변수

Tree Feature Importance와 Logistic 계수의 의미가
다르다는 점도 설명한다.


## 10. 최종 결론

다음 내용을 매우 명확하게 요약한다.

- 12M에서 사용할 Tree 모델
- 12M에서 사용할 불균형 처리 방법
- 18M에서 사용할 Tree 모델
- 18M에서 사용할 불균형 처리 방법
- 다음 단계에서 사용할 변수 Selection 방향
- Logistic Regression 단계에서 검증해야 할 내용


중요:

숫자를 임의로 생성하지 말 것.

제공된 결과에 없는 사실을 만들지 말 것.

모델 성능이 비슷하면 한 모델이 절대적으로
우수하다고 단정하지 말고 차이가 작다고 설명할 것.

2024Q2 Final Test는 모델 선택에 사용한 것이 아니라
마지막 외부 시점 평가라는 점을 구분할 것.
"""


# ============================================================
# 44. API 호출
# ============================================================

print()
print("=" * 80)
print("GPT 최종 보고서 생성 중...")
print("=" * 80)


response = client.responses.create(

    model="gpt-5.6",

    input=prompt
)


report = response.output_text


# ============================================================
# 45. Markdown 저장
# ============================================================

REPORT_PATH = (

    RESULT_DIR /
    "FINAL_TREE_MODEL_REPORT.md"
)


with open(

    REPORT_PATH,

    "w",

    encoding="utf-8"

) as f:

    f.write(report)


print()
print("=" * 80)
print("최종 보고서 생성 완료")
print("=" * 80)

print(
    REPORT_PATH.resolve()
)