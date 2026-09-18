# ============================================================
# Business Crisis Cohort
# 시간순 Expanding Window Validation
#
# 검증:
# Fold 1 : 2019~2022H2 학습 -> 2023H1 검증
# Fold 2 : 2019~2023H1 학습 -> 2023H2 검증
# Fold 3 : 2019~2023H2 학습 -> 2024H1 검증
# Fold 4 : 2019~2024H1 학습 -> 2024H2 검증
# Fold 5 : 2019~2024H2 학습 -> 2025H1 검증
#
# 추가 실험:
# 1. COVID 포함
# 2. COVID 제외 (2020~2021)
#
# 최종:
# 2019~2025H1 학습
# -> 2025H2 예측
#
# Models
#   1. Random Forest
#   2. XGBoost
#   3. LightGBM
#
# Imbalance
#   1. None
#   2. ClassWeight
#   3. RandomOver
#   4. RandomUnder
#   5. SMOTE
#
# Label
# 원본 Y:
#   0 = 폐업
#   1 = 생존
#
# 모델링:
#   event_target = 1 : 폐업
#   event_target = 0 : 생존
# ============================================================


# ============================================================
# 0. IMPORT
# ============================================================

from pathlib import Path
import warnings
import platform
import copy

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import font_manager

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    PrecisionRecallDisplay,
)

from imblearn.over_sampling import (
    RandomOverSampler,
    SMOTE,
)

from imblearn.under_sampling import (
    RandomUnderSampler,
)

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

import shap


# ============================================================
# 1. 한글 폰트
# ============================================================

def set_korean_font():

    system = platform.system()

    installed_fonts = {
        font.name
        for font in font_manager.fontManager.ttflist
    }

    if system == "Windows":

        candidates = [
            "Malgun Gothic",
            "맑은 고딕",
        ]

    elif system == "Darwin":

        candidates = [
            "AppleGothic",
            "Arial Unicode MS",
        ]

    else:

        candidates = [
            "Noto Sans CJK KR",
            "Noto Sans KR",
            "NanumGothic",
        ]

    selected_font = None

    for candidate in candidates:

        if candidate in installed_fonts:

            selected_font = candidate
            break

    if selected_font:

        plt.rcParams["font.family"] = selected_font

        print(
            f"[폰트] {selected_font}"
        )

    else:

        print(
            "[경고] 한글 폰트를 "
            "찾지 못했습니다."
        )

    plt.rcParams[
        "axes.unicode_minus"
    ] = False


set_korean_font()


# ============================================================
# 2. CONFIG
# ============================================================

RANDOM_STATE = 42

TARGET_HORIZON = "24M"

TARGET = f"Y_{TARGET_HORIZON}"


# ------------------------------------------------------------
# 데이터
# ------------------------------------------------------------

DATA_PATH = Path(
    r"C:\Users\DC\2026\IMBANK\iM_Blockchain_AI\data\business_crisis_cohort.csv"
)


# ------------------------------------------------------------
# 결과
# ------------------------------------------------------------

OUTPUT_DIR = (
    Path(
        r"C:\Users\DC\2026\IMBANK\iM_Blockchain_AI\results"
    )
    / TARGET_HORIZON
    / "time_validation"
)


METRIC_DIR = OUTPUT_DIR / "metrics"
FIGURE_DIR = OUTPUT_DIR / "figures"
IMPORTANCE_DIR = OUTPUT_DIR / "importance"
SHAP_DIR = OUTPUT_DIR / "shap"
PREDICTION_DIR = OUTPUT_DIR / "predictions"


for directory in [

    METRIC_DIR,
    FIGURE_DIR,
    IMPORTANCE_DIR,
    SHAP_DIR,
    PREDICTION_DIR,

]:

    directory.mkdir(
        parents=True,
        exist_ok=True,
    )


# ============================================================
# 3. 시간 설정
# ============================================================

TRAIN_START = pd.Timestamp(
    "2019-01-01"
)


# ------------------------------------------------------------
# COVID 정의
# ------------------------------------------------------------

COVID_START = pd.Timestamp(
    "2020-01-01"
)

COVID_END = pd.Timestamp(
    "2021-12-31"
)


# ------------------------------------------------------------
# Expanding Window
# ------------------------------------------------------------

VALIDATION_FOLDS = [

    {
        "Fold": 1,
        "Train_End": "2022-12-31",
        "Valid_Start": "2023-01-01",
        "Valid_End": "2023-06-30",
        "Valid_Name": "2023H1",
    },

    {
        "Fold": 2,
        "Train_End": "2023-06-30",
        "Valid_Start": "2023-07-01",
        "Valid_End": "2023-12-31",
        "Valid_Name": "2023H2",
    },

    {
        "Fold": 3,
        "Train_End": "2023-12-31",
        "Valid_Start": "2024-01-01",
        "Valid_End": "2024-06-30",
        "Valid_Name": "2024H1",
    },

    {
        "Fold": 4,
        "Train_End": "2024-06-30",
        "Valid_Start": "2024-07-01",
        "Valid_End": "2024-12-31",
        "Valid_Name": "2024H2",
    },

    {
        "Fold": 5,
        "Train_End": "2024-12-31",
        "Valid_Start": "2025-01-01",
        "Valid_End": "2025-06-30",
        "Valid_Name": "2025H1",
    },

]


# ------------------------------------------------------------
# 최종 모델
#
# 2019~2025H1 학습
# 2025H2 예측
# ------------------------------------------------------------

FINAL_TRAIN_END = pd.Timestamp(
    "2025-06-30"
)

FINAL_PREDICT_START = pd.Timestamp(
    "2025-07-01"
)

FINAL_PREDICT_END = pd.Timestamp(
    "2025-12-31"
)


# ============================================================
# 4. 데이터 로드
# ============================================================

print("\n" + "=" * 80)

print(
    "Business Crisis Cohort "
    "Time Validation"
)

print(
    f"Horizon = {TARGET_HORIZON}"
)

print("=" * 80)


if not DATA_PATH.exists():

    raise FileNotFoundError(
        f"데이터가 없습니다.\n"
        f"{DATA_PATH.resolve()}"
    )


df = pd.read_csv(
    DATA_PATH,
    low_memory=False,
)


print(
    f"\n전체 데이터: "
    f"{len(df):,}"
)

print(
    f"전체 컬럼: "
    f"{df.shape[1]:,}"
)


# ============================================================
# 5. 필수 컬럼
# ============================================================

required_cols = [
    "t0",
    TARGET,
]


for col in required_cols:

    if col not in df.columns:

        raise KeyError(
            f"{col} 컬럼이 없습니다."
        )


# ============================================================
# 6. t0 날짜 변환
# ============================================================

df["t0"] = pd.to_datetime(
    df["t0"],
    errors="coerce",
)


invalid_t0 = (
    df["t0"].isna().sum()
)


print(
    f"\nt0 변환 실패: "
    f"{invalid_t0:,}"
)


df = (
    df[
        df["t0"].notna()
    ]
    .copy()
)


# ============================================================
# 7. 시간 파생변수
# ============================================================

df["t0_year"] = (
    df["t0"].dt.year
)

df["t0_month"] = (
    df["t0"].dt.month
)

df["t0_half"] = np.where(
    df["t0_month"] <= 6,
    1,
    2,
)

df["t0_period"] = (
    df["t0_year"].astype(str)
    + "H"
    + df["t0_half"].astype(str)
)


# ============================================================
# 8. event_target
#
# Y = 0 폐업
# Y = 1 생존
#
# event_target:
# 1 폐업
# 0 생존
# ============================================================

df["event_target"] = np.where(
    df[TARGET].notna(),
    1 - df[TARGET],
    np.nan,
)


# ============================================================
# 9. 모델 Feature 제외 컬럼
# ============================================================

EXCLUDE_COLS = [

    "관리번호",

    "Y_12M",
    "Y_18M",
    "Y_24M",

    "horizon_end_12M",
    "horizon_end_18M",
    "horizon_end_24M",

    "label_status_12M",
    "label_status_18M",
    "label_status_24M",

    "event_target",

    # 날짜는 split 전용
    "t0",

    "episode_start",

    # t0에서 만든 split용 변수
    "t0_year",
    "t0_month",
    "t0_half",
    "t0_period",
]


# ============================================================
# 10. 강제 범주형
# ============================================================

FORCE_CATEGORICAL = [

    "구",
    "읍면동",
    "법정동",
    "법정동코드",
    "행정동",
    "행정동코드",

    "업종",
    "기준분기",
    "quarter_of_year",

]


# ============================================================
# 11. Feature 생성
# ============================================================

def get_feature_columns(data):

    exclude = [

        col

        for col in EXCLUDE_COLS

        if col in data.columns

    ]

    X = data.drop(
        columns=exclude
    ).copy()


    # 상수 컬럼 제거
    constant_cols = [

        col

        for col in X.columns

        if X[col].nunique(
            dropna=False
        ) <= 1

    ]


    if constant_cols:

        X = X.drop(
            columns=constant_cols
        )


    return (
        X.columns.tolist(),
        constant_cols,
    )


feature_columns, constant_cols = (
    get_feature_columns(df)
)


print(
    f"\nFeature 수: "
    f"{len(feature_columns):,}"
)


if constant_cols:

    print(
        f"상수 제거: "
        f"{len(constant_cols):,}"
    )


# ============================================================
# 12. 전처리 생성 함수
#
# 중요:
# Fold마다 Train에만 fit
# ============================================================

def create_preprocessor(
    X_train
):

    categorical_cols = [

        col

        for col in X_train.columns

        if (
            X_train[col].dtype
            == "object"

            or

            col in FORCE_CATEGORICAL
        )

    ]


    numeric_cols = [

        col

        for col in X_train.columns

        if col not in categorical_cols

    ]


    numeric_pipeline = Pipeline(

        steps=[

            (
                "imputer",

                SimpleImputer(
                    strategy="median"
                ),
            ),

        ]
    )


    categorical_pipeline = Pipeline(

        steps=[

            (
                "imputer",

                SimpleImputer(
                    strategy="most_frequent"
                ),
            ),

            (
                "onehot",

                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),

        ]
    )


    preprocessor = ColumnTransformer(

        transformers=[

            (
                "numeric",
                numeric_pipeline,
                numeric_cols,
            ),

            (
                "categorical",
                categorical_pipeline,
                categorical_cols,
            ),

        ],

        remainder="drop",
    )


    return (
        preprocessor,
        numeric_cols,
        categorical_cols,
    )


# ============================================================
# 13. 모델 생성
# ============================================================

def get_models(
    scale_pos_weight=1.0,
    use_class_weight=False,
):

    rf_weight = (
        "balanced"
        if use_class_weight
        else None
    )


    xgb_weight = (
        scale_pos_weight
        if use_class_weight
        else 1.0
    )


    lgbm_weight = (
        "balanced"
        if use_class_weight
        else None
    )


    return {

        "RandomForest":

            RandomForestClassifier(

                n_estimators=500,
                max_depth=None,
                min_samples_leaf=2,

                class_weight=rf_weight,

                random_state=RANDOM_STATE,

                n_jobs=-1,
            ),


        "XGBoost":

            XGBClassifier(

                n_estimators=500,

                learning_rate=0.05,

                max_depth=6,

                min_child_weight=1,

                subsample=0.8,

                colsample_bytree=0.8,

                scale_pos_weight=
                    xgb_weight,

                eval_metric="logloss",

                random_state=
                    RANDOM_STATE,

                n_jobs=-1,
            ),


        "LightGBM":

            LGBMClassifier(

                n_estimators=500,

                learning_rate=0.05,

                num_leaves=31,

                subsample=0.8,

                colsample_bytree=0.8,

                class_weight=
                    lgbm_weight,

                random_state=
                    RANDOM_STATE,

                verbosity=-1,

                n_jobs=-1,
            ),

    }


# ============================================================
# 14. Sampler 생성
#
# 매 Fold마다 새로 생성
# ============================================================

def get_sampler(
    method
):

    if method == "None":

        return None


    if method == "RandomOver":

        return RandomOverSampler(
            random_state=RANDOM_STATE
        )


    if method == "RandomUnder":

        return RandomUnderSampler(
            random_state=RANDOM_STATE
        )


    if method == "SMOTE":

        return SMOTE(
            random_state=RANDOM_STATE,
            k_neighbors=5,
        )


    return None


IMBALANCE_METHODS = [

    "None",
    "ClassWeight",
    "RandomOver",
    "RandomUnder",
    "SMOTE",

]


# ============================================================
# 15. 평가 함수
# ============================================================

def evaluate_model(
    model,
    X_eval,
    y_eval,
):

    prediction = model.predict(
        X_eval
    )


    probability = (
        model
        .predict_proba(
            X_eval
        )[:, 1]
    )


    metrics = {

        "Accuracy":
            accuracy_score(
                y_eval,
                prediction,
            ),

        "Balanced_Accuracy":
            balanced_accuracy_score(
                y_eval,
                prediction,
            ),

        "Precision_Closure":
            precision_score(
                y_eval,
                prediction,
                pos_label=1,
                zero_division=0,
            ),

        "Recall_Closure":
            recall_score(
                y_eval,
                prediction,
                pos_label=1,
                zero_division=0,
            ),

        "F1_Closure":
            f1_score(
                y_eval,
                prediction,
                pos_label=1,
                zero_division=0,
            ),

    }


    # Validation에 클래스 하나만 있는 경우 보호
    if y_eval.nunique() >= 2:

        metrics["ROC_AUC"] = (
            roc_auc_score(
                y_eval,
                probability,
            )
        )

    else:

        metrics["ROC_AUC"] = np.nan


    if (y_eval == 1).sum() > 0:

        metrics["PR_AUC"] = (
            average_precision_score(
                y_eval,
                probability,
            )
        )

    else:

        metrics["PR_AUC"] = np.nan


    return (
        metrics,
        prediction,
        probability,
    )


# ============================================================
# 16. COVID 필터
# ============================================================

def apply_covid_filter(
    train_df,
    covid_mode,
):

    if covid_mode == "포함":

        return train_df.copy()


    elif covid_mode == "제외":

        mask = ~(
            (
                train_df["t0"]
                >= COVID_START
            )
            &
            (
                train_df["t0"]
                <= COVID_END
            )
        )


        return (
            train_df[
                mask
            ]
            .copy()
        )


    else:

        raise ValueError(
            covid_mode
        )


# ============================================================
# 17. 단일 Fold 학습 함수
# ============================================================

def run_single_fold(
    train_df,
    valid_df,
    model_name,
    imbalance_method,
):

    # --------------------------------------------------------
    # 관측 가능한 Label만 학습
    # --------------------------------------------------------

    train_df = (

        train_df[
            train_df[
                "event_target"
            ].notna()
        ]

        .copy()

    )


    valid_df = (

        valid_df[
            valid_df[
                "event_target"
            ].notna()
        ]

        .copy()

    )


    if len(train_df) == 0:

        raise ValueError(
            "Train 데이터가 없습니다."
        )


    if len(valid_df) == 0:

        raise ValueError(
            "Validation 데이터가 없습니다."
        )


    if (
        train_df[
            "event_target"
        ].nunique()
        < 2
    ):

        raise ValueError(
            "Train에 클래스가 "
            "하나만 존재합니다."
        )


    # --------------------------------------------------------
    # X / y
    # --------------------------------------------------------

    X_train = (
        train_df[
            feature_columns
        ]
        .copy()
    )


    X_valid = (
        valid_df[
            feature_columns
        ]
        .copy()
    )


    y_train = (
        train_df[
            "event_target"
        ]
        .astype(int)
    )


    y_valid = (
        valid_df[
            "event_target"
        ]
        .astype(int)
    )


    # --------------------------------------------------------
    # 전처리
    #
    # 반드시 Train에만 fit
    # --------------------------------------------------------

    (
        preprocessor,
        numeric_cols,
        categorical_cols,

    ) = create_preprocessor(
        X_train
    )


    X_train_processed = (
        preprocessor
        .fit_transform(
            X_train
        )
    )


    X_valid_processed = (
        preprocessor
        .transform(
            X_valid
        )
    )


    # --------------------------------------------------------
    # 불균형 비율
    # --------------------------------------------------------

    n_survival = (
        y_train == 0
    ).sum()


    n_closure = (
        y_train == 1
    ).sum()


    if n_closure == 0:

        raise ValueError(
            "폐업 클래스가 없습니다."
        )


    scale_pos_weight = (
        n_survival
        / n_closure
    )


    # --------------------------------------------------------
    # ClassWeight
    # --------------------------------------------------------

    use_class_weight = (
        imbalance_method
        == "ClassWeight"
    )


    models = get_models(

        scale_pos_weight=
            scale_pos_weight,

        use_class_weight=
            use_class_weight,

    )


    model = models[
        model_name
    ]


    # --------------------------------------------------------
    # Sampling
    # --------------------------------------------------------

    if imbalance_method in [

        "RandomOver",
        "RandomUnder",
        "SMOTE",

    ]:

        sampler = get_sampler(
            imbalance_method
        )


        # SMOTE에서 minority가 너무 적은 경우
        if imbalance_method == "SMOTE":

            minority_count = (
                y_train
                .value_counts()
                .min()
            )


            if minority_count <= 5:

                k = max(
                    1,
                    minority_count - 1
                )


                sampler = SMOTE(

                    random_state=
                        RANDOM_STATE,

                    k_neighbors=k,

                )


        (
            X_fit,
            y_fit,

        ) = sampler.fit_resample(

            X_train_processed,
            y_train,

        )


    else:

        X_fit = (
            X_train_processed
        )

        y_fit = (
            y_train.to_numpy()
        )


    # --------------------------------------------------------
    # 학습
    # --------------------------------------------------------

    model.fit(
        X_fit,
        y_fit,
    )


    # --------------------------------------------------------
    # 검증
    # --------------------------------------------------------

    (
        metrics,
        prediction,
        probability,

    ) = evaluate_model(

        model,

        X_valid_processed,

        y_valid,

    )


    return {

        "metrics": metrics,

        "model": model,

        "preprocessor":
            preprocessor,

        "prediction":
            prediction,

        "probability":
            probability,

        "y_valid":
            y_valid,

        "X_valid_processed":
            X_valid_processed,

    }


# ============================================================
# 18. Expanding Window 실험
# ============================================================

fold_results = []


COVID_MODES = [
    "포함",
    "제외",
]


for covid_mode in COVID_MODES:

    print("\n")
    print("=" * 80)

    print(
        f"COVID {covid_mode}"
    )

    print("=" * 80)


    for fold_info in VALIDATION_FOLDS:

        fold = fold_info[
            "Fold"
        ]

        train_end = pd.Timestamp(
            fold_info[
                "Train_End"
            ]
        )

        valid_start = pd.Timestamp(
            fold_info[
                "Valid_Start"
            ]
        )

        valid_end = pd.Timestamp(
            fold_info[
                "Valid_End"
            ]
        )

        valid_name = (
            fold_info[
                "Valid_Name"
            ]
        )


        # ----------------------------------------------------
        # Expanding Train
        # ----------------------------------------------------

        base_train = df[

            (
                df["t0"]
                >= TRAIN_START
            )

            &

            (
                df["t0"]
                <= train_end
            )

        ].copy()


        # ----------------------------------------------------
        # COVID 포함 / 제외
        # ----------------------------------------------------

        train_df = apply_covid_filter(

            base_train,
            covid_mode,

        )


        # ----------------------------------------------------
        # Validation은 절대 COVID 실험에 따라 변경하지 않음
        # ----------------------------------------------------

        valid_df = df[

            (
                df["t0"]
                >= valid_start
            )

            &

            (
                df["t0"]
                <= valid_end
            )

        ].copy()


        # 관측 가능 Label
        train_observed = (
            train_df[
                train_df[
                    "event_target"
                ].notna()
            ]
        )

        valid_observed = (
            valid_df[
                valid_df[
                    "event_target"
                ].notna()
            ]
        )


        print("\n" + "-" * 80)

        print(
            f"Fold {fold} / "
            f"{valid_name}"
        )

        print(
            f"Train: "
            f"2019-01-01 ~ "
            f"{train_end.date()}"
        )

        print(
            f"COVID: {covid_mode}"
        )

        print(
            f"Train N: "
            f"{len(train_observed):,}"
        )

        print(
            f"Valid N: "
            f"{len(valid_observed):,}"
        )

        print(
            f"Valid 폐업: "
            f"{int((valid_observed['event_target'] == 1).sum()):,}"
        )

        print("-" * 80)


        # ----------------------------------------------------
        # Model × Imbalance
        # ----------------------------------------------------

        for imbalance_method in (
            IMBALANCE_METHODS
        ):

            for model_name in [

                "RandomForest",
                "XGBoost",
                "LightGBM",

            ]:

                try:

                    result = (
                        run_single_fold(

                            train_df=
                                train_df,

                            valid_df=
                                valid_df,

                            model_name=
                                model_name,

                            imbalance_method=
                                imbalance_method,

                        )
                    )


                    row = {

                        "Horizon":
                            TARGET_HORIZON,

                        "COVID":
                            covid_mode,

                        "Fold":
                            fold,

                        "Validation":
                            valid_name,

                        "Train_Start":
                            TRAIN_START,

                        "Train_End":
                            train_end,

                        "Valid_Start":
                            valid_start,

                        "Valid_End":
                            valid_end,

                        "Train_N":
                            len(
                                train_observed
                            ),

                        "Valid_N":
                            len(
                                valid_observed
                            ),

                        "Model":
                            model_name,

                        "Imbalance":
                            imbalance_method,

                        **result[
                            "metrics"
                        ],

                    }


                    fold_results.append(
                        row
                    )


                    print(

                        f"[완료] "
                        f"{model_name:<12} "
                        f"{imbalance_method:<12} "

                        f"PR-AUC="
                        f"{row['PR_AUC']:.4f} "

                        f"F1="
                        f"{row['F1_Closure']:.4f}"

                    )


                except Exception as e:

                    print(

                        f"[SKIP] "
                        f"{model_name} / "
                        f"{imbalance_method} / "
                        f"{e}"

                    )


# ============================================================
# 19. Fold 결과 저장
# ============================================================

fold_results_df = pd.DataFrame(
    fold_results
)


if fold_results_df.empty:

    raise RuntimeError(
        "성공한 검증 결과가 없습니다."
    )


fold_results_df.to_csv(

    METRIC_DIR
    / "expanding_window_fold_results.csv",

    index=False,

    encoding="utf-8-sig",

)


# ============================================================
# 20. Fold 평균 성능
# ============================================================

METRIC_COLUMNS = [

    "PR_AUC",
    "ROC_AUC",

    "Precision_Closure",
    "Recall_Closure",
    "F1_Closure",

    "Balanced_Accuracy",
    "Accuracy",

]


mean_results_df = (

    fold_results_df

    .groupby(
        [
            "COVID",
            "Model",
            "Imbalance",
        ],
        as_index=False,
    )

    .agg(

        Fold_Count=(
            "Fold",
            "nunique",
        ),

        Mean_PR_AUC=(
            "PR_AUC",
            "mean",
        ),

        Std_PR_AUC=(
            "PR_AUC",
            "std",
        ),

        Mean_ROC_AUC=(
            "ROC_AUC",
            "mean",
        ),

        Std_ROC_AUC=(
            "ROC_AUC",
            "std",
        ),

        Mean_Precision_Closure=(
            "Precision_Closure",
            "mean",
        ),

        Mean_Recall_Closure=(
            "Recall_Closure",
            "mean",
        ),

        Mean_F1_Closure=(
            "F1_Closure",
            "mean",
        ),

        Mean_Balanced_Accuracy=(
            "Balanced_Accuracy",
            "mean",
        ),

        Mean_Accuracy=(
            "Accuracy",
            "mean",
        ),

    )

)


# 5 Fold 전부 성공한 조합 우선
mean_results_df[
    "All_Folds"
] = (
    mean_results_df[
        "Fold_Count"
    ]
    == len(
        VALIDATION_FOLDS
    )
)


mean_results_df = (

    mean_results_df

    .sort_values(

        [

            "All_Folds",
            "Mean_PR_AUC",
            "Mean_F1_Closure",
            "Mean_Recall_Closure",

        ],

        ascending=[
            False,
            False,
            False,
            False,
        ],

    )

    .reset_index(
        drop=True
    )

)


mean_results_df.to_csv(

    METRIC_DIR
    / "expanding_window_mean_results.csv",

    index=False,

    encoding="utf-8-sig",

)


# ============================================================
# 21. Fold별 Pivot
# ============================================================

fold_pivot = (

    fold_results_df

    .pivot_table(

        index=[
            "COVID",
            "Model",
            "Imbalance",
        ],

        columns=
            "Validation",

        values=
            "PR_AUC",

        aggfunc="mean",

    )

    .reset_index()

)


fold_pivot[
    "평균_PR_AUC"
] = (

    fold_results_df

    .groupby(
        [
            "COVID",
            "Model",
            "Imbalance",
        ]
    )[
        "PR_AUC"
    ]

    .mean()

    .values

)


fold_pivot.to_csv(

    METRIC_DIR
    / "fold_pr_auc_table.csv",

    index=False,

    encoding="utf-8-sig",

)


# ============================================================
# 22. COVID 포함 vs 제외 비교
# ============================================================

covid_comparison = (

    mean_results_df

    .pivot_table(

        index=[
            "Model",
            "Imbalance",
        ],

        columns="COVID",

        values=[
            "Mean_PR_AUC",
            "Mean_ROC_AUC",
            "Mean_F1_Closure",
            "Mean_Recall_Closure",
            "Mean_Precision_Closure",
            "Mean_Balanced_Accuracy",
        ],

    )

)


covid_comparison.columns = [

    f"{metric}_{covid}"

    for metric, covid
    in covid_comparison.columns

]


covid_comparison = (
    covid_comparison
    .reset_index()
)


# 차이: 제외 - 포함
for metric in [

    "Mean_PR_AUC",
    "Mean_ROC_AUC",
    "Mean_F1_Closure",
    "Mean_Recall_Closure",
    "Mean_Precision_Closure",
    "Mean_Balanced_Accuracy",

]:

    included = (
        f"{metric}_포함"
    )

    excluded = (
        f"{metric}_제외"
    )

    if (
        included
        in covid_comparison.columns

        and

        excluded
        in covid_comparison.columns
    ):

        covid_comparison[
            f"{metric}_차이_제외-포함"
        ] = (

            covid_comparison[
                excluded
            ]

            -

            covid_comparison[
                included
            ]

        )


covid_comparison.to_csv(

    METRIC_DIR
    / "covid_performance_comparison.csv",

    index=False,

    encoding="utf-8-sig",

)


# ============================================================
# 23. 평균 결과 출력
# ============================================================

print("\n")
print("=" * 100)

print(
    "5-Fold 평균 성능"
)

print("=" * 100)


display_cols = [

    "COVID",
    "Model",
    "Imbalance",
    "Fold_Count",

    "Mean_PR_AUC",
    "Std_PR_AUC",

    "Mean_ROC_AUC",

    "Mean_Precision_Closure",
    "Mean_Recall_Closure",
    "Mean_F1_Closure",

    "Mean_Balanced_Accuracy",

]


print(

    mean_results_df[
        display_cols
    ]

    .round(4)

    .to_string(
        index=False
    )

)


# ============================================================
# 24. 최종 조합 선택
#
# 5개 Fold 모두 성공한 조합 중
# 평균 PR-AUC 기준
# ============================================================

candidate_results = (

    mean_results_df[
        mean_results_df[
            "All_Folds"
        ]
    ]

    .copy()

)


if candidate_results.empty:

    candidate_results = (
        mean_results_df.copy()
    )


best_row = (
    candidate_results.iloc[0]
)


BEST_COVID = (
    best_row["COVID"]
)

BEST_MODEL = (
    best_row["Model"]
)

BEST_IMBALANCE = (
    best_row["Imbalance"]
)


print("\n")
print("=" * 80)

print(
    "최종 학습 조합"
)

print("=" * 80)


print(
    f"COVID      : "
    f"{BEST_COVID}"
)

print(
    f"Model      : "
    f"{BEST_MODEL}"
)

print(
    f"Imbalance  : "
    f"{BEST_IMBALANCE}"
)

print(
    f"평균 PR-AUC: "
    f"{best_row['Mean_PR_AUC']:.4f}"
)

print(
    f"평균 F1    : "
    f"{best_row['Mean_F1_Closure']:.4f}"
)


# ============================================================
# 25. 최종 학습 데이터
#
# 2019-01-01 ~ 2025-06-30
#
# 반드시 Y 관측 가능한 행만 학습
# ============================================================

final_train_df = df[

    (
        df["t0"]
        >= TRAIN_START
    )

    &

    (
        df["t0"]
        <= FINAL_TRAIN_END
    )

    &

    (
        df[
            "event_target"
        ].notna()
    )

].copy()


final_train_df = (
    apply_covid_filter(

        final_train_df,

        BEST_COVID,

    )
)


# ============================================================
# 26. 2025 하반기 예측 대상
#
# 여기서는 Y 존재 여부와 관계없이
# t0가 2025H2인 모든 행을 예측 대상으로 사용
# ============================================================

final_predict_df = df[

    (
        df["t0"]
        >= FINAL_PREDICT_START
    )

    &

    (
        df["t0"]
        <= FINAL_PREDICT_END
    )

].copy()


print("\n")
print("=" * 80)

print(
    "최종 학습 / 예측 데이터"
)

print("=" * 80)


print(
    f"Train: "
    f"{len(final_train_df):,}"
)

print(
    f"2025H2 예측 대상: "
    f"{len(final_predict_df):,}"
)


if len(final_train_df) == 0:

    raise RuntimeError(
        "최종 학습 데이터가 없습니다."
    )


if len(final_predict_df) == 0:

    raise RuntimeError(
        "2025 하반기 예측 대상이 없습니다."
    )


# ============================================================
# 27. 최종 X/y
# ============================================================

X_final_train = (
    final_train_df[
        feature_columns
    ]
    .copy()
)


y_final_train = (
    final_train_df[
        "event_target"
    ]
    .astype(int)
)


X_final_predict = (
    final_predict_df[
        feature_columns
    ]
    .copy()
)


# ============================================================
# 28. 최종 전처리
# ============================================================

(
    final_preprocessor,
    final_numeric_cols,
    final_categorical_cols,

) = create_preprocessor(
    X_final_train
)


X_final_train_processed = (

    final_preprocessor

    .fit_transform(
        X_final_train
    )

)


X_final_predict_processed = (

    final_preprocessor

    .transform(
        X_final_predict
    )

)


# ============================================================
# 29. Feature 이름
# ============================================================

final_feature_names = (

    final_preprocessor
    .get_feature_names_out()

)


final_feature_names = [

    feature

    .replace(
        "numeric__",
        ""
    )

    .replace(
        "categorical__",
        ""
    )

    for feature
    in final_feature_names

]


# ============================================================
# 30. 최종 class imbalance
# ============================================================

n_survival = (
    y_final_train == 0
).sum()

n_closure = (
    y_final_train == 1
).sum()


if n_closure == 0:

    raise RuntimeError(
        "최종 학습 데이터에 "
        "폐업 클래스가 없습니다."
    )


final_scale_pos_weight = (
    n_survival
    / n_closure
)


# ============================================================
# 31. 최종 모델
# ============================================================

use_class_weight = (
    BEST_IMBALANCE
    == "ClassWeight"
)


final_models = get_models(

    scale_pos_weight=
        final_scale_pos_weight,

    use_class_weight=
        use_class_weight,

)


final_model = (
    final_models[
        BEST_MODEL
    ]
)


# ============================================================
# 32. 최종 Sampling
# ============================================================

if BEST_IMBALANCE in [

    "RandomOver",
    "RandomUnder",
    "SMOTE",

]:

    sampler = get_sampler(
        BEST_IMBALANCE
    )


    if BEST_IMBALANCE == "SMOTE":

        minority_count = (
            y_final_train
            .value_counts()
            .min()
        )


        if minority_count <= 5:

            sampler = SMOTE(

                random_state=
                    RANDOM_STATE,

                k_neighbors=
                    max(
                        1,
                        minority_count - 1
                    ),

            )


    (
        X_final_fit,
        y_final_fit,

    ) = sampler.fit_resample(

        X_final_train_processed,
        y_final_train,

    )


else:

    X_final_fit = (
        X_final_train_processed
    )

    y_final_fit = (
        y_final_train.to_numpy()
    )


# ============================================================
# 33. 최종 모델 학습
# ============================================================

print("\n")
print("=" * 80)

print(
    "최종 모델 학습"
)

print("=" * 80)


final_model.fit(

    X_final_fit,
    y_final_fit,

)


# ============================================================
# 34. 2025H2 예측
# ============================================================

final_probability = (

    final_model

    .predict_proba(
        X_final_predict_processed
    )[:, 1]

)


final_prediction = (

    final_model

    .predict(
        X_final_predict_processed
    )

)


# ============================================================
# 35. 예측 결과
# ============================================================

prediction_output = (
    final_predict_df.copy()
)


prediction_output[
    "폐업예측확률"
] = final_probability


prediction_output[
    "예측폐업여부"
] = final_prediction


prediction_output[
    "예측결과"
] = np.where(

    prediction_output[
        "예측폐업여부"
    ] == 1,

    "폐업",

    "생존",

)


# 위험도 순
prediction_output = (

    prediction_output

    .sort_values(
        "폐업예측확률",
        ascending=False,
    )

    .reset_index(
        drop=True
    )

)


prediction_output.to_csv(

    PREDICTION_DIR
    / "2025H2_closure_predictions.csv",

    index=False,

    encoding="utf-8-sig",

)


# ============================================================
# 36. 2025H2 실제 Label이 존재하는 경우 성능도 계산
#
# 없는 경우 그냥 미래예측 결과만 저장
# ============================================================

observed_2025h2_mask = (
    final_predict_df[
        "event_target"
    ].notna()
)


if observed_2025h2_mask.sum() > 0:

    observed_indices = np.where(
        observed_2025h2_mask
    )[0]


    y_2025h2 = (

        final_predict_df.loc[
            observed_2025h2_mask,
            "event_target"
        ]

        .astype(int)

    )


    pred_2025h2 = (
        final_prediction[
            observed_indices
        ]
    )


    prob_2025h2 = (
        final_probability[
            observed_indices
        ]
    )


    observed_metrics = {

        "Observed_N":
            len(y_2025h2),

        "Accuracy":
            accuracy_score(
                y_2025h2,
                pred_2025h2,
            ),

        "Balanced_Accuracy":
            balanced_accuracy_score(
                y_2025h2,
                pred_2025h2,
            ),

        "Precision_Closure":
            precision_score(
                y_2025h2,
                pred_2025h2,
                zero_division=0,
            ),

        "Recall_Closure":
            recall_score(
                y_2025h2,
                pred_2025h2,
                zero_division=0,
            ),

        "F1_Closure":
            f1_score(
                y_2025h2,
                pred_2025h2,
                zero_division=0,
            ),

    }


    if y_2025h2.nunique() >= 2:

        observed_metrics[
            "ROC_AUC"
        ] = roc_auc_score(

            y_2025h2,
            prob_2025h2,

        )


    if (
        y_2025h2 == 1
    ).sum() > 0:

        observed_metrics[
            "PR_AUC"
        ] = average_precision_score(

            y_2025h2,
            prob_2025h2,

        )


    pd.DataFrame(
        [observed_metrics]
    ).to_csv(

        METRIC_DIR
        / "2025H2_observed_performance.csv",

        index=False,

        encoding="utf-8-sig",

    )


# ============================================================
# 37. 최종 Feature Importance
# ============================================================

if hasattr(
    final_model,
    "feature_importances_"
):

    importance_df = pd.DataFrame({

        "변수":
            final_feature_names,

        "중요도":
            final_model
            .feature_importances_,

    })


    importance_df = (

        importance_df

        .sort_values(
            "중요도",
            ascending=False,
        )

        .reset_index(
            drop=True
        )

    )


    importance_df.to_csv(

        IMPORTANCE_DIR
        / "final_model_feature_importance.csv",

        index=False,

        encoding="utf-8-sig",

    )


    top20 = (
        importance_df
        .head(20)
        .copy()
    )


    fig, ax = plt.subplots(
        figsize=(10, 8)
    )


    ax.barh(

        top20[
            "변수"
        ][::-1],

        top20[
            "중요도"
        ][::-1],

    )


    ax.set_title(

        f"최종 모델 Feature Importance\n"
        f"{BEST_MODEL} / "
        f"{BEST_IMBALANCE} / "
        f"COVID {BEST_COVID}"

    )


    ax.set_xlabel(
        "Feature Importance"
    )

    ax.set_ylabel(
        "변수"
    )


    plt.tight_layout()


    plt.savefig(

        IMPORTANCE_DIR
        / "final_model_feature_importance.png",

        dpi=200,

        bbox_inches="tight",

    )


    plt.close(fig)


# ============================================================
# 38. 최종 SHAP
# ============================================================

print("\n")
print("=" * 80)

print(
    "최종 모델 SHAP"
)

print("=" * 80)


SHAP_SUCCESS = False
shap_importance = None


try:

    SHAP_SAMPLE_SIZE = min(
        2000,
        len(
            X_final_predict_processed
        ),
    )


    rng = (
        np.random.default_rng(
            RANDOM_STATE
        )
    )


    sample_indices = rng.choice(

        len(
            X_final_predict_processed
        ),

        size=
            SHAP_SAMPLE_SIZE,

        replace=False,

    )


    X_shap = (

        X_final_predict_processed[
            sample_indices
        ]

    )


    X_shap_df = pd.DataFrame(

        X_shap,

        columns=
            final_feature_names,

    )


    explainer = (
        shap.TreeExplainer(
            final_model
        )
    )


    shap_result = (
        explainer(
            X_shap_df
        )
    )


    shap_values = (
        shap_result.values
    )


    if shap_values.ndim == 3:

        closure_shap_values = (

            shap_values[
                :,
                :,
                1
            ]

        )

    else:

        closure_shap_values = (
            shap_values
        )


    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    plt.figure(
        figsize=(11, 9)
    )


    shap.summary_plot(

        closure_shap_values,

        X_shap_df,

        max_display=20,

        show=False,

    )


    plt.title(

        "2025H2 폐업 예측 "
        "SHAP Summary\n"

        f"{BEST_MODEL} / "
        f"{BEST_IMBALANCE} / "
        f"COVID {BEST_COVID}"

    )


    plt.tight_layout()


    plt.savefig(

        SHAP_DIR
        / "final_model_shap_summary.png",

        dpi=200,

        bbox_inches="tight",

    )


    plt.close("all")


    # --------------------------------------------------------
    # SHAP Importance
    # --------------------------------------------------------

    mean_abs_shap = (

        np.abs(
            closure_shap_values
        )

        .mean(
            axis=0
        )

    )


    shap_importance = pd.DataFrame({

        "변수":
            final_feature_names,

        "평균절대SHAP":
            mean_abs_shap,

    })


    shap_importance = (

        shap_importance

        .sort_values(
            "평균절대SHAP",
            ascending=False,
        )

        .reset_index(
            drop=True
        )

    )


    shap_importance.to_csv(

        SHAP_DIR
        / "final_model_shap_importance.csv",

        index=False,

        encoding="utf-8-sig",

    )


    shap_top20 = (
        shap_importance
        .head(20)
        .copy()
    )


    fig, ax = plt.subplots(
        figsize=(10, 8)
    )


    ax.barh(

        shap_top20[
            "변수"
        ][::-1],

        shap_top20[
            "평균절대SHAP"
        ][::-1],

    )


    ax.set_title(
        "2025H2 폐업 예측 "
        "SHAP 중요도 Top 20"
    )


    ax.set_xlabel(
        "평균 |SHAP value|"
    )

    ax.set_ylabel(
        "변수"
    )


    plt.tight_layout()


    plt.savefig(

        SHAP_DIR
        / "final_model_shap_bar.png",

        dpi=200,

        bbox_inches="tight",

    )


    plt.close(fig)


    SHAP_SUCCESS = True


except Exception as e:

    print(
        f"[SHAP 오류] {e}"
    )


# ============================================================
# 39. Fold PR-AUC 그래프
# ============================================================

for covid_mode in COVID_MODES:

    subset = (
        fold_results_df[
            fold_results_df[
                "COVID"
            ] == covid_mode
        ]
    )


    # 최종 조합과 동일한 조합
    subset = subset[

        (
            subset[
                "Model"
            ]
            == BEST_MODEL
        )

        &

        (
            subset[
                "Imbalance"
            ]
            == BEST_IMBALANCE
        )

    ]


    if len(subset) > 0:

        fig, ax = plt.subplots(
            figsize=(9, 5)
        )


        ax.plot(

            subset[
                "Validation"
            ],

            subset[
                "PR_AUC"
            ],

            marker="o",

        )


        ax.set_title(

            f"Expanding Window PR-AUC\n"
            f"{BEST_MODEL} / "
            f"{BEST_IMBALANCE} / "
            f"COVID {covid_mode}"

        )


        ax.set_xlabel(
            "검증 기간"
        )

        ax.set_ylabel(
            "PR-AUC"
        )


        plt.tight_layout()


        plt.savefig(

            FIGURE_DIR
            / (
                f"fold_pr_auc_"
                f"covid_{covid_mode}.png"
            ),

            dpi=200,

            bbox_inches="tight",

        )


        plt.close(fig)


# ============================================================
# 40. 결과 요약
# ============================================================

summary = []


summary.append(
    "Business Crisis Cohort "
    "시간순 검증 결과"
)

summary.append(
    "=" * 70
)

summary.append("")

summary.append(
    f"예측 Horizon: "
    f"{TARGET_HORIZON}"
)

summary.append("")

summary.append(
    "[Expanding Window]"
)

summary.append(
    "Fold1: 2019~2022H2 "
    "-> 2023H1"
)

summary.append(
    "Fold2: 2019~2023H1 "
    "-> 2023H2"
)

summary.append(
    "Fold3: 2019~2023H2 "
    "-> 2024H1"
)

summary.append(
    "Fold4: 2019~2024H1 "
    "-> 2024H2"
)

summary.append(
    "Fold5: 2019~2024H2 "
    "-> 2025H1"
)

summary.append("")

summary.append(
    "[COVID 비교]"
)

summary.append(
    "포함: 전체 학습기간 사용"
)

summary.append(
    f"제외: "
    f"{COVID_START.date()} ~ "
    f"{COVID_END.date()} 제거"
)

summary.append("")

summary.append(
    "[최종 선택]"
)

summary.append(
    f"COVID: {BEST_COVID}"
)

summary.append(
    f"Model: {BEST_MODEL}"
)

summary.append(
    f"Imbalance: "
    f"{BEST_IMBALANCE}"
)

summary.append(
    f"평균 PR-AUC: "
    f"{best_row['Mean_PR_AUC']:.4f}"
)

summary.append(
    f"평균 ROC-AUC: "
    f"{best_row['Mean_ROC_AUC']:.4f}"
)

summary.append(
    f"평균 폐업 Precision: "
    f"{best_row['Mean_Precision_Closure']:.4f}"
)

summary.append(
    f"평균 폐업 Recall: "
    f"{best_row['Mean_Recall_Closure']:.4f}"
)

summary.append(
    f"평균 폐업 F1: "
    f"{best_row['Mean_F1_Closure']:.4f}"
)

summary.append(
    f"평균 Balanced Accuracy: "
    f"{best_row['Mean_Balanced_Accuracy']:.4f}"
)

summary.append("")

summary.append(
    "[최종 모델]"
)

summary.append(
    "학습: 2019-01-01 ~ "
    "2025-06-30"
)

summary.append(
    "예측: 2025-07-01 ~ "
    "2025-12-31"
)

summary.append(
    f"최종 학습 N: "
    f"{len(final_train_df):,}"
)

summary.append(
    f"2025H2 예측 N: "
    f"{len(final_predict_df):,}"
)


summary_text = (
    "\n".join(
        summary
    )
)


with open(

    OUTPUT_DIR
    / "summary.txt",

    "w",

    encoding="utf-8",

) as f:

    f.write(
        summary_text
    )


# ============================================================
# 41. 최종 출력
# ============================================================

print("\n")
print(summary_text)


print("\n")
print("=" * 80)

print(
    "모든 작업 완료"
)

print("=" * 80)


print(
    "\n[저장 파일]"
)

print(
    "1. metrics/"
    "expanding_window_fold_results.csv"
)

print(
    "2. metrics/"
    "expanding_window_mean_results.csv"
)

print(
    "3. metrics/"
    "fold_pr_auc_table.csv"
)

print(
    "4. metrics/"
    "covid_performance_comparison.csv"
)

print(
    "5. predictions/"
    "2025H2_closure_predictions.csv"
)

print(
    "6. importance/"
    "final_model_feature_importance.csv"
)

print(
    "7. shap/"
    "final_model_shap_importance.csv"
)

print(
    "8. summary.txt"
)

print(
    f"\n결과 폴더:\n"
    f"{OUTPUT_DIR.resolve()}"
)