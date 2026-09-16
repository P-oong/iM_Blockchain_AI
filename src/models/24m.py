# ============================================================
# Business Crisis Cohort
# 회복탄력성 / 폐업 위험 예측 모델링
#
# Models
#   1. Random Forest
#   2. XGBoost
#   3. LightGBM
#
# Imbalance Methods
#   1. None
#   2. Class Weight
#   3. Random Over Sampling
#   4. Random Under Sampling
#   5. SMOTE
#
# Original Label
#   Y_* = 0 : 폐업
#   Y_* = 1 : 생존
#
# Modeling Label
#   event_target = 1 : 폐업
#   event_target = 0 : 생존
#
# 주요 평가 지표
#   - PR-AUC
#   - ROC-AUC
#   - 폐업 Precision
#   - 폐업 Recall
#   - 폐업 F1
#   - Balanced Accuracy
#
# 결과
#   - 모델 비교 CSV
#   - Confusion Matrix
#   - ROC Curve
#   - PR Curve
#   - Feature Importance
#   - SHAP Summary
#   - SHAP Importance
#   - 텍스트 요약
# ============================================================


# ============================================================
# 0. BASIC IMPORT
# ============================================================

from pathlib import Path
import warnings
import platform

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd


# ============================================================
# 1. MATPLOTLIB
# ============================================================

# Tkinter GUI 오류 방지를 위해
# pyplot보다 반드시 먼저 실행

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import font_manager


# ============================================================
# 2. 한글 폰트 설정
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

    if selected_font is not None:

        plt.rcParams[
            "font.family"
        ] = selected_font

        print(
            f"[폰트] {selected_font}"
        )

    else:

        print(
            "[경고] 한글 폰트를 "
            "자동으로 찾지 못했습니다."
        )

    # 마이너스 깨짐 방지
    plt.rcParams[
        "axes.unicode_minus"
    ] = False


set_korean_font()


# ============================================================
# 3. ML IMPORT
# ============================================================

from sklearn.model_selection import (
    train_test_split,
)

from sklearn.compose import (
    ColumnTransformer,
)

from sklearn.preprocessing import (
    OneHotEncoder,
)

from sklearn.impute import (
    SimpleImputer,
)

from sklearn.pipeline import (
    Pipeline,
)

from sklearn.ensemble import (
    RandomForestClassifier,
)

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
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

from xgboost import (
    XGBClassifier,
)

from lightgbm import (
    LGBMClassifier,
)

import shap


# ============================================================
# 4. CONFIG
# ============================================================

RANDOM_STATE = 42

TEST_SIZE = 0.20


# ------------------------------------------------------------
# 예측 Horizon
#
# "12M"
# "18M"
# "24M"
#
# 여기만 변경하면 됨
# ------------------------------------------------------------

TARGET_HORIZON = "24M"


TARGET = (
    f"Y_{TARGET_HORIZON}"
)

LABEL_STATUS = (
    f"label_status_{TARGET_HORIZON}"
)


# ------------------------------------------------------------
# 데이터 경로
# ------------------------------------------------------------

DATA_PATH = Path(
    r"C:\Users\DC\2026\IMBANK\iM_Blockchain_AI\data\business_crisis_cohort.csv"
)


# ------------------------------------------------------------
# 결과 저장 경로
# ------------------------------------------------------------

OUTPUT_DIR = (
    Path(r"C:\Users\DC\2026\IMBANK\iM_Blockchain_AI\results")
    / TARGET_HORIZON
)

METRIC_DIR = (
    OUTPUT_DIR
    / "metrics"
)

FIGURE_DIR = (
    OUTPUT_DIR
    / "figures"
)

IMPORTANCE_DIR = (
    OUTPUT_DIR
    / "importance"
)

SHAP_DIR = (
    OUTPUT_DIR
    / "shap"
)


for directory in [

    METRIC_DIR,
    FIGURE_DIR,
    IMPORTANCE_DIR,
    SHAP_DIR,

]:

    directory.mkdir(
        parents=True,
        exist_ok=True,
    )


# ============================================================
# 5. 데이터 로드
# ============================================================

print("\n")
print("=" * 70)

print(
    "Business Crisis Cohort "
    "Modeling"
)

print(
    f"Horizon = "
    f"{TARGET_HORIZON}"
)

print("=" * 70)


if not DATA_PATH.exists():

    raise FileNotFoundError(
        f"\n데이터 파일을 "
        f"찾을 수 없습니다.\n"
        f"{DATA_PATH.resolve()}"
    )


df = pd.read_csv(
    DATA_PATH
)


print(
    f"\n전체 데이터 수: "
    f"{len(df):,}"
)

print(
    f"전체 컬럼 수: "
    f"{df.shape[1]:,}"
)


# ============================================================
# 6. TARGET 존재 확인
# ============================================================

if TARGET not in df.columns:

    raise KeyError(
        f"{TARGET} 컬럼이 없습니다."
    )


# ============================================================
# 7. 관측 가능한 Label만 사용
# ============================================================

# label_status 문자열에 의존하지 않고
# 실제 Y가 존재하는 행만 사용

model_df = (
    df[
        df[TARGET].notna()
    ]
    .copy()
)


print("\n")
print("-" * 70)

print(
    f"{TARGET} 관측 가능 데이터"
)

print("-" * 70)

print(
    f"{len(model_df):,} / "
    f"{len(df):,}"
)


# ============================================================
# 8. 원본 Label 확인
# ============================================================

print(
    "\n[원본 Label 빈도]"
)

print(
    model_df[
        TARGET
    ]
    .value_counts(
        dropna=False
    )
    .sort_index()
)


print(
    "\n[원본 Label 비율]"
)

print(
    (
        model_df[
            TARGET
        ]
        .value_counts(
            normalize=True
        )
        .sort_index()
        * 100
    )
    .round(2)
)


# ============================================================
# 9. Target 재정의
# ============================================================

# 기존:
#
# 0 = 폐업
# 1 = 생존
#
# 모델링:
#
# 1 = 폐업
# 0 = 생존
#
# sklearn에서 positive class를
# 폐업으로 통일하기 위함


model_df[
    "event_target"
] = (

    1
    - model_df[
        TARGET
    ].astype(int)

)


print("\n")
print("-" * 70)

print(
    "모델링 Target"
)

print("-" * 70)

print(
    "0 = 생존"
)

print(
    "1 = 폐업"
)


target_counts = (
    model_df[
        "event_target"
    ]
    .value_counts()
    .sort_index()
)


print(
    "\n[Target 빈도]"
)

print(
    target_counts.rename(
        {
            0: "생존",
            1: "폐업",
        }
    )
)


print(
    "\n[Target 비율]"
)

print(
    (
        target_counts
        / target_counts.sum()
        * 100
    )
    .round(2)
    .rename(
        {
            0: "생존",
            1: "폐업",
        }
    )
)


# ============================================================
# 10. Leakage 변수 제거
# ============================================================

# Y / horizon / label_status는
# 절대 X에 사용하지 않음


EXCLUDE_COLS = [

    # ------------------------
    # ID
    # ------------------------

    "관리번호",

    # ------------------------
    # 모든 Y
    # ------------------------

    "Y_12M",
    "Y_18M",
    "Y_24M",

    # ------------------------
    # Horizon
    # ------------------------

    "horizon_end_12M",
    "horizon_end_18M",
    "horizon_end_24M",

    # ------------------------
    # Label status
    # ------------------------

    "label_status_12M",
    "label_status_18M",
    "label_status_24M",

    # ------------------------
    # 새 Target
    # ------------------------

    "event_target",

    # ------------------------
    # 날짜
    # ------------------------

    "t0",
    "episode_start",
]


EXCLUDE_COLS = [

    col

    for col in EXCLUDE_COLS

    if col in model_df.columns

]


X = model_df.drop(
    columns=EXCLUDE_COLS
)

y = model_df[
    "event_target"
]


print(
    "\n제외된 컬럼:"
)

for col in EXCLUDE_COLS:

    print(
        f" - {col}"
    )


# ============================================================
# 11. 상수 컬럼 제거
# ============================================================

constant_cols = [

    col

    for col in X.columns

    if X[col].nunique(
        dropna=False
    ) <= 1

]


if constant_cols:

    print(
        "\n[상수 컬럼 제거]"
    )

    for col in constant_cols:

        print(
            f" - {col}"
        )

    X = X.drop(
        columns=constant_cols
    )


# ============================================================
# 12. 범주형 / 수치형 구분
# ============================================================

# 숫자로 저장되어 있어도
# 범주형 의미를 가진 변수

FORCE_CATEGORICAL = [

    "구",
    "읍면동",
    "업종",
    "기준분기",
    "행정동코드",
    "quarter_of_year",

]


categorical_cols = [

    col

    for col in X.columns

    if (

        X[col].dtype
        == "object"

        or

        col in FORCE_CATEGORICAL

    )

]


numeric_cols = [

    col

    for col in X.columns

    if col not in categorical_cols

]


print("\n")
print("-" * 70)

print(
    "Feature 정보"
)

print("-" * 70)


print(
    f"전체 Feature: "
    f"{X.shape[1]}"
)


print(
    f"Numeric: "
    f"{len(numeric_cols)}"
)


print(
    f"Categorical: "
    f"{len(categorical_cols)}"
)


print(
    "\n[범주형 변수]"
)

for col in categorical_cols:

    print(
        f" - {col}"
    )


# ============================================================
# 13. Train / Test Split
# ============================================================

X_train, X_test, \
y_train, y_test = (

    train_test_split(

        X,
        y,

        test_size=TEST_SIZE,

        stratify=y,

        random_state=RANDOM_STATE,

    )

)


print("\n")
print("-" * 70)

print(
    "Train / Test"
)

print("-" * 70)


print(
    f"Train: "
    f"{len(X_train):,}"
)


print(
    f"Test : "
    f"{len(X_test):,}"
)


print(
    "\n[Train Target]"
)

print(
    y_train
    .value_counts()
    .sort_index()
    .rename(
        {
            0: "생존",
            1: "폐업",
        }
    )
)


print(
    "\n[Test Target]"
)

print(
    y_test
    .value_counts()
    .sort_index()
    .rename(
        {
            0: "생존",
            1: "폐업",
        }
    )
)


# ============================================================
# 14. 전처리 Pipeline
# ============================================================

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


# ============================================================
# 15. Train으로 전처리 Fit
# ============================================================

X_train_processed = (
    preprocessor
    .fit_transform(
        X_train
    )
)


X_test_processed = (
    preprocessor
    .transform(
        X_test
    )
)


# ============================================================
# 16. Feature 이름
# ============================================================

feature_names = (

    preprocessor
    .get_feature_names_out()

)


feature_names = [

    feature
    .replace(
        "numeric__",
        ""
    )
    .replace(
        "categorical__",
        ""
    )

    for feature in feature_names

]


print(
    f"\n전처리 후 Feature 수: "
    f"{len(feature_names):,}"
)


# ============================================================
# 17. NaN / Inf 확인
# ============================================================

if np.isnan(
    X_train_processed
).any():

    raise ValueError(
        "전처리 후 Train에 "
        "NaN이 존재합니다."
    )


if np.isinf(
    X_train_processed
).any():

    raise ValueError(
        "전처리 후 Train에 "
        "Inf가 존재합니다."
    )


# ============================================================
# 18. 불균형 비율
# ============================================================

n_survival = (
    y_train == 0
).sum()


n_closure = (
    y_train == 1
).sum()


if n_closure == 0:

    raise ValueError(
        "Train 데이터에 "
        "폐업 클래스가 없습니다."
    )


scale_pos_weight = (

    n_survival
    / n_closure

)


print(
    "\nscale_pos_weight = "
    f"{scale_pos_weight:.4f}"
)


# ============================================================
# 19. Sampling Methods
# ============================================================

SAMPLERS = {

    "None":

        None,


    "RandomOver":

        RandomOverSampler(

            random_state=
                RANDOM_STATE

        ),


    "RandomUnder":

        RandomUnderSampler(

            random_state=
                RANDOM_STATE

        ),


    "SMOTE":

        SMOTE(

            random_state=
                RANDOM_STATE,

            k_neighbors=5,

        ),

}


# ============================================================
# 20. 모델 생성 함수
# ============================================================

def get_models(
    use_class_weight=False
):

    # -------------------------
    # Random Forest
    # -------------------------

    rf_weight = (

        "balanced"

        if use_class_weight

        else None

    )


    # -------------------------
    # XGBoost
    # -------------------------

    xgb_weight = (

        scale_pos_weight

        if use_class_weight

        else 1.0

    )


    # -------------------------
    # LightGBM
    # -------------------------

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

                class_weight=
                    rf_weight,

                random_state=
                    RANDOM_STATE,

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
# 21. 평가 함수
# ============================================================

def evaluate_model(
    model,
    X_eval,
    y_eval,
):

    prediction = (
        model.predict(
            X_eval
        )
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


        # Positive = 폐업

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


        "ROC_AUC":

            roc_auc_score(
                y_eval,
                probability,
            ),


        "PR_AUC":

            average_precision_score(
                y_eval,
                probability,
            ),

    }


    return (
        metrics,
        prediction,
        probability,
    )


# ============================================================
# 22. 평가 그래프
# ============================================================

def save_evaluation_plots(
    model_name,
    imbalance_name,
    y_true,
    prediction,
    probability,
):

    file_prefix = (

        f"{model_name}_"
        f"{imbalance_name}"

    )


    # ========================================================
    # Confusion Matrix
    # ========================================================

    fig, ax = plt.subplots(
        figsize=(6, 5)
    )


    ConfusionMatrixDisplay.from_predictions(

        y_true,

        prediction,

        display_labels=[
            "생존",
            "폐업",
        ],

        cmap="Blues",

        ax=ax,

    )


    ax.set_title(

        f"{TARGET_HORIZON} "
        f"혼동행렬\n"
        f"{model_name} / "
        f"{imbalance_name}"

    )


    ax.set_xlabel(
        "예측값"
    )


    ax.set_ylabel(
        "실제값"
    )


    plt.tight_layout()


    plt.savefig(

        FIGURE_DIR
        / (
            f"{file_prefix}_"
            f"confusion_matrix.png"
        ),

        dpi=200,

        bbox_inches="tight",

    )


    plt.close(fig)


    # ========================================================
    # ROC Curve
    # ========================================================

    fig, ax = plt.subplots(
        figsize=(6, 5)
    )


    RocCurveDisplay.from_predictions(

        y_true,

        probability,

        ax=ax,

    )


    ax.set_title(

        f"{TARGET_HORIZON} "
        f"ROC Curve\n"
        f"{model_name} / "
        f"{imbalance_name}"

    )


    plt.tight_layout()


    plt.savefig(

        FIGURE_DIR
        / (
            f"{file_prefix}_"
            f"roc_curve.png"
        ),

        dpi=200,

        bbox_inches="tight",

    )


    plt.close(fig)


    # ========================================================
    # PR Curve
    # ========================================================

    fig, ax = plt.subplots(
        figsize=(6, 5)
    )


    PrecisionRecallDisplay.from_predictions(

        y_true,

        probability,

        ax=ax,

    )


    ax.set_title(

        f"{TARGET_HORIZON} "
        f"Precision-Recall Curve\n"
        f"{model_name} / "
        f"{imbalance_name}"

    )


    plt.tight_layout()


    plt.savefig(

        FIGURE_DIR
        / (
            f"{file_prefix}_"
            f"pr_curve.png"
        ),

        dpi=200,

        bbox_inches="tight",

    )


    plt.close(fig)


# ============================================================
# 23. Feature Importance
# ============================================================

def save_feature_importance(
    model,
    model_name,
    imbalance_name,
):

    if not hasattr(
        model,
        "feature_importances_"
    ):

        return


    importance_df = pd.DataFrame(

        {

            "변수":
                feature_names,

            "중요도":
                model
                .feature_importances_,

        }

    )


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


    # CSV

    importance_df.to_csv(

        IMPORTANCE_DIR
        / (
            f"{model_name}_"
            f"{imbalance_name}.csv"
        ),

        index=False,

        encoding="utf-8-sig",

    )


    # Top 20

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

        f"{TARGET_HORIZON} "
        f"변수 중요도 Top 20\n"
        f"{model_name} / "
        f"{imbalance_name}"

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
        / (
            f"{model_name}_"
            f"{imbalance_name}.png"
        ),

        dpi=200,

        bbox_inches="tight",

    )


    plt.close(fig)


# ============================================================
# 24. 결과 저장 객체
# ============================================================

results = []

trained_models = {}


# ============================================================
# 25. Sampling 실험
# ============================================================

for (
    imbalance_name,
    sampler
) in SAMPLERS.items():


    print("\n")
    print("=" * 70)

    print(
        f"불균형 처리: "
        f"{imbalance_name}"
    )

    print("=" * 70)


    # --------------------------------------------------------
    # Sampling
    # --------------------------------------------------------

    if sampler is None:

        X_resampled = (
            X_train_processed
        )

        y_resampled = (
            y_train.to_numpy()
        )


    else:

        try:

            X_resampled, \
            y_resampled = (

                sampler
                .fit_resample(

                    X_train_processed,

                    y_train,

                )

            )

        except ValueError as e:

            print(
                f"[SKIP] "
                f"{imbalance_name}: "
                f"{e}"
            )

            continue


    unique_values, counts = (
        np.unique(

            y_resampled,

            return_counts=True,

        )
    )


    print(
        "학습 Label:"
    )


    print(
        dict(
            zip(
                unique_values,
                counts,
            )
        )
    )


    # --------------------------------------------------------
    # Models
    # --------------------------------------------------------

    models = get_models(
        use_class_weight=False
    )


    for (
        model_name,
        model
    ) in models.items():


        print(
            f"\n[학습] "
            f"{model_name} "
            f"/ {imbalance_name}"
        )


        # ----------------------------------------------------
        # Train
        # ----------------------------------------------------

        model.fit(

            X_resampled,

            y_resampled,

        )


        # ----------------------------------------------------
        # Evaluation
        # ----------------------------------------------------

        (
            metrics,
            prediction,
            probability,

        ) = evaluate_model(

            model,

            X_test_processed,

            y_test,

        )


        # ----------------------------------------------------
        # Result
        # ----------------------------------------------------

        result_row = {

            "Horizon":
                TARGET_HORIZON,

            "Model":
                model_name,

            "Imbalance":
                imbalance_name,

            **metrics,

        }


        results.append(
            result_row
        )


        # ----------------------------------------------------
        # Model 저장
        # ----------------------------------------------------

        trained_models[
            (
                model_name,
                imbalance_name,
            )
        ] = model


        # ----------------------------------------------------
        # Plot
        # ----------------------------------------------------

        save_evaluation_plots(

            model_name,

            imbalance_name,

            y_test,

            prediction,

            probability,

        )


        # ----------------------------------------------------
        # Importance
        # ----------------------------------------------------

        save_feature_importance(

            model,

            model_name,

            imbalance_name,

        )


# ============================================================
# 26. Class Weight 실험
# ============================================================

print("\n")
print("=" * 70)

print(
    "불균형 처리: "
    "ClassWeight"
)

print("=" * 70)


weighted_models = (
    get_models(
        use_class_weight=True
    )
)


for (
    model_name,
    model
) in weighted_models.items():


    imbalance_name = (
        "ClassWeight"
    )


    print(
        f"\n[학습] "
        f"{model_name} "
        f"/ {imbalance_name}"
    )


    model.fit(

        X_train_processed,

        y_train,

    )


    (
        metrics,
        prediction,
        probability,

    ) = evaluate_model(

        model,

        X_test_processed,

        y_test,

    )


    result_row = {

        "Horizon":
            TARGET_HORIZON,

        "Model":
            model_name,

        "Imbalance":
            imbalance_name,

        **metrics,

    }


    results.append(
        result_row
    )


    trained_models[
        (
            model_name,
            imbalance_name,
        )
    ] = model


    save_evaluation_plots(

        model_name,

        imbalance_name,

        y_test,

        prediction,

        probability,

    )


    save_feature_importance(

        model,

        model_name,

        imbalance_name,

    )


# ============================================================
# 27. 결과 DataFrame
# ============================================================

results_df = pd.DataFrame(
    results
)


if results_df.empty:

    raise RuntimeError(
        "성공적으로 학습된 "
        "모델이 없습니다."
    )


# ============================================================
# 28. 결과 정렬
# ============================================================

# 불균형 데이터에서는
# PR-AUC를 우선 기준으로 사용

results_df = (

    results_df

    .sort_values(

        [

            "PR_AUC",
            "F1_Closure",
            "Recall_Closure",

        ],

        ascending=[

            False,
            False,
            False,

        ],

    )

    .reset_index(
        drop=True
    )

)


# ============================================================
# 29. 결과 저장
# ============================================================

results_df.to_csv(

    METRIC_DIR
    / "model_comparison.csv",

    index=False,

    encoding="utf-8-sig",

)


print("\n")
print("=" * 70)

print(
    "전체 모델 비교"
)

print("=" * 70)


display_columns = [

    "Model",
    "Imbalance",
    "PR_AUC",
    "ROC_AUC",
    "Precision_Closure",
    "Recall_Closure",
    "F1_Closure",
    "Balanced_Accuracy",
    "Accuracy",

]


print(

    results_df[
        display_columns
    ]

    .round(4)

    .to_string(
        index=False
    )

)


# ============================================================
# 30. 최상위 모델
# ============================================================

# 주의:
# 여기서 "best"는
# 단순히 PR-AUC가 가장 높은 실험 조합을 의미함.
# 최종 운영모델 선정은 별도 검증 필요.

best_row = (
    results_df.iloc[0]
)


best_model_name = (
    best_row[
        "Model"
    ]
)


best_method = (
    best_row[
        "Imbalance"
    ]
)


best_model = (

    trained_models[
        (
            best_model_name,
            best_method,
        )
    ]

)


print("\n")
print("=" * 70)

print(
    "PR-AUC 기준 "
    "상위 모델"
)

print("=" * 70)


print(
    f"Model       : "
    f"{best_model_name}"
)


print(
    f"Imbalance   : "
    f"{best_method}"
)


print(
    f"PR-AUC      : "
    f"{best_row['PR_AUC']:.4f}"
)


print(
    f"ROC-AUC     : "
    f"{best_row['ROC_AUC']:.4f}"
)


print(
    f"폐업 Recall : "
    f"{best_row['Recall_Closure']:.4f}"
)


print(
    f"폐업 Precision : "
    f"{best_row['Precision_Closure']:.4f}"
)


print(
    f"폐업 F1     : "
    f"{best_row['F1_Closure']:.4f}"
)


# ============================================================
# 31. SHAP 데이터 Sampling
# ============================================================

print("\n")
print("=" * 70)

print(
    "SHAP 분석 시작"
)

print("=" * 70)


SHAP_SAMPLE_SIZE = min(

    2000,

    len(
        X_test_processed
    ),

)


rng = (
    np.random.default_rng(
        RANDOM_STATE
    )
)


sample_indices = (
    rng.choice(

        len(
            X_test_processed
        ),

        size=
            SHAP_SAMPLE_SIZE,

        replace=False,

    )
)


X_shap = (

    X_test_processed[
        sample_indices
    ]

)


X_shap_df = (
    pd.DataFrame(

        X_shap,

        columns=
            feature_names,

    )
)


# ============================================================
# 32. SHAP 계산
# ============================================================

try:

    explainer = (
        shap.TreeExplainer(
            best_model
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


    # --------------------------------------------------------
    # Binary classifier 처리
    # --------------------------------------------------------

    if shap_values.ndim == 3:

        # class 1 = 폐업
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


    # ========================================================
    # 33. SHAP Summary Plot
    # ========================================================

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

        f"{TARGET_HORIZON} "
        f"폐업 예측 SHAP Summary\n"
        f"{best_model_name} / "
        f"{best_method}"

    )


    plt.tight_layout()


    plt.savefig(

        SHAP_DIR
        / (
            "best_model_"
            "shap_summary.png"
        ),

        dpi=200,

        bbox_inches="tight",

    )


    plt.close("all")


    # ========================================================
    # 34. SHAP 중요도
    # ========================================================

    mean_abs_shap = (

        np.abs(
            closure_shap_values
        )

        .mean(
            axis=0
        )

    )


    shap_importance = (
        pd.DataFrame(

            {

                "변수":
                    feature_names,

                "평균절대SHAP":
                    mean_abs_shap,

            }

        )
    )


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


    # ========================================================
    # 35. SHAP CSV
    # ========================================================

    shap_importance.to_csv(

        SHAP_DIR
        / (
            "best_model_"
            "shap_importance.csv"
        ),

        index=False,

        encoding="utf-8-sig",

    )


    # ========================================================
    # 36. SHAP Bar
    # ========================================================

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

        f"{TARGET_HORIZON} "
        f"폐업 예측 SHAP 중요도 Top 20\n"
        f"{best_model_name} / "
        f"{best_method}"

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
        / (
            "best_model_"
            "shap_bar.png"
        ),

        dpi=200,

        bbox_inches="tight",

    )


    plt.close(fig)


    SHAP_SUCCESS = True


except Exception as e:

    SHAP_SUCCESS = False

    shap_importance = None

    print(
        "\n[SHAP 오류]"
    )

    print(
        str(e)
    )


# ============================================================
# 37. 사람이 읽을 수 있는 요약 생성
# ============================================================

summary = []


summary.append(
    "Business Crisis Cohort 모델링 결과"
)

summary.append(
    "=" * 60
)


summary.append(
    f"예측 기간: "
    f"{TARGET_HORIZON}"
)


summary.append(
    f"분석 대상 사업체: "
    f"{len(model_df):,}개"
)


summary.append("")


# ------------------------------------------------------------
# Label
# ------------------------------------------------------------

summary.append(
    "[예측 대상]"
)


summary.append(
    f"매출위기 노출 이후 "
    f"{TARGET_HORIZON} 이내 "
    f"폐업 여부"
)


summary.append("")


# ------------------------------------------------------------
# Class distribution
# ------------------------------------------------------------

summary.append(
    "[데이터 구성]"
)


summary.append(

    f"생존: "
    f"{target_counts.get(0, 0):,}개"

)


summary.append(

    f"폐업: "
    f"{target_counts.get(1, 0):,}개"

)


closure_rate = (

    target_counts.get(
        1,
        0
    )

    / target_counts.sum()

    * 100

)


summary.append(

    f"폐업 비율: "
    f"{closure_rate:.2f}%"

)


summary.append("")


# ------------------------------------------------------------
# Model
# ------------------------------------------------------------

summary.append(
    "[PR-AUC 기준 상위 모델]"
)


summary.append(

    f"모델: "
    f"{best_model_name}"

)


summary.append(

    f"불균형 처리: "
    f"{best_method}"

)


summary.append("")


# ------------------------------------------------------------
# Metrics
# ------------------------------------------------------------

summary.append(
    "[모델 성능]"
)


summary.append(

    f"PR-AUC: "
    f"{best_row['PR_AUC']:.3f}"

)


summary.append(

    f"ROC-AUC: "
    f"{best_row['ROC_AUC']:.3f}"

)


summary.append(

    f"폐업 Recall: "
    f"{best_row['Recall_Closure']:.3f}"

)


summary.append(

    f"폐업 Precision: "
    f"{best_row['Precision_Closure']:.3f}"

)


summary.append(

    f"폐업 F1: "
    f"{best_row['F1_Closure']:.3f}"

)


summary.append(

    f"Balanced Accuracy: "
    f"{best_row['Balanced_Accuracy']:.3f}"

)


# ------------------------------------------------------------
# SHAP
# ------------------------------------------------------------

if (
    SHAP_SUCCESS
    and
    shap_importance
    is not None
):

    summary.append("")

    summary.append(
        "[폐업 예측 주요 변수]"
    )


    for index, row in (

        shap_importance
        .head(10)
        .iterrows()

    ):

        summary.append(

            f"{index + 1}. "
            f"{row['변수']}"

        )


# ============================================================
# 38. Summary 저장
# ============================================================

summary_text = (
    "\n".join(
        summary
    )
)


SUMMARY_PATH = (

    OUTPUT_DIR
    / "summary.txt"

)


with open(

    SUMMARY_PATH,

    "w",

    encoding="utf-8",

) as file:

    file.write(
        summary_text
    )


# ============================================================
# 39. 최종 출력
# ============================================================

print("\n")
print(summary_text)


print("\n")
print("=" * 70)

print(
    "모델링 완료"
)

print("=" * 70)


print(
    f"\n결과 폴더:\n"
    f"{OUTPUT_DIR.resolve()}"
)


print(
    "\n주요 결과:"
)


print(
    "1. metrics/model_comparison.csv"
)


print(
    "2. figures/*.png"
)


print(
    "3. importance/*.csv"
)


print(
    "4. importance/*.png"
)


print(
    "5. shap/best_model_shap_summary.png"
)


print(
    "6. shap/best_model_shap_importance.csv"
)


print(
    "7. summary.txt"
)

# ============================================================
# Markdown 결과 보고서 생성
# ============================================================

REPORT_PATH = OUTPUT_DIR / "model_report.md"


def make_markdown_table(df):
    """
    pandas DataFrame을 Markdown Table로 변환
    tabulate 설치 없이 동작
    """

    df = df.copy()

    header = "| " + " | ".join(df.columns.astype(str)) + " |"
    separator = "| " + " | ".join(["---"] * len(df.columns)) + " |"

    rows = []

    for _, row in df.iterrows():

        values = []

        for value in row:

            if isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value))

        rows.append(
            "| " + " | ".join(values) + " |"
        )

    return "\n".join(
        [header, separator] + rows
    )


# ============================================================
# 1. 성능 결과 정리
# ============================================================

performance_table = results_df[
    [
        "Model",
        "Imbalance",
        "PR_AUC",
        "ROC_AUC",
        "Precision_Closure",
        "Recall_Closure",
        "F1_Closure",
        "Balanced_Accuracy",
    ]
].copy()


performance_table.columns = [
    "모델",
    "불균형 처리",
    "PR-AUC",
    "ROC-AUC",
    "폐업 Precision",
    "폐업 Recall",
    "폐업 F1",
    "Balanced Accuracy",
]


# ============================================================
# 2. Feature Importance
# ============================================================

if hasattr(best_model, "feature_importances_"):

    feature_importance_report = pd.DataFrame(
        {
            "변수": feature_names,
            "Feature Importance":
                best_model.feature_importances_,
        }
    )

    feature_importance_report = (
        feature_importance_report
        .sort_values(
            "Feature Importance",
            ascending=False
        )
        .head(20)
        .reset_index(drop=True)
    )

    feature_importance_report.insert(
        0,
        "순위",
        range(
            1,
            len(feature_importance_report) + 1
        )
    )

else:

    feature_importance_report = None


# ============================================================
# 3. SHAP Importance
# ============================================================

if SHAP_SUCCESS and shap_importance is not None:

    shap_report = (
        shap_importance
        .head(20)
        .copy()
        .reset_index(drop=True)
    )

    shap_report.insert(
        0,
        "순위",
        range(
            1,
            len(shap_report) + 1
        )
    )

else:

    shap_report = None


# ============================================================
# 4. 데이터 기본 정보
# ============================================================

survival_count = int(
    target_counts.get(0, 0)
)

closure_count = int(
    target_counts.get(1, 0)
)

total_count = (
    survival_count
    + closure_count
)

survival_rate = (
    survival_count
    / total_count
    * 100
)

closure_rate = (
    closure_count
    / total_count
    * 100
)


data_summary = pd.DataFrame(
    {
        "구분": [
            "전체",
            "생존",
            "폐업",
        ],

        "사업체 수": [
            total_count,
            survival_count,
            closure_count,
        ],

        "비율": [
            "100%",
            f"{survival_rate:.2f}%",
            f"{closure_rate:.2f}%",
        ],
    }
)


# ============================================================
# 5. Markdown 작성
# ============================================================

report = []


report.append(
    f"# {TARGET_HORIZON} 사업체 폐업 예측 모델 결과"
)

report.append("")


# ------------------------------------------------------------
# 데이터
# ------------------------------------------------------------

report.append(
    "## 1. 데이터"
)

report.append("")

report.append(
    make_markdown_table(
        data_summary
    )
)

report.append("")

report.append(
    f"- 예측 대상: 위기 노출 이후 "
    f"**{TARGET_HORIZON} 이내 폐업 여부**"
)

report.append(
    "- 모델링 기준: **폐업 = 1 / 생존 = 0**"
)

report.append("")


# ------------------------------------------------------------
# 성능
# ------------------------------------------------------------

report.append(
    "## 2. 모델 성능"
)

report.append("")

report.append(
    make_markdown_table(
        performance_table
    )
)

report.append("")


# ------------------------------------------------------------
# 상위 모델
# ------------------------------------------------------------

report.append(
    "### PR-AUC 기준 상위 실험"
)

report.append("")


best_summary = pd.DataFrame(
    {
        "항목": [
            "모델",
            "불균형 처리",
            "PR-AUC",
            "ROC-AUC",
            "폐업 Precision",
            "폐업 Recall",
            "폐업 F1",
        ],

        "결과": [
            best_model_name,
            best_method,
            f"{best_row['PR_AUC']:.4f}",
            f"{best_row['ROC_AUC']:.4f}",
            f"{best_row['Precision_Closure']:.4f}",
            f"{best_row['Recall_Closure']:.4f}",
            f"{best_row['F1_Closure']:.4f}",
        ],
    }
)


report.append(
    make_markdown_table(
        best_summary
    )
)

report.append("")


# ------------------------------------------------------------
# Feature Importance
# ------------------------------------------------------------

report.append(
    "## 3. Feature Importance"
)

report.append("")

report.append(
    f"**{best_model_name} + "
    f"{best_method}** 모델 기준입니다."
)

report.append("")


if feature_importance_report is not None:

    report.append(
        make_markdown_table(
            feature_importance_report
        )
    )

else:

    report.append(
        "Feature Importance를 "
        "계산할 수 없습니다."
    )


report.append("")


# ------------------------------------------------------------
# SHAP
# ------------------------------------------------------------

report.append(
    "## 4. SHAP"
)

report.append("")

report.append(
    "평균 절대 SHAP 값이 클수록 "
    "모델 예측에 미치는 영향이 큰 변수입니다."
)

report.append("")


if shap_report is not None:

    report.append(
        make_markdown_table(
            shap_report
        )
    )

else:

    report.append(
        "SHAP 결과를 생성하지 못했습니다."
    )


report.append("")


# ------------------------------------------------------------
# 그래프
# ------------------------------------------------------------

report.append(
    "### SHAP 그래프"
)

report.append("")

report.append(
    "![SHAP Summary]"
    "(shap/best_model_shap_summary.png)"
)

report.append("")

report.append(
    "![SHAP Importance]"
    "(shap/best_model_shap_bar.png)"
)

report.append("")


# ------------------------------------------------------------
# 간단한 지표 설명
# ------------------------------------------------------------

report.append(
    "## 5. 지표 해석"
)

report.append("")


metric_description = pd.DataFrame(
    {
        "지표": [
            "PR-AUC",
            "ROC-AUC",
            "폐업 Precision",
            "폐업 Recall",
            "폐업 F1",
            "Feature Importance",
            "SHAP",
        ],

        "의미": [
            "불균형 데이터에서 폐업 사업자를 얼마나 잘 구분하는지 평가",
            "생존과 폐업을 전반적으로 얼마나 잘 구분하는지 평가",
            "폐업이라고 예측한 사업자 중 실제 폐업한 비율",
            "실제 폐업 사업자 중 모델이 찾아낸 비율",
            "폐업 Precision과 Recall의 균형",
            "모델이 예측할 때 많이 활용한 변수",
            "각 변수가 개별 예측에 얼마나 영향을 미쳤는지 설명",
        ],
    }
)


report.append(
    make_markdown_table(
        metric_description
    )
)


# ============================================================
# 6. 저장
# ============================================================

REPORT_PATH.write_text(
    "\n".join(report),
    encoding="utf-8",
)


print("\n" + "=" * 70)

print(
    "Markdown 보고서 생성 완료"
)

print("=" * 70)

print(
    REPORT_PATH.resolve()
)