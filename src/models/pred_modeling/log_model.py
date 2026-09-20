# ============================================================
# logistic_analysis.py
#
# 전체 변수 기반 Logistic Regression
# - Tree 결과와 완전히 독립
# - 12M / 18M
# - 불균형 처리 방법 실행 시 직접 선택
# - Expanding Window Validation
# - 2024Q2 Final Test
# - 전체 결과 CSV / Excel 저장
# ============================================================

from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import joblib

from tqdm.auto import tqdm

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

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

warnings.filterwarnings("ignore")


# ============================================================
# 1. CONFIG
# ============================================================

RANDOM_STATE = 42

PROJECT_ROOT = Path(
    r"C:\Users\DC\2026\IMBANK\iM_Blockchain_AI"
)

# ★ 실제 데이터 경로로 수정
DATA_PATH = (
    PROJECT_ROOT /
    "data/processed/variable/dmdrlt5.csv"
)

RESULT_DIR = (
    PROJECT_ROOT /
    "logistic_model_results"
)

RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 2. DATA LOAD
# ============================================================

print("=" * 80)
print("DATA LOAD")
print("=" * 80)

df = pd.read_csv(
    DATA_PATH,
    low_memory=False
)

print("Shape:", df.shape)


# ============================================================
# 3. 기본 정리
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
# 4. TARGET
# ============================================================

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


# ============================================================
# 5. Validation 설정
# ============================================================

VALIDATION_QUARTERS = list(
    pd.period_range(
        start="2020Q1",
        end="2024Q1",
        freq="Q"
    )
)

FINAL_TEST_QUARTER = "2024Q2"


# ============================================================
# 6. 모델에서 제외할 변수
# ============================================================

DROP_COLS = [

    # ID
    "관리번호",

    # 시간 / 관리
    "기준분기",
    "t0",
    "_quarter_period",

    # Cohort 선정용
    "crisis_flag",
    "episode_start",

    # 행정 코드
    "행정동코드",

    # --------------------------------------------------------
    # Label 관련 - 절대 X에 포함 금지
    # --------------------------------------------------------

    "horizon_end_12M",
    "Y_12M",
    "label_status_12M",

    "horizon_end_18M",
    "Y_18M",
    "label_status_18M",

    "horizon_end_24M",
    "Y_24M",
    "label_status_24M",

    # --------------------------------------------------------
    # 업력 변수 중복
    #
    # 업력_일수 / 개월 / 연수는 사실상 같은 정보이므로
    # 업력_연수만 사용
    # --------------------------------------------------------

    "업력_일수",
    "업력_개월수",

    # 매핑용
    "서비스인구_매핑동",
]


# 실제 존재하는 컬럼만 제거
DROP_COLS = [
    c for c in DROP_COLS
    if c in df.columns
]


# ============================================================
# 7. 전체 Feature
# ============================================================

FEATURE_COLS = [
    c for c in df.columns
    if c not in DROP_COLS
]


print()
print("=" * 80)
print("LOGISTIC 전체 변수")
print("=" * 80)

print(
    "Feature 수:",
    len(FEATURE_COLS)
)

for i, col in enumerate(
    FEATURE_COLS,
    start=1
):
    print(
        f"{i:03d}. {col}"
    )


# ============================================================
# 8. Numeric / Categorical
# ============================================================

NUMERIC_COLS = []
CATEGORICAL_COLS = []


for col in FEATURE_COLS:

    if (
        pd.api.types.is_object_dtype(df[col])
        or
        pd.api.types.is_bool_dtype(df[col])
        or
        isinstance(
            df[col].dtype,
            pd.CategoricalDtype
        )
    ):

        CATEGORICAL_COLS.append(
            col
        )

    else:

        NUMERIC_COLS.append(
            col
        )


print()
print(
    "Numeric:",
    len(NUMERIC_COLS)
)

print(
    "Categorical:",
    len(CATEGORICAL_COLS)
)

print(
    "Categorical Variables:",
    CATEGORICAL_COLS
)


# ============================================================
# 9. 불균형 처리 방법 직접 선택
# ============================================================

IMBALANCE_OPTIONS = {

    "1": "None",
    "2": "ClassWeight",
    "3": "RandomUnder",
    "4": "RandomOver",
    "5": "SMOTE"
}


def select_imbalance_method(
    horizon
):

    print()
    print("=" * 80)
    print(
        f"{horizon} 불균형 처리 방법"
    )
    print("=" * 80)

    print("1. None")
    print("2. ClassWeight")
    print("3. RandomUnder")
    print("4. RandomOver")
    print("5. SMOTE")

    while True:

        choice = input(
            f"\n{horizon} 선택 [1~5]: "
        ).strip()

        if choice in IMBALANCE_OPTIONS:

            method = (
                IMBALANCE_OPTIONS[
                    choice
                ]
            )

            print(
                f"→ {horizon}: "
                f"{method}"
            )

            return method

        print(
            "1~5 중 하나를 입력하세요."
        )


# ============================================================
# 10. 12M / 18M 각각 선택
# ============================================================

SELECTED_IMBALANCE = {}


for horizon in [
    "12M",
    "18M"
]:

    SELECTED_IMBALANCE[
        horizon
    ] = select_imbalance_method(
        horizon
    )


print()
print("=" * 80)
print("분석 설정")
print("=" * 80)

print(
    "12M:",
    SELECTED_IMBALANCE["12M"]
)

print(
    "18M:",
    SELECTED_IMBALANCE["18M"]
)


# ============================================================
# 11. Preprocessor
# ============================================================

def make_preprocessor():

    # --------------------------------------------------------
    # Numeric
    # --------------------------------------------------------

    numeric_pipeline = Pipeline(

        steps=[

            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                )
            ),

            (
                "scaler",
                StandardScaler()
            )
        ]
    )


    # --------------------------------------------------------
    # Categorical
    # --------------------------------------------------------

    categorical_pipeline = Pipeline(

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
                    drop="first",
                    sparse_output=False
                )
            )
        ]
    )


    transformers = []


    if NUMERIC_COLS:

        transformers.append(

            (
                "num",
                numeric_pipeline,
                NUMERIC_COLS
            )
        )


    if CATEGORICAL_COLS:

        transformers.append(

            (
                "cat",
                categorical_pipeline,
                CATEGORICAL_COLS
            )
        )


    return ColumnTransformer(

        transformers=transformers,

        remainder="drop",

        verbose_feature_names_out=False
    )


# ============================================================
# 12. Logistic Pipeline
# ============================================================

def build_pipeline(
    imbalance_method,
    y_train
):

    preprocessor = (
        make_preprocessor()
    )


    # --------------------------------------------------------
    # 기본 Logistic Regression
    # --------------------------------------------------------
    #
    # sklearn 기본 설정에 가깝게 사용
    #
    # L2
    # C = 1
    # lbfgs
    #
    # max_iter만 수렴 안정성을 위해 증가
    # --------------------------------------------------------

    model_kwargs = {

        "penalty": "l2",

        "C": 1.0,

        "solver": "lbfgs",

        "max_iter": 3000,

        "random_state":
            RANDOM_STATE
    }


    # --------------------------------------------------------
    # Class Weight
    # --------------------------------------------------------

    if imbalance_method == "ClassWeight":

        model_kwargs[
            "class_weight"
        ] = "balanced"


    model = LogisticRegression(
        **model_kwargs
    )


    # --------------------------------------------------------
    # Sampling 없음
    # --------------------------------------------------------

    if imbalance_method in [
        "None",
        "ClassWeight"
    ]:

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
    # Random Under
    # --------------------------------------------------------

    if imbalance_method == "RandomUnder":

        sampler = RandomUnderSampler(
            random_state=RANDOM_STATE
        )


    # --------------------------------------------------------
    # Random Over
    # --------------------------------------------------------

    elif imbalance_method == "RandomOver":

        sampler = RandomOverSampler(
            random_state=RANDOM_STATE
        )


    # --------------------------------------------------------
    # SMOTE
    # --------------------------------------------------------

    elif imbalance_method == "SMOTE":

        counts = (
            pd.Series(y_train)
            .value_counts()
        )

        minority_n = (
            counts.min()
        )


        if minority_n <= 5:

            print(
                "SMOTE 표본 부족 → "
                "RandomOver 사용"
            )

            sampler = (
                RandomOverSampler(
                    random_state=
                        RANDOM_STATE
                )
            )

        else:

            sampler = SMOTE(

                random_state=
                    RANDOM_STATE,

                k_neighbors=min(
                    5,
                    minority_n - 1
                )
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
# 13. 평가 함수
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

        result[
            "ROC_AUC"
        ] = roc_auc_score(
            y_true,
            y_prob
        )

        result[
            "PR_AUC"
        ] = average_precision_score(
            y_true,
            y_prob
        )

    else:

        result["ROC_AUC"] = np.nan
        result["PR_AUC"] = np.nan


    # --------------------------------------------------------
    # 기본 Metrics
    # --------------------------------------------------------

    result[
        "Accuracy"
    ] = accuracy_score(
        y_true,
        y_pred
    )


    result[
        "Balanced_Accuracy"
    ] = balanced_accuracy_score(
        y_true,
        y_pred
    )


    result[
        "Precision"
    ] = precision_score(
        y_true,
        y_pred,
        zero_division=0
    )


    result[
        "Recall"
    ] = recall_score(
        y_true,
        y_pred,
        zero_division=0
    )


    result[
        "F1"
    ] = f1_score(
        y_true,
        y_pred,
        zero_division=0
    )


    result[
        "Brier"
    ] = brier_score_loss(
        y_true,
        y_prob
    )


    # --------------------------------------------------------
    # Confusion Matrix
    # --------------------------------------------------------

    tn, fp, fn, tp = (
        confusion_matrix(
            y_true,
            y_pred,
            labels=[0, 1]
        )
        .ravel()
    )


    result["TN"] = tn
    result["FP"] = fp
    result["FN"] = fn
    result["TP"] = tp


    result[
        "Specificity"
    ] = (

        tn / (tn + fp)

        if (tn + fp) > 0

        else np.nan
    )


    result[
        "N"
    ] = len(y_true)


    result[
        "Survival_Rate"
    ] = np.mean(
        y_true
    )


    return result


# ============================================================
# 14. Expanding Window Validation
# ============================================================

def run_validation(
    data,
    horizon,
    target_col,
    status_col,
    imbalance_method
):

    print()
    print("=" * 80)
    print(
        horizon,
        "LOGISTIC"
    )
    print(
        "Imbalance:",
        imbalance_method
    )
    print("=" * 80)


    # --------------------------------------------------------
    # 관측된 Label만
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


    results = []


    # --------------------------------------------------------
    # tqdm
    # --------------------------------------------------------

    pbar = tqdm(

        VALIDATION_QUARTERS,

        desc=(
            f"{horizon} Logistic "
            f"[{imbalance_method}]"
        ),

        unit="fold",

        dynamic_ncols=True
    )


    # ========================================================
    # FOLD
    # ========================================================

    for fold_no, valid_q in enumerate(
        pbar,
        start=1
    ):


        train_df = d[

            d["_quarter_period"]
            < valid_q

        ].copy()


        valid_df = d[

            d["_quarter_period"]
            == valid_q

        ].copy()


        pbar.set_postfix({

            "valid":
                str(valid_q),

            "train":
                len(train_df),

            "valid_n":
                len(valid_df)
        })


        if (
            len(train_df) == 0
            or
            len(valid_df) == 0
        ):

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


        if y_train.nunique() < 2:

            continue


        try:

            pipeline = build_pipeline(

                imbalance_method=
                    imbalance_method,

                y_train=y_train
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


            metrics = evaluate_model(
                y_valid,
                y_prob
            )


            metrics.update({

                "Horizon":
                    horizon,

                "Model":
                    "LogisticRegression",

                "Imbalance":
                    imbalance_method,

                "Fold":
                    fold_no,

                "Train_End":
                    str(
                        valid_q - 1
                    ),

                "Valid_Quarter":
                    str(valid_q),

                "Train_N":
                    len(train_df),

                "Valid_N":
                    len(valid_df),

                "Train_Y0":
                    int(
                        (y_train == 0).sum()
                    ),

                "Train_Y1":
                    int(
                        (y_train == 1).sum()
                    ),

                "Valid_Y0":
                    int(
                        (y_valid == 0).sum()
                    ),

                "Valid_Y1":
                    int(
                        (y_valid == 1).sum()
                    ),

                "N_Features":
                    len(FEATURE_COLS)
            })


            results.append(
                metrics
            )


        except Exception as e:

            tqdm.write(

                f"ERROR | "
                f"{horizon} | "
                f"{valid_q} | "
                f"{e}"
            )


    return pd.DataFrame(
        results
    )


# ============================================================
# 15. 12M + 18M 실행
# ============================================================

all_validation = []


for horizon, config in (
    TARGET_CONFIG.items()
):

    result = run_validation(

        data=df,

        horizon=horizon,

        target_col=
            config["target"],

        status_col=
            config["status"],

        imbalance_method=
            SELECTED_IMBALANCE[
                horizon
            ]
    )


    all_validation.append(
        result
    )


validation_results = pd.concat(

    all_validation,

    ignore_index=True
)


# ============================================================
# 16. 평균 성능
# ============================================================

METRIC_COLS = [

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

    validation_results

    .groupby(
        [
            "Horizon",
            "Model",
            "Imbalance"
        ]
    )[METRIC_COLS]

    .mean()

    .reset_index()
)


performance_std = (

    validation_results

    .groupby(
        [
            "Horizon",
            "Model",
            "Imbalance"
        ]
    )[METRIC_COLS]

    .std()

    .reset_index()
)


performance_std = (
    performance_std.rename(
        columns={
            c: f"{c}_STD"
            for c in METRIC_COLS
        }
    )
)


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
)


# ============================================================
# 17. Final Test
# ============================================================

final_results = []

coefficient_results = []

prediction_results = []


for horizon, config in (
    TARGET_CONFIG.items()
):

    target_col = (
        config["target"]
    )

    status_col = (
        config["status"]
    )

    imbalance_method = (
        SELECTED_IMBALANCE[
            horizon
        ]
    )


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
    print("=" * 80)

    print(
        "Imbalance:",
        imbalance_method
    )

    print(
        "Features:",
        len(FEATURE_COLS)
    )

    print(
        "Train:",
        len(train_df)
    )

    print(
        "Test:",
        len(test_df)
    )


    if len(test_df) == 0:

        print(
            "Observed Final Test 없음"
        )

        continue


    X_train = train_df[
        FEATURE_COLS
    ]

    y_train = train_df[
        target_col
    ]


    X_test = test_df[
        FEATURE_COLS
    ]

    y_test = test_df[
        target_col
    ]


    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    pipeline = build_pipeline(

        imbalance_method=
            imbalance_method,

        y_train=y_train
    )


    pipeline.fit(
        X_train,
        y_train
    )


    # --------------------------------------------------------
    # Predict
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
            "LogisticRegression",

        "Imbalance":
            imbalance_method,

        "Test_Quarter":
            FINAL_TEST_QUARTER,

        "Train_N":
            len(train_df),

        "Test_N":
            len(test_df),

        "N_Features":
            len(FEATURE_COLS)
    })


    final_results.append(
        metrics
    )


    # ========================================================
    # Coefficients
    # ========================================================

    preprocessor = (
        pipeline
        .named_steps[
            "preprocessor"
        ]
    )

    model = (
        pipeline
        .named_steps[
            "model"
        ]
    )


    feature_names = (
        preprocessor
        .get_feature_names_out()
    )


    coefficients = (
        model.coef_[0]
    )


    coef_df = pd.DataFrame({

        "Horizon":
            horizon,

        "Feature":
            feature_names,

        "Coefficient":
            coefficients,

        "Abs_Coefficient":
            np.abs(
                coefficients
            ),

        "Odds_Ratio":
            np.exp(
                np.clip(
                    coefficients,
                    -20,
                    20
                )
            )
    })


    coef_df[
        "Direction"
    ] = np.where(

        coef_df[
            "Coefficient"
        ] > 0,

        "생존 증가",

        np.where(

            coef_df[
                "Coefficient"
            ] < 0,

            "생존 감소",

            "중립"
        )
    )


    coef_df = (

        coef_df

        .sort_values(
            "Abs_Coefficient",
            ascending=False
        )

        .reset_index(drop=True)
    )


    coef_df[
        "Rank"
    ] = np.arange(
        1,
        len(coef_df) + 1
    )


    coefficient_results.append(
        coef_df
    )


    # ========================================================
    # Predictions
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


    prediction_results.append(
        pred_df
    )


    # --------------------------------------------------------
    # Model 저장
    # --------------------------------------------------------

    joblib.dump(

        pipeline,

        RESULT_DIR /
        f"{horizon}_LOGISTIC_MODEL.joblib"
    )


# ============================================================
# 18. 결과 통합
# ============================================================

final_performance = pd.DataFrame(
    final_results
)


if coefficient_results:

    coefficient_df = pd.concat(
        coefficient_results,
        ignore_index=True
    )

else:

    coefficient_df = (
        pd.DataFrame()
    )


if prediction_results:

    prediction_df = pd.concat(
        prediction_results,
        ignore_index=True
    )

else:

    prediction_df = (
        pd.DataFrame()
    )


# ============================================================
# 19. 저장
# ============================================================

validation_results.to_csv(

    RESULT_DIR /
    "01_validation_folds.csv",

    index=False,

    encoding="utf-8-sig"
)


performance_summary.to_csv(

    RESULT_DIR /
    "02_performance_summary.csv",

    index=False,

    encoding="utf-8-sig"
)


final_performance.to_csv(

    RESULT_DIR /
    "03_final_test.csv",

    index=False,

    encoding="utf-8-sig"
)


coefficient_df.to_csv(

    RESULT_DIR /
    "04_coefficients.csv",

    index=False,

    encoding="utf-8-sig"
)


prediction_df.to_csv(

    RESULT_DIR /
    "05_predictions.csv",

    index=False,

    encoding="utf-8-sig"
)


# ============================================================
# 20. Excel
# ============================================================

EXCEL_PATH = (

    RESULT_DIR /
    "LOGISTIC_ANALYSIS.xlsx"
)


with pd.ExcelWriter(

    EXCEL_PATH,

    engine="openpyxl"

) as writer:


    performance_summary.to_excel(

        writer,

        sheet_name=
            "01_Performance",

        index=False
    )


    final_performance.to_excel(

        writer,

        sheet_name=
            "02_Final_Test",

        index=False
    )


    coefficient_df.to_excel(

        writer,

        sheet_name=
            "03_Coefficients",

        index=False
    )


    validation_results.to_excel(

        writer,

        sheet_name=
            "04_Fold_Detail",

        index=False
    )


    prediction_df.to_excel(

        writer,

        sheet_name=
            "05_Predictions",

        index=False
    )


# ============================================================
# 21. 출력
# ============================================================

print()
print("=" * 80)
print("LOGISTIC REGRESSION 완료")
print("=" * 80)

print()
print("[Expanding Window 평균 성능]")
print()

print(
    performance_summary.to_string(
        index=False
    )
)


print()
print("[2024Q2 Final Test]")
print()

print(
    final_performance.to_string(
        index=False
    )
)


print()
print("[Coefficient TOP 30]")
print()

if len(coefficient_df) > 0:

    print(
        coefficient_df[
            coefficient_df[
                "Rank"
            ] <= 30
        ].to_string(
            index=False
        )
    )


print()
print(
    "저장 폴더:",
    RESULT_DIR.resolve()
)

print(
    "Excel:",
    EXCEL_PATH.resolve()
)
# ============================================================
# 22. GPT API 기반 Logistic 최종 결과 해석
# ============================================================

import os
from openai import OpenAI
from dotenv import load_dotenv


# ============================================================
# 22-1. OpenAI API
# ============================================================

ENV_PATH = PROJECT_ROOT / ".env"

load_dotenv(
    dotenv_path=ENV_PATH,
    override=True
)

client = OpenAI(
    api_key=os.environ["OPENAI_API_KEY"]
)


# ============================================================
# 22-2. GPT 전달용 성능 결과
# ============================================================

performance_for_gpt = (
    performance_summary
    .round(4)
)

final_for_gpt = (
    final_performance
    .round(4)
)


# ============================================================
# 22-3. Fold별 결과
# ============================================================

fold_cols = [
    "Horizon",
    "Imbalance",
    "Fold",
    "Valid_Quarter",
    "Train_N",
    "Valid_N",
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

fold_cols = [
    c for c in fold_cols
    if c in validation_results.columns
]

fold_for_gpt = (
    validation_results[
        fold_cols
    ]
    .round(4)
)


# ============================================================
# 22-4. Logistic 계수 TOP 변수
# ============================================================

# Horizon별 절대계수 TOP 30
coef_top_for_gpt = (

    coefficient_df

    .sort_values(
        [
            "Horizon",
            "Abs_Coefficient"
        ],
        ascending=[
            True,
            False
        ]
    )

    .groupby(
        "Horizon",
        group_keys=False
    )

    .head(30)

    .copy()
)


coef_top_for_gpt = (
    coef_top_for_gpt
    .round(4)
)


# ============================================================
# 22-5. 양의 계수 TOP 15
# ============================================================

positive_coef = (

    coefficient_df[
        coefficient_df[
            "Coefficient"
        ] > 0
    ]

    .sort_values(
        [
            "Horizon",
            "Coefficient"
        ],
        ascending=[
            True,
            False
        ]
    )

    .groupby(
        "Horizon",
        group_keys=False
    )

    .head(15)

    .copy()
    .round(4)
)


# ============================================================
# 22-6. 음의 계수 TOP 15
# ============================================================

negative_coef = (

    coefficient_df[
        coefficient_df[
            "Coefficient"
        ] < 0
    ]

    .sort_values(
        [
            "Horizon",
            "Coefficient"
        ],
        ascending=[
            True,
            True
        ]
    )

    .groupby(
        "Horizon",
        group_keys=False
    )

    .head(15)

    .copy()
    .round(4)
)


# ============================================================
# 22-7. Markdown 변환
# ============================================================

performance_md = (
    performance_for_gpt
    .to_markdown(index=False)
)

final_md = (
    final_for_gpt
    .to_markdown(index=False)
)

fold_md = (
    fold_for_gpt
    .to_markdown(index=False)
)

coef_top_md = (
    coef_top_for_gpt
    .to_markdown(index=False)
)

positive_md = (
    positive_coef
    .to_markdown(index=False)
)

negative_md = (
    negative_coef
    .to_markdown(index=False)
)


# ============================================================
# 22-8. Prompt
# ============================================================

prompt = f"""
당신은 머신러닝 및 통계 분석 결과를 해석하는 연구 분석가이다.

현재 연구는 위기 상황에 노출된 사업자의
향후 영업 지속 여부를 예측하는 연구이다.

이번 분석은 Tree 모델과 별개로 수행한
독립적인 Logistic Regression baseline 분석이다.

Tree 기반 Feature Selection 결과를 사용하지 않았으며,
사전에 정의된 전체 설명변수를 Logistic Regression에 사용하였다.


============================================================
[종속변수]
============================================================

12M:
- Y=1: t0 이후 12개월까지 영업 지속
- Y=0: 12개월 이내 폐업

18M:
- Y=1: t0 이후 18개월까지 영업 지속
- Y=0: 18개월 이내 폐업

따라서 현재 Logistic Regression의 positive class는
'생존(Y=1)'이다.

Precision, Recall, F1 및 PR-AUC는 이 정의를 기준으로
해석해야 한다.


============================================================
[모델]
============================================================

Logistic Regression

기본 설정:
- penalty = L2
- C = 1.0
- solver = lbfgs

수치형 변수:
- Median Imputation
- StandardScaler

범주형 변수:
- Most Frequent Imputation
- One-Hot Encoding
- 첫 범주는 reference category로 제거


============================================================
[검증 방법]
============================================================

Expanding Window Validation을 사용하였다.

예:

2019Q1~2019Q4 → 2020Q1 검증
2019Q1~2020Q1 → 2020Q2 검증
2019Q1~2020Q2 → 2020Q3 검증
...
→ 2024Q1 검증

2024Q2는 모델 선택에 사용하지 않고
Final Test로 별도 평가하였다.


============================================================
[불균형 처리]
============================================================

12M:
{SELECTED_IMBALANCE["12M"]}

18M:
{SELECTED_IMBALANCE["18M"]}

불균형 처리 방법은 본 Logistic 분석 내부에서
여러 방법을 비교하여 선택한 것이 아니라,
연구자가 사전에 지정한 방법이다.

따라서 해당 방법이 Logistic에서 가장 좋은 방법이라고
단정해서는 안 된다.


============================================================
[1. Expanding Window 평균 성능]
============================================================

{performance_md}


============================================================
[2. Fold별 성능]
============================================================

{fold_md}


============================================================
[3. 2024Q2 Final Test]
============================================================

{final_md}


============================================================
[4. 절대계수 기준 주요 변수]
============================================================

{coef_top_md}


============================================================
[5. 양의 계수 주요 변수]
============================================================

{positive_md}


============================================================
[6. 음의 계수 주요 변수]
============================================================

{negative_md}


위 결과를 바탕으로 연구 보고서용 분석 결과를 작성하라.

반드시 제공된 데이터만 근거로 해석하고,
존재하지 않는 숫자나 결과를 만들어내지 말 것.


# Logistic Regression 분석 결과


## 1. 분석 개요

다음 내용을 간략히 설명한다.

- 연구 목적
- 12M / 18M 종속변수
- Logistic Regression 사용 목적
- 전체 설명변수를 사용했다는 점
- Expanding Window Validation
- 2024Q2 Final Test


## 2. 12개월 생존 예측 성능

12M의 주요 성능을 표로 정리한다.

다음 지표를 중심으로 분석한다.

- ROC-AUC
- PR-AUC
- Accuracy
- Balanced Accuracy
- Precision
- Recall
- Specificity
- F1
- Brier Score

Accuracy 하나만으로 성능을 판단하지 않는다.

불균형 데이터라는 점을 고려하여
Balanced Accuracy, PR-AUC, Recall 등을 함께 해석한다.


## 3. 18개월 생존 예측 성능

18M에 대해서도 동일하게 분석한다.


## 4. 시간에 따른 성능 안정성

Fold별 결과를 이용하여
2020Q1~2024Q1 동안 성능이 얼마나 안정적인지 분석한다.

특정 분기에서 성능이 크게 상승하거나 하락한 경우
실제 숫자를 근거로 언급한다.

다만 원인을 데이터에서 확인할 수 없다면
원인을 단정하지 않는다.


## 5. 2024Q2 Final Test

Validation 결과와 2024Q2 Final Test 결과를 비교한다.

다음을 구분한다.

- Validation 평균 성능
- Final Test 성능

Final Test가 Validation보다 높거나 낮은 경우
그 차이를 설명한다.

단, Final Test는 모델 선택에 사용되지 않은
외부 시점 평가라는 점을 명시한다.


## 6. Logistic 주요 변수 분석

Coefficient의 절대값이 큰 변수를 중심으로 분석한다.

수치형 변수는 StandardScaler를 적용했기 때문에
표준화된 변수 기준의 coefficient라는 점을 설명한다.

단, 범주형 변수의 coefficient는
reference category 대비 효과라는 점을 구분한다.


## 7. 생존과 양의 관계를 보이는 변수

Coefficient > 0인 주요 변수를 표로 정리한다.

표:

| 변수 | Coefficient | Odds Ratio | 해석 |

Y=1이 생존이므로,
양의 coefficient는 다른 조건이 동일할 때
생존 log-odds 증가와 연결됨을 설명한다.

인과관계로 표현하지 않는다.


## 8. 생존과 음의 관계를 보이는 변수

Coefficient < 0인 주요 변수를 표로 정리한다.

표:

| 변수 | Coefficient | Odds Ratio | 해석 |

음의 coefficient는 생존 log-odds 감소와 연결되며,
반대로 폐업 위험과의 연관 가능성을 의미한다.

그러나 인과효과라고 단정하지 않는다.


## 9. 12M과 18M 비교

다음을 비교한다.

- 예측 성능 차이
- 주요 변수 차이
- coefficient 방향 차이
- 두 Horizon에서 공통적으로 나타나는 변수

같은 변수가 12M과 18M에서
같은 방향을 가지는지 확인한다.


## 10. 결과 해석 시 주의사항

다음 사항을 반드시 언급한다.

1. Logistic coefficient는 인과효과가 아니다.

2. 상관성이 높은 설명변수가 여러 개 존재할 경우
   coefficient 크기와 방향이 불안정할 수 있다.

3. 전체 변수를 사용했으므로 다중공선성 가능성이 있다.

4. Odds Ratio는 다른 변수가 동일하다는 조건에서의
   조건부 관계로 해석해야 한다.

5. 현재 불균형 처리 방법은 Logistic 내부에서
   최적화한 것이 아니다.


## 11. 최종 요약

마지막에 연구자가 바로 사용할 수 있도록 표로 정리한다.

| 구분 | 12M | 18M |
|---|---|---|
| 불균형 처리 | | |
| Validation ROC-AUC | | |
| Validation PR-AUC | | |
| Validation Balanced Accuracy | | |
| Validation F1 | | |
| Final ROC-AUC | | |
| Final PR-AUC | | |
| Final Balanced Accuracy | | |
| 주요 양의 변수 | | |
| 주요 음의 변수 | | |

마지막으로 3~5문장으로 전체 결과를 요약한다.

중요:
특정 변수가 '폐업을 유발한다',
'생존을 증가시킨다'처럼 인과적으로 표현하지 말고,
'생존 가능성과 양의 관계',
'생존 가능성과 음의 관계'와 같이 표현한다.
"""


# ============================================================
# 22-9. API 호출
# ============================================================

print()
print("=" * 80)
print("GPT Logistic 결과 분석 시작")
print("=" * 80)


response = client.responses.create(

    model="gpt-5.6",

    input=prompt
)


report = response.output_text


# ============================================================
# 22-10. Markdown 저장
# ============================================================

REPORT_PATH = (
    RESULT_DIR /
    "FINAL_LOGISTIC_REPORT.md"
)


with open(
    REPORT_PATH,
    "w",
    encoding="utf-8"
) as f:

    f.write(report)


print()
print("=" * 80)
print("GPT 분석 완료")
print("=" * 80)

print(
    "보고서:",
    REPORT_PATH.resolve()
)