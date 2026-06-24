import warnings

import numpy as np
import pandas as pd

from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder


warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names, but LGBMClassifier was fitted with feature names",
)


SEED = 42
N_SPLITS = 5

target_col = "임신 성공 여부"
id_col = "ID"


def safe_divide(numerator, denominator):
    denominator = denominator.replace(0, np.nan)
    return numerator / denominator


def add_age_features(df):
    df = df.copy()

    age_map = {
        "만18-34세": 0,
        "만35-37세": 1,
        "만38-39세": 2,
        "만40-42세": 3,
        "만43-44세": 4,
        "만45-50세": 5,
        "알 수 없음": np.nan,
    }

    donor_age_map = {
        "만20세 이하": 0,
        "만21-25세": 1,
        "만26-30세": 2,
        "만31-35세": 3,
        "만36-40세": 4,
        "만41-45세": 5,
        "알 수 없음": np.nan,
    }

    if "시술 당시 나이" in df.columns:
        df["시술 당시 나이_ord"] = df["시술 당시 나이"].map(age_map)

    for col in ["난자 기증자 나이", "정자 기증자 나이"]:
        if col in df.columns:
            df[f"{col}_ord"] = df[col].map(donor_age_map)

    return df


def add_history_features(df):
    df = df.copy()

    count_map = {
        "0회": 0,
        "1회": 1,
        "2회": 2,
        "3회": 3,
        "4회": 4,
        "5회": 5,
        "6회 이상": 6,
    }

    count_cols = [
        "총 시술 횟수",
        "클리닉 내 총 시술 횟수",
        "IVF 시술 횟수",
        "DI 시술 횟수",
        "총 임신 횟수",
        "IVF 임신 횟수",
        "DI 임신 횟수",
        "총 출산 횟수",
        "IVF 출산 횟수",
        "DI 출산 횟수",
    ]

    for col in count_cols:
        if col in df.columns:
            df[f"{col}_num"] = df[col].map(count_map)

    diff_pairs = {
        "총 시술_임신 차이": ("총 시술 횟수_num", "총 임신 횟수_num"),
        "총 임신_출산 차이": ("총 임신 횟수_num", "총 출산 횟수_num"),
        "IVF 시술_임신 차이": ("IVF 시술 횟수_num", "IVF 임신 횟수_num"),
        "IVF 임신_출산 차이": ("IVF 임신 횟수_num", "IVF 출산 횟수_num"),
        "DI 시술_임신 차이": ("DI 시술 횟수_num", "DI 임신 횟수_num"),
        "DI 임신_출산 차이": ("DI 임신 횟수_num", "DI 출산 횟수_num"),
    }

    for new_col, (left_col, right_col) in diff_pairs.items():
        if {left_col, right_col}.issubset(df.columns):
            df[new_col] = df[left_col] - df[right_col]

    return df


def add_cause_features(df):
    df = df.copy()

    cause_cols = [col for col in df.columns if col.startswith("불임 원인")]
    male_cause_cols = [col for col in cause_cols if "남성" in col or "정자" in col]
    female_cause_cols = [
        col
        for col in cause_cols
        if any(keyword in col for keyword in ["여성", "난관", "배란", "자궁", "내막"])
    ]

    if cause_cols:
        df["불임 원인 개수"] = df[cause_cols].sum(axis=1)

    if male_cause_cols:
        df["남성 관련 불임 원인 개수"] = df[male_cause_cols].sum(axis=1)

    if female_cause_cols:
        df["여성 관련 불임 원인 개수"] = df[female_cause_cols].sum(axis=1)

    return df


def add_ratio_features(df):
    df = df.copy()

    ratio_pairs = {
        "생성배아수_혼합난자수_비율": ("총 생성 배아 수", "혼합된 난자 수"),
        "이식배아수_생성배아수_비율": ("이식된 배아 수", "총 생성 배아 수"),
        "저장배아수_생성배아수_비율": ("저장된 배아 수", "총 생성 배아 수"),
        "미세주입생성배아수_미세주입난자수_비율": (
            "미세주입에서 생성된 배아 수",
            "미세주입된 난자 수",
        ),
        "미세주입이식배아수_이식배아수_비율": ("미세주입 배아 이식 수", "이식된 배아 수"),
        "파트너정자혼합난자수_혼합난자수_비율": ("파트너 정자와 혼합된 난자 수", "혼합된 난자 수"),
        "기증자정자혼합난자수_혼합난자수_비율": ("기증자 정자와 혼합된 난자 수", "혼합된 난자 수"),
    }

    for new_col, (num_col, den_col) in ratio_pairs.items():
        if {num_col, den_col}.issubset(df.columns):
            df[new_col] = safe_divide(df[num_col], df[den_col])

    return df


def add_treatment_features(df):
    df = df.copy()

    if "특정 시술 유형" in df.columns:
        treatment = df["특정 시술 유형"].fillna("Unknown").astype(str)
        df["특정시술_ICSI포함"] = treatment.str.contains("ICSI", regex=False).astype(int)
        df["특정시술_IVF포함"] = treatment.str.contains("IVF", regex=False).astype(int)
        df["특정시술_AH포함"] = treatment.str.contains("AH", regex=False).astype(int)
        df["특정시술_BLASTOCYST포함"] = treatment.str.contains("BLASTOCYST", regex=False).astype(int)

    return df


def make_features(df, groups):
    df = df.copy()

    if "age" in groups:
        df = add_age_features(df)

    if "history" in groups:
        df = add_history_features(df)

    if "cause" in groups:
        df = add_cause_features(df)

    if "ratio" in groups:
        df = add_ratio_features(df)

    if "treatment" in groups:
        df = add_treatment_features(df)

    return df


def run_lightgbm_cv(X, y, X_test):
    categorical_cols = X.select_dtypes(include=["object", "string"]).columns.tolist()
    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()

    categorical_preprocessor = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="constant", fill_value="__MISSING__"),
            ),
            (
                "ordinal",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", "passthrough", numeric_cols),
            ("categorical", categorical_preprocessor, categorical_cols),
        ]
    )

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

    fold_scores = []
    best_iterations = []
    oof_pred = np.zeros(len(X))
    test_pred = np.zeros(len(X_test))

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), start=1):
        X_train, X_valid = X.iloc[tr_idx], X.iloc[va_idx]
        y_train, y_valid = y.iloc[tr_idx], y.iloc[va_idx]

        X_train_transformed = np.asarray(preprocessor.fit_transform(X_train))
        X_valid_transformed = np.asarray(preprocessor.transform(X_valid))
        X_test_transformed = np.asarray(preprocessor.transform(X_test))

        model = LGBMClassifier(
            n_estimators=1000,
            learning_rate=0.03,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary",
            random_state=SEED,
            n_jobs=-1,
            verbose=-1,
        )

        model.fit(
            X_train_transformed,
            y_train,
            eval_set=[(X_valid_transformed, y_valid)],
            eval_metric="auc",
            callbacks=[
                early_stopping(stopping_rounds=50, verbose=False),
                log_evaluation(period=0),
            ],
        )

        valid_pred = model.predict_proba(X_valid_transformed)[:, 1]
        oof_pred[va_idx] = valid_pred

        test_pred += model.predict_proba(X_test_transformed)[:, 1] / N_SPLITS

        score = roc_auc_score(y_valid, valid_pred)
        fold_scores.append(score)
        best_iterations.append(model.best_iteration_)

        print(
            f"  Fold {fold}: ROC-AUC = {score:.6f}, "
            f"best iteration = {model.best_iteration_}"
        )

    return {
        "cv_mean": np.mean(fold_scores),
        "cv_std": np.std(fold_scores),
        "oof_auc": roc_auc_score(y, oof_pred),
        "best_iterations": best_iterations,
        "test_min": test_pred.min(),
        "test_max": test_pred.max(),
        "test_mean": test_pred.mean(),
        "test_nan": np.isnan(test_pred).sum(),
    }


train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")

X_base = train.drop(columns=[id_col, target_col])
y = train[target_col]
X_test_base = test.drop(columns=[id_col])

experiments = {
    "baseline": [],
    "age": ["age"],
    "history": ["history"],
    "cause": ["cause"],
    "ratio": ["ratio"],
    "treatment": ["treatment"],
    "all": ["age", "history", "cause", "ratio", "treatment"],
}

results = []

for exp_name, groups in experiments.items():
    print(f"\n===== {exp_name} =====")

    X = make_features(X_base, groups)
    X_test = make_features(X_test_base, groups)

    print(f"Feature count: {X.shape[1]}")
    result = run_lightgbm_cv(X, y, X_test)

    result["experiment"] = exp_name
    result["groups"] = ",".join(groups) if groups else "none"
    result["feature_count"] = X.shape[1]
    results.append(result)

    print(f"{exp_name} OOF ROC-AUC: {result['oof_auc']:.6f}")

result_df = pd.DataFrame(results)
result_df = result_df[
    [
        "experiment",
        "groups",
        "feature_count",
        "cv_mean",
        "cv_std",
        "oof_auc",
        "best_iterations",
        "test_min",
        "test_max",
        "test_mean",
        "test_nan",
    ]
]

print("\n===== Summary =====")
print(result_df.to_string(index=False))
