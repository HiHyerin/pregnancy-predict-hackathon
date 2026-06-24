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


def safe_divide(numerator, denominator):
    denominator = denominator.replace(0, np.nan)
    return numerator / denominator


def make_features(df):
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

    count_map = {
        "0회": 0,
        "1회": 1,
        "2회": 2,
        "3회": 3,
        "4회": 4,
        "5회": 5,
        "6회 이상": 6,
    }

    if "시술 당시 나이" in df.columns:
        df["시술 당시 나이_ord"] = df["시술 당시 나이"].map(age_map)

    for col in ["난자 기증자 나이", "정자 기증자 나이"]:
        if col in df.columns:
            df[f"{col}_ord"] = df[col].map(donor_age_map)

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

    if {"총 시술 횟수_num", "총 임신 횟수_num"}.issubset(df.columns):
        df["총 시술_임신 차이"] = df["총 시술 횟수_num"] - df["총 임신 횟수_num"]

    if {"총 임신 횟수_num", "총 출산 횟수_num"}.issubset(df.columns):
        df["총 임신_출산 차이"] = df["총 임신 횟수_num"] - df["총 출산 횟수_num"]

    if {"IVF 시술 횟수_num", "IVF 임신 횟수_num"}.issubset(df.columns):
        df["IVF 시술_임신 차이"] = df["IVF 시술 횟수_num"] - df["IVF 임신 횟수_num"]

    if {"IVF 임신 횟수_num", "IVF 출산 횟수_num"}.issubset(df.columns):
        df["IVF 임신_출산 차이"] = df["IVF 임신 횟수_num"] - df["IVF 출산 횟수_num"]

    if {"DI 시술 횟수_num", "DI 임신 횟수_num"}.issubset(df.columns):
        df["DI 시술_임신 차이"] = df["DI 시술 횟수_num"] - df["DI 임신 횟수_num"]

    if {"DI 임신 횟수_num", "DI 출산 횟수_num"}.issubset(df.columns):
        df["DI 임신_출산 차이"] = df["DI 임신 횟수_num"] - df["DI 출산 횟수_num"]

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

    if "특정 시술 유형" in df.columns:
        treatment = df["특정 시술 유형"].fillna("Unknown").astype(str)
        df["특정시술_ICSI포함"] = treatment.str.contains("ICSI", regex=False).astype(int)
        df["특정시술_IVF포함"] = treatment.str.contains("IVF", regex=False).astype(int)
        df["특정시술_AH포함"] = treatment.str.contains("AH", regex=False).astype(int)
        df["특정시술_BLASTOCYST포함"] = treatment.str.contains("BLASTOCYST", regex=False).astype(int)

    return df


train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")

SEED = 42
N_SPLITS = 5

target_col = "임신 성공 여부"
id_col = "ID"

X = train.drop(columns=[id_col, target_col])
y = train[target_col]
X_test = test.drop(columns=[id_col])

X = make_features(X)
X_test = make_features(X_test)

categorical_cols = X.select_dtypes(include=["object", "string"]).columns.tolist()
numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()

print(f"Original feature count: {train.drop(columns=[id_col, target_col]).shape[1]}")
print(f"Feature count after FE: {X.shape[1]}")
print(f"Added feature count   : {X.shape[1] - train.drop(columns=[id_col, target_col]).shape[1]}")

# LightGBM은 수치형 NaN을 직접 처리할 수 있으므로 그대로 둔다.
# 범주형 Feature는 문자열 그대로 학습할 수 없어서 fold 내부에서 숫자 코드로 변환한다.
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
oof_pred = np.zeros(len(train))
test_pred = np.zeros(len(test))

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
        f"Fold {fold}: ROC-AUC = {score:.6f}, "
        f"best iteration = {model.best_iteration_}"
    )

print(f"CV ROC-AUC mean: {np.mean(fold_scores):.6f}")
print(f"CV ROC-AUC std : {np.std(fold_scores):.6f}")
print(f"OOF ROC-AUC    : {roc_auc_score(y, oof_pred):.6f}")
print(f"Best iterations: {best_iterations}")

print(f"Test prediction min : {test_pred.min():.6f}")
print(f"Test prediction max : {test_pred.max():.6f}")
print(f"Test prediction mean: {test_pred.mean():.6f}")
print(f"Test prediction NaN : {np.isnan(test_pred).sum()}")
print(f"Test prediction head: {test_pred[:10]}")
