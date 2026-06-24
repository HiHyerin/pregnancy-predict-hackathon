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
TOP_N_PERMUTATION = 20

target_col = "임신 성공 여부"
id_col = "ID"


def get_feature_groups(feature_names):
    groups = {
        "age": [
            "시술 당시 나이",
            "난자 기증자 나이",
            "정자 기증자 나이",
        ],
        "history": [
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
        ],
        "infertility_cause": [
            col
            for col in feature_names
            if col.startswith("남성")
            or col.startswith("여성")
            or col.startswith("부부")
            or col.startswith("불명확")
            or col.startswith("불임 원인")
        ],
        "egg_embryo_count": [
            "총 생성 배아 수",
            "미세주입된 난자 수",
            "미세주입에서 생성된 배아 수",
            "이식된 배아 수",
            "미세주입 배아 이식 수",
            "저장된 배아 수",
            "미세주입 후 저장된 배아 수",
            "해동된 배아 수",
            "해동 난자 수",
            "수집된 신선 난자 수",
            "저장된 신선 난자 수",
            "혼합된 난자 수",
            "파트너 정자와 혼합된 난자 수",
            "기증자 정자와 혼합된 난자 수",
        ],
        "treatment_type": [
            "시술 시기 코드",
            "시술 유형",
            "특정 시술 유형",
            "배란 자극 여부",
            "배란 유도 유형",
            "배아 생성 주요 이유",
            "난자 출처",
            "정자 출처",
        ],
        "embryo_usage_flag": [
            "단일 배아 이식 여부",
            "동결 배아 사용 여부",
            "신선 배아 사용 여부",
            "기증 배아 사용 여부",
            "대리모 여부",
            "착상 전 유전 검사 사용 여부",
            "착상 전 유전 진단 사용 여부",
            "PGD 시술 여부",
            "PGS 시술 여부",
        ],
        "elapsed_day": [
            "난자 채취 경과일",
            "난자 해동 경과일",
            "난자 혼합 경과일",
            "배아 이식 경과일",
            "배아 해동 경과일",
            "임신 시도 또는 마지막 임신 경과 연수",
        ],
    }

    return {
        group_name: [col for col in cols if col in feature_names]
        for group_name, cols in groups.items()
    }


def permutation_importance_one_feature(
    model,
    X_valid,
    y_valid,
    feature_name,
    base_score,
    rng,
):
    X_permuted = X_valid.copy()
    X_permuted[feature_name] = rng.permutation(X_permuted[feature_name].values)
    permuted_pred = model.predict_proba(X_permuted)[:, 1]
    permuted_score = roc_auc_score(y_valid, permuted_pred)
    return base_score - permuted_score


def permutation_importance_feature_group(
    model,
    X_valid,
    y_valid,
    group_name,
    group_features,
    base_score,
    rng,
):
    if not group_features:
        return None

    X_permuted = X_valid.copy()

    for feature_name in group_features:
        X_permuted[feature_name] = rng.permutation(X_permuted[feature_name].values)

    permuted_pred = model.predict_proba(X_permuted)[:, 1]
    permuted_score = roc_auc_score(y_valid, permuted_pred)

    return {
        "group": group_name,
        "feature_count": len(group_features),
        "importance_drop": base_score - permuted_score,
    }


train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")

X = train.drop(columns=[id_col, target_col])
y = train[target_col]
X_test = test.drop(columns=[id_col])

categorical_cols = X.select_dtypes(include=["object", "string"]).columns.tolist()
numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
feature_names = numeric_cols + categorical_cols
feature_groups = get_feature_groups(feature_names)

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
rng = np.random.default_rng(SEED)

fold_scores = []
best_iterations = []
oof_pred = np.zeros(len(train))
test_pred = np.zeros(len(test))

importance_rows = []
permutation_rows = []
group_permutation_rows = []

for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), start=1):
    X_train, X_valid = X.iloc[tr_idx], X.iloc[va_idx]
    y_train, y_valid = y.iloc[tr_idx], y.iloc[va_idx]

    X_train_transformed = pd.DataFrame(
        preprocessor.fit_transform(X_train),
        columns=feature_names,
        index=X_train.index,
    )
    X_valid_transformed = pd.DataFrame(
        preprocessor.transform(X_valid),
        columns=feature_names,
        index=X_valid.index,
    )
    X_test_transformed = pd.DataFrame(
        preprocessor.transform(X_test),
        columns=feature_names,
        index=X_test.index,
    )

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

    gain_importance = model.booster_.feature_importance(importance_type="gain")
    split_importance = model.booster_.feature_importance(importance_type="split")

    fold_importance = pd.DataFrame(
        {
            "fold": fold,
            "feature": feature_names,
            "gain": gain_importance,
            "split": split_importance,
        }
    )
    importance_rows.append(fold_importance)

    top_features = (
        fold_importance.sort_values("gain", ascending=False)
        .head(TOP_N_PERMUTATION)["feature"]
        .tolist()
    )

    for feature_name in top_features:
        importance_drop = permutation_importance_one_feature(
            model=model,
            X_valid=X_valid_transformed,
            y_valid=y_valid,
            feature_name=feature_name,
            base_score=score,
            rng=rng,
        )

        permutation_rows.append(
            {
                "fold": fold,
                "feature": feature_name,
                "importance_drop": importance_drop,
            }
        )

    for group_name, group_features in feature_groups.items():
        result = permutation_importance_feature_group(
            model=model,
            X_valid=X_valid_transformed,
            y_valid=y_valid,
            group_name=group_name,
            group_features=group_features,
            base_score=score,
            rng=rng,
        )

        if result is not None:
            result["fold"] = fold
            group_permutation_rows.append(result)

    print(
        f"Fold {fold}: ROC-AUC = {score:.6f}, "
        f"best iteration = {model.best_iteration_}"
    )

print(f"\nCV ROC-AUC mean: {np.mean(fold_scores):.6f}")
print(f"CV ROC-AUC std : {np.std(fold_scores):.6f}")
print(f"OOF ROC-AUC    : {roc_auc_score(y, oof_pred):.6f}")
print(f"Best iterations: {best_iterations}")

print(f"\nTest prediction min : {test_pred.min():.6f}")
print(f"Test prediction max : {test_pred.max():.6f}")
print(f"Test prediction mean: {test_pred.mean():.6f}")
print(f"Test prediction NaN : {np.isnan(test_pred).sum()}")

importance_df = pd.concat(importance_rows, ignore_index=True)
importance_summary = (
    importance_df.groupby("feature", as_index=False)
    .agg(
        gain_mean=("gain", "mean"),
        gain_std=("gain", "std"),
        split_mean=("split", "mean"),
        split_std=("split", "std"),
    )
    .sort_values("gain_mean", ascending=False)
)

print("\n===== Gain Importance Top 30 =====")
print(importance_summary.head(30).to_string(index=False))

print("\n===== Split Importance Top 30 =====")
print(
    importance_summary.sort_values("split_mean", ascending=False)
    .head(30)
    .to_string(index=False)
)

permutation_df = pd.DataFrame(permutation_rows)
permutation_summary = (
    permutation_df.groupby("feature", as_index=False)
    .agg(
        permutation_drop_mean=("importance_drop", "mean"),
        permutation_drop_std=("importance_drop", "std"),
        appeared_folds=("fold", "nunique"),
    )
    .sort_values("permutation_drop_mean", ascending=False)
)

print(f"\n===== Permutation Importance Top Features =====")
print(permutation_summary.head(30).to_string(index=False))

group_permutation_df = pd.DataFrame(group_permutation_rows)
group_permutation_summary = (
    group_permutation_df.groupby("group", as_index=False)
    .agg(
        feature_count=("feature_count", "first"),
        permutation_drop_mean=("importance_drop", "mean"),
        permutation_drop_std=("importance_drop", "std"),
    )
    .sort_values("permutation_drop_mean", ascending=False)
)

print("\n===== Group Permutation Importance =====")
print(group_permutation_summary.to_string(index=False))
