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


def select_columns(feature_groups, include_groups=None, exclude_groups=None):
    include_groups = include_groups or []
    exclude_groups = exclude_groups or []

    if include_groups:
        selected_cols = []

        for group_name in include_groups:
            selected_cols.extend(feature_groups[group_name])
    else:
        selected_cols = [
            col
            for group_cols in feature_groups.values()
            for col in group_cols
        ]

    excluded_cols = set()

    for group_name in exclude_groups:
        excluded_cols.update(feature_groups[group_name])

    selected_cols = [col for col in selected_cols if col not in excluded_cols]

    return list(dict.fromkeys(selected_cols))


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

feature_names = X_base.columns.tolist()
feature_groups = get_feature_groups(feature_names)

experiments = {
    "all_original": {
        "include": [],
        "exclude": [],
    },
    "core_egg_age_elapsed": {
        "include": ["egg_embryo_count", "age", "elapsed_day"],
        "exclude": [],
    },
    "core_plus_treatment": {
        "include": ["egg_embryo_count", "age", "elapsed_day", "treatment_type"],
        "exclude": [],
    },
    "core_plus_history": {
        "include": ["egg_embryo_count", "age", "elapsed_day", "history"],
        "exclude": [],
    },
    "core_plus_treatment_history": {
        "include": ["egg_embryo_count", "age", "elapsed_day", "treatment_type", "history"],
        "exclude": [],
    },
    "remove_infertility_cause": {
        "include": [],
        "exclude": ["infertility_cause"],
    },
    "remove_embryo_usage_flag": {
        "include": [],
        "exclude": ["embryo_usage_flag"],
    },
    "remove_low_importance_groups": {
        "include": [],
        "exclude": ["infertility_cause", "embryo_usage_flag"],
    },
}

results = []

for exp_name, config in experiments.items():
    print(f"\n===== {exp_name} =====")

    selected_cols = select_columns(
        feature_groups=feature_groups,
        include_groups=config["include"],
        exclude_groups=config["exclude"],
    )

    X = X_base[selected_cols]
    X_test = X_test_base[selected_cols]

    print(f"Selected groups : {config['include'] if config['include'] else 'all'}")
    print(f"Excluded groups : {config['exclude'] if config['exclude'] else 'none'}")
    print(f"Feature count   : {X.shape[1]}")

    result = run_lightgbm_cv(X, y, X_test)

    result["experiment"] = exp_name
    result["include_groups"] = ",".join(config["include"]) if config["include"] else "all"
    result["exclude_groups"] = ",".join(config["exclude"]) if config["exclude"] else "none"
    result["feature_count"] = X.shape[1]
    results.append(result)

    print(f"{exp_name} OOF ROC-AUC: {result['oof_auc']:.6f}")

result_df = pd.DataFrame(results)
result_df = result_df[
    [
        "experiment",
        "include_groups",
        "exclude_groups",
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
