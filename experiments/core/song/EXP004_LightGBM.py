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

train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")

SEED = 42
N_SPLITS = 5

target_col = "임신 성공 여부"
id_col = "ID"

X = train.drop(columns=[id_col, target_col])
y = train[target_col]
X_test = test.drop(columns=[id_col])

categorical_cols = X.select_dtypes(include=["object", "string"]).columns.tolist()
numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()

# LightGBM은 수치형 NaN을 직접 처리할 수 있으므로 그대로 둔다.
# 범주형 Feature는 문자열 그대로 학습할 수 없어서 fold 내부에서 숫자 코드로 변환한다.
# 결측 범주는 "__MISSING__"이라는 별도 범주로 유지한다.
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
