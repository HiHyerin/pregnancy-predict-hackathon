import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


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

# LogisticRegression은 NaN을 직접 처리하지 못한다.
# 중앙값으로 채우되, 원래 결측이었다는 정보를 indicator로 함께 남긴다.
numeric_preprocessor = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("scaler", StandardScaler()),
    ]
)

# 범주형 Feature: 결측치를 별도 범주로 만들고 One-hot encoding을 적용한다.
categorical_preprocessor = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="constant", fill_value="__MISSING__"),
        ),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", numeric_preprocessor, numeric_cols),
        ("categorical", categorical_preprocessor, categorical_cols),
    ]
)

skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

fold_scores = []
oof_pred = np.zeros(len(train))
test_pred = np.zeros(len(test))

for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), start=1):
    X_train, X_valid = X.iloc[tr_idx], X.iloc[va_idx]
    y_train, y_valid = y.iloc[tr_idx], y.iloc[va_idx]

    # Pipeline을 fold 안에서 학습해 validation 정보가 전처리에 섞이지 않게 한다.
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                LogisticRegression(
                    solver="liblinear",
                    max_iter=1000,
                    random_state=SEED,
                ),
            ),
        ]
    )

    model.fit(X_train, y_train)

    pred = model.predict_proba(X_valid)[:, 1]
    oof_pred[va_idx] = pred

    # 각 fold에서 학습된 전처리와 모델로 test를 예측해 평균한다.
    test_pred += model.predict_proba(X_test)[:, 1] / N_SPLITS

    score = roc_auc_score(y_valid, pred)
    fold_scores.append(score)
    print(f"Fold {fold}: ROC-AUC = {score:.6f}")

print(f"CV ROC-AUC mean: {np.mean(fold_scores):.6f}")
print(f"CV ROC-AUC std : {np.std(fold_scores):.6f}")
print(f"OOF ROC-AUC    : {roc_auc_score(y, oof_pred):.6f}")

print(f"Test prediction min : {test_pred.min():.6f}")
print(f"Test prediction max : {test_pred.max():.6f}")
print(f"Test prediction mean: {test_pred.mean():.6f}")
print(f"Test prediction NaN : {np.isnan(test_pred).sum()}")
print(f"Test prediction head: {test_pred[:10]}")
