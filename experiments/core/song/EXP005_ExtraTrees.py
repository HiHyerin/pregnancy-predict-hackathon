import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder


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

# ExtraTrees는 NaN을 직접 처리하지 못하므로 수치형 결측치를 중앙값으로 채운다.
# 대신 add_indicator=True로 원래 결측이었는지에 대한 정보를 함께 남긴다.
numeric_preprocessor = SimpleImputer(strategy="median", add_indicator=True)

# 범주형 Feature는 결측치를 별도 범주로 만든 뒤 숫자 코드로 변환한다.
# validation/test에서 train fold에 없던 새로운 범주가 나오면 -1로 처리한다.
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

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                ExtraTreesClassifier(
                    n_estimators=300,
                    min_samples_leaf=10,
                    max_features="sqrt",
                    random_state=SEED,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    model.fit(X_train, y_train)

    valid_pred = model.predict_proba(X_valid)[:, 1]
    oof_pred[va_idx] = valid_pred

    test_pred += model.predict_proba(X_test)[:, 1] / N_SPLITS

    score = roc_auc_score(y_valid, valid_pred)
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
