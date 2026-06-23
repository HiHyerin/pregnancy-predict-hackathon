import numpy as np
import pandas as pd

from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


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

# CatBoost의 범주형 Feature에는 NaN 대신 별도의 문자열 범주를 전달한다.
# 수치형 결측치는 CatBoost가 직접 처리하므로 그대로 유지한다.
for col in categorical_cols:
    X[col] = X[col].fillna("__MISSING__").astype(str)
    X_test[col] = X_test[col].fillna("__MISSING__").astype(str)

skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

fold_scores = []
best_iterations = []
oof_pred = np.zeros(len(train))
test_pred = np.zeros(len(test))

for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), start=1):
    X_train, X_valid = X.iloc[tr_idx], X.iloc[va_idx]
    y_train, y_valid = y.iloc[tr_idx], y.iloc[va_idx]

    model = CatBoostClassifier(
        iterations=200,
        learning_rate=0.1,
        depth=6,
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=SEED,
        verbose=0,
        allow_writing_files=False,
    )

    model.fit(
        X_train,
        y_train,
        eval_set=(X_valid, y_valid),
        cat_features=categorical_cols,
        early_stopping_rounds=20,
        use_best_model=True,
    )

    valid_pred = model.predict_proba(X_valid)[:, 1]
    oof_pred[va_idx] = valid_pred

    test_pred += model.predict_proba(X_test)[:, 1] / N_SPLITS

    score = roc_auc_score(y_valid, valid_pred)
    fold_scores.append(score)
    best_iterations.append(model.get_best_iteration())

    print(
        f"Fold {fold}: ROC-AUC = {score:.6f}, "
        f"best iteration = {model.get_best_iteration()}"
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
