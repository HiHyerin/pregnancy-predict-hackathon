import numpy as np
import pandas as pd

from sklearn.dummy import DummyClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


train = pd.read_csv("data/train.csv")

SEED = 42
N_SPLITS = 5

target_col = "임신 성공 여부"
id_col = "ID"

feature_cols = [c for c in train.columns if c not in [id_col, target_col]]

X = train[feature_cols]
y = train[target_col]

# 이진 분류이므로 각 fold의 0과 1 비율이 비슷하도록 나눈다.
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

fold_scores = []
oof_pred = np.zeros(len(train))

for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), start=1):
    X_train, X_valid = X.iloc[tr_idx], X.iloc[va_idx]
    y_train, y_valid = y.iloc[tr_idx], y.iloc[va_idx]

    # 학습 데이터의 클래스 비율만 이용하는 최소 기준 모델
    model = DummyClassifier(strategy="prior")
    model.fit(X_train, y_train)

    # ROC-AUC는 클래스가 아니라 임신 성공(1)의 확률로 평가한다.
    pred = model.predict_proba(X_valid)[:, 1]
    oof_pred[va_idx] = pred

    score = roc_auc_score(y_valid, pred)
    fold_scores.append(score)
    print(f"Fold {fold}: ROC-AUC = {score:.6f}")

print(f"CV ROC-AUC mean: {np.mean(fold_scores):.6f}")
print(f"CV ROC-AUC std : {np.std(fold_scores):.6f}")
print(f"OOF ROC-AUC    : {roc_auc_score(y, oof_pred):.6f}")
