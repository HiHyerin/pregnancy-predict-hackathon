# [1/9] Import
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
from catboost import CatBoostClassifier
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# [2/9] Fixed RandomSeed
SEED = 42
N_FOLDS = 5
np.random.seed(SEED)

# [3/9] Data Load
import os

if os.path.exists('/kaggle/input'):
    data_dir = '/kaggle/input/datasets/kimjunghyeun4/dataset'
else:
    data_dir = '.'

train = pd.read_csv(f'{data_dir}/train.csv')
test  = pd.read_csv(f'{data_dir}/test.csv')
submission = pd.read_csv(f'{data_dir}/sample_submission.csv')

print(f'Train: {train.shape}, Test: {test.shape}')

# [4/9] Feature & Target Split
TARGET = '임신 성공 여부'
ID_COL = 'ID'

# 84% 이상 결측 컬럼만 제거 (난자 해동 경과일, 배아 해동 경과일 등)
# 배아 이식 경과일(17% 결측)은 살림
drop_cols = []
for col in train.columns:
    if col in [ID_COL, TARGET]:
        continue
    if train[col].isnull().mean() > 0.8:
        drop_cols.append(col)

print(f'제거 컬럼 ({len(drop_cols)}개):', drop_cols)

train = train.drop(columns=[ID_COL] + drop_cols)
test  = test.drop(columns=[ID_COL] + drop_cols)

X = train.drop(columns=[TARGET])
y = train[TARGET]

print(f'Feature 수: {X.shape[1]}')
print(f'Target 분포:\n{y.value_counts()}')

# [5/9] Data Preprocessing
# --- Ordinal 매핑 정의 ---
count_order = ['0회', '1회', '2회', '3회', '4회', '5회', '6회 이상']
age_order   = ['만18-34세', '만35-37세', '만38-39세', '만40-42세', '만43-44세', '만45-50세', '알 수 없음']
donor_age_order = ['만20세 이하', '만21-25세', '만26-30세', '만31-35세', '만36-40세', '만41-45세', '알 수 없음']

count_map     = {v: i for i, v in enumerate(count_order)}
age_map       = {v: i for i, v in enumerate(age_order)}
donor_age_map = {v: i for i, v in enumerate(donor_age_order)}

count_cols = ['총 시술 횟수', '클리닉 내 총 시술 횟수', 'IVF 시술 횟수', 'DI 시술 횟수',
              '총 임신 횟수', 'IVF 임신 횟수', 'DI 임신 횟수',
              '총 출산 횟수', 'IVF 출산 횟수', 'DI 출산 횟수']

def add_features(df):
    df = df.copy()

    # 1) 배아 품질 비율 피처 (importance 상위권)
    df['수정률'] = df['총 생성 배아 수'] / (df['혼합된 난자 수'] + 1)
    df['배아 이용률'] = df['이식된 배아 수'] / (df['총 생성 배아 수'] + 1)
    df['ICSI 배아 비율'] = df['미세주입에서 생성된 배아 수'] / (df['총 생성 배아 수'] + 1)
    df['배아 저장률'] = df['저장된 배아 수'] / (df['총 생성 배아 수'] + 1)
    df['난자 손실률'] = 1 - (df['혼합된 난자 수'] / (df['수집된 신선 난자 수'] + 1))
    df['미세주입 이식률'] = df['미세주입 배아 이식 수'] / (df['미세주입에서 생성된 배아 수'] + 1)
    df['신선난자 활용률'] = df['혼합된 난자 수'] / (df['수집된 신선 난자 수'] + 1)
    df['파트너정자 비율'] = df['파트너 정자와 혼합된 난자 수'] / (df['혼합된 난자 수'] + 1)

    # 2) 불임 원인 집약
    sperm_cols  = ['불임 원인 - 정자 농도', '불임 원인 - 정자 면역학적 요인',
                   '불임 원인 - 정자 운동성', '불임 원인 - 정자 형태']
    female_cols = ['불임 원인 - 난관 질환', '불임 원인 - 배란 장애',
                   '불임 원인 - 여성 요인', '불임 원인 - 자궁경부 문제', '불임 원인 - 자궁내막증']
    all_cause_cols = ['남성 주 불임 원인', '남성 부 불임 원인', '여성 주 불임 원인',
                      '여성 부 불임 원인', '부부 주 불임 원인', '부부 부 불임 원인',
                      '불명확 불임 원인'] + sperm_cols + female_cols

    df['정자 문제 합계'] = df[sperm_cols].sum(axis=1)
    df['여성 불임 원인 합계'] = df[female_cols].sum(axis=1)
    df['총 불임 원인 수'] = df[all_cause_cols].sum(axis=1)

    # 3) 시술 이력 조합
    df['시술_횟수_num'] = df['총 시술 횟수'].map(count_map).fillna(0)
    df['임신_횟수_num'] = df['총 임신 횟수'].map(count_map).fillna(0)
    df['나이_num'] = df['시술 당시 나이'].map(age_map).fillna(6)

    df['과거 임신 성공률'] = df['임신_횟수_num'] / (df['시술_횟수_num'] + 1)
    df['나이×시술횟수'] = df['나이_num'] * df['시술_횟수_num']

    df = df.drop(columns=['시술_횟수_num', '임신_횟수_num', '나이_num'])

    # 4) 현재시술용 여부
    df['현재시술용_여부'] = df['배아 생성 주요 이유'].apply(
        lambda x: 1 if isinstance(x, str) and '현재 시술용' in x else 0
    )

    # 5) 시술 유형 파생
    df['ICSI 포함 여부'] = df['특정 시술 유형'].apply(
        lambda x: 1 if isinstance(x, str) and 'ICSI' in x else 0
    )
    df['배반포_시술_여부'] = df['특정 시술 유형'].apply(
        lambda x: 1 if isinstance(x, str) and 'BLASTOCYST' in x else 0
    )

    return df

def preprocess(df_train, df_test):
    df_tr = df_train.copy()
    df_te = df_test.copy()

    # 1) 파생 피처 추가
    df_tr = add_features(df_tr)
    df_te = add_features(df_te)

    # 2) 횟수 컬럼 Ordinal 인코딩
    for col in count_cols:
        if col in df_tr.columns:
            df_tr[col] = df_tr[col].map(count_map)
            df_te[col] = df_te[col].map(count_map)

    # 3) 나이 컬럼 Ordinal 인코딩
    df_tr['시술 당시 나이'] = df_tr['시술 당시 나이'].map(age_map)
    df_te['시술 당시 나이'] = df_te['시술 당시 나이'].map(age_map)

    # 4) 기증자 나이 Ordinal 인코딩
    for col in ['난자 기증자 나이', '정자 기증자 나이']:
        if col in df_tr.columns:
            df_tr[col] = df_tr[col].map(donor_age_map)
            df_te[col] = df_te[col].map(donor_age_map)

    # 5) 결측치 처리 (train 통계 기준)
    # CatBoost는 범주형 결측을 'None' 문자열로, 수치형은 그대로 넘김
    cat_cols = df_tr.select_dtypes(include='object').columns.tolist()
    num_cols = df_tr.select_dtypes(include=['int64', 'float64']).columns.tolist()

    for col in cat_cols:
        df_tr[col] = df_tr[col].fillna('None')
        df_te[col] = df_te[col].fillna('None')

    for col in num_cols:
        if df_tr[col].isnull().sum() > 0:
            median_val = df_tr[col].median()
            df_tr[col] = df_tr[col].fillna(median_val)
            df_te[col] = df_te[col].fillna(median_val)

    return df_tr, df_te

X_proc, test_proc = preprocess(X, test)

# CatBoost용 범주형 컬럼 인덱스 추출
cat_features = [col for col in X_proc.select_dtypes(include='object').columns]
print('전처리 완료')
print(f'X_proc: {X_proc.shape}, test_proc: {test_proc.shape}')
print(f'범주형 피처 수: {len(cat_features)}')

# [6/9] Model Definition
def get_model():
    return CatBoostClassifier(
        random_seed=SEED,
        iterations=3000,
        learning_rate=0.03,
        depth=8,
        l2_leaf_reg=3,
        eval_metric='AUC',
        auto_class_weights='Balanced',
        verbose=0
    )

# [7/9] Model Fitting (StratifiedKFold CV)
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

oof_preds  = np.zeros(len(X_proc))
test_preds = np.zeros(len(test_proc))
fold_scores = []

for fold, (tr_idx, val_idx) in enumerate(tqdm(skf.split(X_proc, y), total=N_FOLDS, desc='Fold')):
    X_tr, X_val = X_proc.iloc[tr_idx], X_proc.iloc[val_idx]
    y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

    model = get_model()
    model.fit(
        X_tr, y_tr,
        eval_set=(X_val, y_val),
        cat_features=cat_features,
        early_stopping_rounds=150
    )

    val_prob = model.predict_proba(X_val)[:, 1]
    oof_preds[val_idx] = val_prob
    test_preds += model.predict_proba(test_proc)[:, 1] / N_FOLDS

    score = roc_auc_score(y_val, val_prob)
    fold_scores.append(score)
    print(f'  Fold {fold+1} ROC-AUC: {score:.5f}')

oof_score = roc_auc_score(y, oof_preds)
print(f'\nOOF ROC-AUC : {oof_score:.5f}')
print(f'Fold 평균   : {np.mean(fold_scores):.5f} ± {np.std(fold_scores):.5f}')

# [8/9] Inference
print(f'Test 예측값 범위: {test_preds.min():.4f} ~ {test_preds.max():.4f}')
print(f'결측치: {np.isnan(test_preds).sum()}')

# [9/9] Submit
submission['probability'] = test_preds
file_name = f'submit_v1.csv'
submission.to_csv(file_name, index=False)
