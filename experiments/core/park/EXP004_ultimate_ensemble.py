import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
from catboost import CatBoostClassifier, Pool
from xgboost import XGBClassifier
import warnings

warnings.filterwarnings('ignore')

def load_and_preprocess(train_path='train.csv', test_path='test.csv'):
    print("📦 [1/6] 데이터 로드 및 '도메인 교차 피처' 생성을 시작합니다...")
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    y_train = train['임신 성공 여부']
    X_train = train.drop(columns=['ID', '임신 성공 여부'])
    X_test = test.drop(columns=['ID'])
    test_id = test['ID']

    def engineer_features(df):
        df_copy = df.copy()
        for col in df_copy.columns:
            if df_copy[col].dtype == 'object' and df_copy[col].astype(str).str.contains('회').any():
                df_copy[col] = df_copy[col].astype(str).str.replace('회', '')
                df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce')
        
        calc_cols = ['총 출산 횟수', '총 생성 배아 수', '이식된 배아 수', '수집된 신선 난자 수', '총 임신 횟수', '총 시술 횟수']
        for col in calc_cols:
            if col in df_copy.columns:
                df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce').fillna(0)
        
        # 🌟 시뮬레이터 옵션 적용: 도메인 교차 피처 (Domain Cross Features) 추가
        if '총 출산 횟수' in df_copy.columns:
            df_copy['출산_경험_여부'] = (df_copy['총 출산 횟수'] > 0).astype(int)
        if '이식된 배아 수' in df_copy.columns and '총 생성 배아 수' in df_copy.columns:
            df_copy['배아_이식_비율'] = np.where(df_copy['총 생성 배아 수'] > 0, df_copy['이식된 배아 수'] / df_copy['총 생성 배아 수'], 0)
        if '총 임신 횟수' in df_copy.columns and '총 시술 횟수' in df_copy.columns:
            df_copy['시술대비_임신_성공률'] = np.where(df_copy['총 시술 횟수'] > 0, df_copy['총 임신 횟수'] / df_copy['총 시술 횟수'], 0)
        
        # 불임 원인 교차 피처
        cause_cols = [col for col in df_copy.columns if '불임 원인 -' in col]
        if cause_cols:
            df_copy['총_불임_원인_수'] = df_copy[cause_cols].sum(axis=1)

        cat_cols = df_copy.select_dtypes(include=['object', 'category']).columns.tolist()
        return df_copy, cat_cols

    X_train, cat_cols = engineer_features(X_train)
    X_test, _ = engineer_features(X_test)
    
    # 카테고리 인코딩
    X_train_int, X_test_int = X_train.copy(), X_test.copy()
    for col in cat_cols:
        le = LabelEncoder()
        all_data = list(X_train[col].astype(str)) + list(X_test[col].astype(str))
        le.fit(all_data)
        X_train_int[col] = le.transform(X_train[col].astype(str))
        X_test_int[col] = le.transform(X_test[col].astype(str))
        
    X_train_cat, X_test_cat = X_train_int.copy(), X_test_int.copy()
    for col in cat_cols:
        X_train_cat[col] = X_train_cat[col].astype('category')
        X_test_cat[col] = X_test_cat[col].astype('category')

    return X_train_int, X_test_int, X_train_cat, X_test_cat, y_train, test_id, cat_cols

def main():
    MASTER_SEED = 1111
    print(f"\n🎯 [시뮬레이터 1위 세팅] 10-Fold 스태킹 앙상블을 시작합니다! (Seed: {MASTER_SEED})")

    X_train_int, X_test_int, X_train_cat, X_test_cat, y_train, test_id, cat_cols = load_and_preprocess()

    # 🌟 시뮬레이터 옵션 적용: 10-Fold 설정
    N_SPLITS = 10
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=MASTER_SEED)

    print("\n🚀 [2/6] 5개 개별 모델(Level 0) OOF 배열 생성 중...")
    oof_lgb = np.zeros(len(X_train_cat))
    oof_cat = np.zeros(len(X_train_cat))
    oof_xgb = np.zeros(len(X_train_cat))
    oof_ext = np.zeros(len(X_train_cat))
    oof_mlp = np.zeros(len(X_train_cat))
    
    test_preds_lgb = np.zeros(len(X_test_cat))
    test_preds_cat = np.zeros(len(X_test_cat))
    test_preds_xgb = np.zeros(len(X_test_cat))
    test_preds_ext = np.zeros(len(X_test_cat))
    test_preds_mlp = np.zeros(len(X_test_cat))

    # 기본 파라미터 (시간 단축을 위해 고정값 사용)
    best_lgb_params = {'n_estimators': 800, 'learning_rate': 0.03, 'random_state': MASTER_SEED, 'n_jobs': -1}
    best_cat_params = {'iterations': 800, 'learning_rate': 0.04, 'eval_metric': 'AUC', 'random_seed': MASTER_SEED, 'task_type': 'GPU', 'verbose': False}
    best_xgb_params = {'n_estimators': 800, 'learning_rate': 0.03, 'eval_metric': 'auc', 'tree_method': 'hist', 'device': 'cuda', 'random_state': MASTER_SEED}

    print("\n🔄 [3/6] 10-Fold 학습 및 OOF 타겟 인코딩 진행 (Data Leakage 방지)...")
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_int, y_train)):
        print(f"  --- Fold {fold + 1} / {N_SPLITS} ---")
        
        X_tr_cat, X_va_cat = X_train_cat.iloc[train_idx], X_train_cat.iloc[val_idx]
        X_tr_int, X_va_int = X_train_int.iloc[train_idx].copy(), X_train_int.iloc[val_idx].copy()
        X_te_int_fold = X_test_int.copy()
        y_tr, y_va = y_train.iloc[train_idx], y_train.iloc[val_idx]

        # 🌟 시뮬레이터 옵션 적용: OOF 타겟 인코딩 (Target Encoding)
        # 범주형 변수들을 Train의 타겟 평균으로 치환하여 강력한 숫자 피처로 변환
        for col in cat_cols:
            target_mean = y_tr.groupby(X_tr_int[col]).mean()
            X_tr_int[f'{col}_te'] = X_tr_int[col].map(target_mean).fillna(y_tr.mean())
            X_va_int[f'{col}_te'] = X_va_int[col].map(target_mean).fillna(y_tr.mean())
            X_te_int_fold[f'{col}_te'] = X_te_int_fold[col].map(target_mean).fillna(y_tr.mean())

        
        # 🌟 Neural Net과 ExtraTrees를 위한 결측치 처리 및 스케일링
        # 신경망은 빈칸(NaN)을 보면 에러를 내므로 안전한 숫자(-1)로 임시 채워줍니다.
       # 🌟 무적의 전처리: 무한대(inf)와 빈칸(NaN)을 모두 찾아 -1로 강제 덮어쓰기
        X_tr_clean = X_tr_int.replace([np.inf, -np.inf], np.nan).fillna(-1)
        X_va_clean = X_va_int.replace([np.inf, -np.inf], np.nan).fillna(-1)
        X_te_clean = X_te_int_fold.replace([np.inf, -np.inf], np.nan).fillna(-1)

        scaler = StandardScaler()
        X_tr_scaled = scaler.fit_transform(X_tr_clean)
        X_va_scaled = scaler.transform(X_va_clean)
        X_te_scaled = scaler.transform(X_te_clean)

        # 1. LightGBM
        model_lgb = lgb.LGBMClassifier(**best_lgb_params)
        model_lgb.fit(X_tr_cat, y_tr, eval_set=[(X_va_cat, y_va)], callbacks=[lgb.early_stopping(50, verbose=False)])
        oof_lgb[val_idx] = model_lgb.predict_proba(X_va_cat)[:, 1]
        test_preds_lgb += model_lgb.predict_proba(X_test_cat)[:, 1] / N_SPLITS

        # 2. CatBoost
        train_pool = Pool(X_tr_int, y_tr, cat_features=cat_cols)
        val_pool = Pool(X_va_int, y_va, cat_features=cat_cols)
        model_cat = CatBoostClassifier(**best_cat_params)
        model_cat.fit(train_pool, eval_set=val_pool, early_stopping_rounds=50)
        oof_cat[val_idx] = model_cat.predict_proba(val_pool)[:, 1]
        test_preds_cat += model_cat.predict_proba(X_te_int_fold)[:, 1] / N_SPLITS

        # 3. XGBoost
        model_xgb = XGBClassifier(**best_xgb_params, early_stopping_rounds=50)
        model_xgb.fit(X_tr_int, y_tr, eval_set=[(X_va_int, y_va)], verbose=False)
        oof_xgb[val_idx] = model_xgb.predict_proba(X_va_int)[:, 1]
        test_preds_xgb += model_xgb.predict_proba(X_te_int_fold)[:, 1] / N_SPLITS

        # 4. ExtraTrees (과적합 방지용 배깅 모델)
        model_ext = ExtraTreesClassifier(n_estimators=300, max_depth=8, random_state=MASTER_SEED, n_jobs=-1)
        model_ext.fit(X_tr_scaled, y_tr)
        oof_ext[val_idx] = model_ext.predict_proba(X_va_scaled)[:, 1]
        test_preds_ext += model_ext.predict_proba(X_te_scaled)[:, 1] / N_SPLITS

        # 5. Neural Net (MLP 딥러닝 - 다양성 주입)
        model_mlp = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=30, early_stopping=True, random_state=MASTER_SEED)
        model_mlp.fit(X_tr_scaled, y_tr)
        oof_mlp[val_idx] = model_mlp.predict_proba(X_va_scaled)[:, 1]
        test_preds_mlp += model_mlp.predict_proba(X_te_scaled)[:, 1] / N_SPLITS

    # ==========================================
    # 🧠 [4/6] Level 1: 스태킹(Stacking) 메타 모델 훈련
    # ==========================================
    print("\n🧠 [4/6] 5개 모델의 OOF 결과를 모아 메타 모델(Logistic Regression)을 학습시킵니다...")
    
    # 5개 모델이 예측한 확률값을 새로운 피처(Feature)로 만듦
    OOF_train = pd.DataFrame({
        'LGBM': oof_lgb,
        'CatBoost': oof_cat,
        'XGBoost': oof_xgb,
        'ExtraTrees': oof_ext,
        'NeuralNet': oof_mlp
    })

    OOF_test = pd.DataFrame({
        'LGBM': test_preds_lgb,
        'CatBoost': test_preds_cat,
        'XGBoost': test_preds_xgb,
        'ExtraTrees': test_preds_ext,
        'NeuralNet': test_preds_mlp
    })

    # 스태킹 메타 모델: 로지스틱 회귀가 각 모델의 신뢰도를 평가하여 최종 가중치를 결정
    meta_model = LogisticRegression()
    meta_model.fit(OOF_train, y_train)
    # 🌟🌟🌟 [여기에 추가] 내 컴퓨터에서 검증하는 예상 리더보드 점수 🌟🌟🌟
    oof_predictions = meta_model.predict_proba(OOF_train)[:, 1]
    expected_auc = roc_auc_score(y_train, oof_predictions)

    # 메타 모델이 학습한 실제 모델별 가중치(영향력) 확인
    learned_weights = meta_model.coef_[0]
    weights_pct = (np.exp(learned_weights) / np.sum(np.exp(learned_weights))) * 100

    print("\n" + "="*50)
    print(f"📈 [검증 완료] 대회 제출 전 예상 AUC 스코어: {expected_auc:.6f}") # 🚀 예상점수 출력!
    print("="*50)
    
    print("\n" + "="*50)
    print("🏆 [스태킹 메타 모델이 부여한 최종 가중치]")
    # 메타 모델이 학습한 실제 모델별 가중치(영향력) 확인
    learned_weights = meta_model.coef_[0]
    weights_pct = (np.exp(learned_weights) / np.sum(np.exp(learned_weights))) * 100

    print("\n" + "="*50)
    print("🏆 [스태킹 메타 모델이 부여한 최종 가중치]")
    print(f" ▶ LightGBM   : {weights_pct[0]:.1f}%")
    print(f" ▶ CatBoost   : {weights_pct[1]:.1f}%")
    print(f" ▶ XGBoost    : {weights_pct[2]:.1f}%")
    print(f" ▶ ExtraTrees : {weights_pct[3]:.1f}%")
    print(f" ▶ NeuralNet  : {weights_pct[4]:.1f}%")
    print("="*50)

    # ==========================================
    # 🏁 [5/6] 최종 제출 파일 생성
    # ==========================================
    print("\n🏁 [5/6] 최종 스태킹 예측을 수행합니다...")
    final_predictions = meta_model.predict_proba(OOF_test)[:, 1]

    submission = pd.DataFrame({'ID': test_id, 'probability': final_predictions})
    file_name = 'simulator_stacking_ensemble1111.csv'
    submission.to_csv(file_name, index=False)
    
    print(f"\n🎉 모든 세팅 적용 완료! '{file_name}' 파일이 저장되었습니다.")

if __name__ == "__main__":
    main()