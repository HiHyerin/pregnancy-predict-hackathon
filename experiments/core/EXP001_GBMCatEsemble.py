import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
from catboost import CatBoostClassifier, Pool
from scipy.optimize import minimize
import optuna
import warnings

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

def load_and_preprocess(train_path='train.csv', test_path='test.csv'):
    """데이터 로드 및 파생 변수 생성을 한 번에 수행하는 내부 함수"""
    print("📦 [1/6] 데이터를 불러오고 전처리를 시작합니다...")
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    y_train = train['임신 성공 여부']
    X_train = train.drop(columns=['ID', '임신 성공 여부'])
    X_test = test.drop(columns=['ID'])
    test_id = test['ID']

    def engineer_features(df):
        df_copy = df.copy()
        
        # 1. '회' 글자 삭제
        for col in df_copy.columns:
            if df_copy[col].dtype == 'object' and df_copy[col].astype(str).str.contains('회').any():
                df_copy[col] = df_copy[col].astype(str).str.replace('회', '')
                df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce')
        
        # 2. 결측치 0 처리
        calc_cols = ['총 출산 횟수', '총 생성 배아 수', '이식된 배아 수', '수집된 신선 난자 수', '총 임신 횟수', '총 시술 횟수']
        for col in calc_cols:
            if col in df_copy.columns:
                df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce').fillna(0)
        
        # 3. 파생 변수 5종
        if '총 출산 횟수' in df_copy.columns:
            df_copy['출산_경험_여부'] = (df_copy['총 출산 횟수'] > 0).astype(int)
        if '이식된 배아 수' in df_copy.columns and '총 생성 배아 수' in df_copy.columns:
            df_copy['배아_이식_비율'] = np.where(df_copy['총 생성 배아 수'] > 0, df_copy['이식된 배아 수'] / df_copy['총 생성 배아 수'], 0)
        if '총 생성 배아 수' in df_copy.columns and '수집된 신선 난자 수' in df_copy.columns:
            df_copy['난자_대비_배아_비율'] = np.where(df_copy['수집된 신선 난자 수'] > 0, df_copy['총 생성 배아 수'] / df_copy['수집된 신선 난자 수'], 0)
        if '총 임신 횟수' in df_copy.columns and '총 시술 횟수' in df_copy.columns:
            df_copy['과거_임신_성공률'] = np.where(df_copy['총 시술 횟수'] > 0, df_copy['총 임신 횟수'] / df_copy['총 시술 횟수'], 0)
        
        cause_cols = [col for col in df_copy.columns if '불임 원인 -' in col]
        if cause_cols:
            df_copy['총_불임_원인_수'] = df_copy[cause_cols].sum(axis=1)

        # 🌟 4. 의학 칼럼 기반 슈퍼 파생 변수: 반복착상실패(RIF) 의심 환자군
        if all(col in df_copy.columns for col in ['총 시술 횟수', '이식된 배아 수', '총 임신 횟수']):
            df_copy['반복착상실패_의심여부'] = np.where(
                (df_copy['총 시술 횟수'] >= 2) & 
                (df_copy['이식된 배아 수'] > 0) & 
                (df_copy['총 임신 횟수'] == 0), 
                1, 0
            )
                
        cat_cols = df_copy.select_dtypes(include=['object', 'category']).columns.tolist()
        return df_copy, cat_cols

    X_train, cat_cols = engineer_features(X_train)
    X_test, _ = engineer_features(X_test)
    
    # 5. 라벨 인코딩 (Label Encoding)
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
    # ==========================================
    # 1. 통합된 함수로 데이터 불러오기
    # ==========================================
    X_train_int, X_test_int, X_train_cat, X_test_cat, y_train, test_id, cat_cols = load_and_preprocess()

    # ==========================================
    # 2. 하위 20개 쓰레기 변수 삭제 (Feature Selection)
    # ==========================================
    print("\n✂️ [2/6] 하위 20개 쓰레기 변수 삭제 중...")
    temp_model = lgb.LGBMClassifier(n_estimators=150, random_state=42, n_jobs=-1)
    temp_model.fit(X_train_cat, y_train)
    
    importance_df = pd.DataFrame({'feature': X_train_cat.columns, 'importance': temp_model.feature_importances_}).sort_values('importance', ascending=False)
    features_to_drop = importance_df.tail(20)['feature'].tolist()
    
    X_train_cat = X_train_cat.drop(columns=features_to_drop)
    X_test_cat = X_test_cat.drop(columns=features_to_drop)
    X_train_int = X_train_int.drop(columns=features_to_drop)
    X_test_int = X_test_int.drop(columns=features_to_drop)
    cat_cols_remaining = [col for col in cat_cols if col not in features_to_drop]
    print(f"✨ 변수 정리 완료! (남은 알짜배기 변수 {X_train_cat.shape[1]}개)")

    N_SPLITS = 5
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

    # ==========================================
    # 3. Optuna: LightGBM 최적화 (새 데이터 맞춤)
    # ==========================================
    print("\n🤖 [3/6] Optuna가 LightGBM 최적 파라미터를 찾는 중...")
    def lgb_objective(trial):
        params = {
            'n_estimators': 300,
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 20, 100),
            'max_depth': trial.suggest_int('max_depth', 4, 10),
            'min_child_samples': trial.suggest_int('min_child_samples', 20, 100),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'random_state': 42,
            'n_jobs': -1
        }
        cv_scores = []
        for tr_idx, val_idx in skf.split(X_train_cat, y_train):
            X_tr, y_tr = X_train_cat.iloc[tr_idx], y_train.iloc[tr_idx]
            X_va, y_va = X_train_cat.iloc[val_idx], y_train.iloc[val_idx]
            model = lgb.LGBMClassifier(**params)
            model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], eval_metric='auc', callbacks=[lgb.early_stopping(30, verbose=False)])
            cv_scores.append(roc_auc_score(y_va, model.predict_proba(X_va)[:, 1]))
        return np.mean(cv_scores)

    study_lgb = optuna.create_study(direction='maximize')
    study_lgb.optimize(lgb_objective, n_trials=20) 
    best_lgb_params = study_lgb.best_params
    best_lgb_params.update({'n_estimators': 1000, 'random_state': 42})

    # ==========================================
    # 4. Optuna: CatBoost 최적화 (새 데이터 맞춤 + GPU 가동)
    # ==========================================
    print("\n🤖 [4/6] Optuna가 CatBoost(GPU) 최적 파라미터를 찾는 중...")
    def cat_objective(trial):
        params = {
            'iterations': 300,
            'learning_rate': trial.suggest_float('learning_rate', 0.02, 0.1, log=True),
            'depth': trial.suggest_int('depth', 4, 8),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10, log=True),
            'random_strength': trial.suggest_float('random_strength', 0.1, 1.0),
            'eval_metric': 'AUC',
            'random_seed': 42,
            'verbose': False,
            'task_type': 'GPU' 
        }
        cv_scores = []
        for tr_idx, val_idx in skf.split(X_train_int, y_train):
            X_tr, y_tr = X_train_int.iloc[tr_idx], y_train.iloc[tr_idx]
            X_va, y_va = X_train_int.iloc[val_idx], y_train.iloc[val_idx]
            train_pool = Pool(X_tr, y_tr, cat_features=cat_cols_remaining)
            val_pool = Pool(X_va, y_va, cat_features=cat_cols_remaining)
            model = CatBoostClassifier(**params)
            model.fit(train_pool, eval_set=val_pool, early_stopping_rounds=30)
            cv_scores.append(roc_auc_score(y_va, model.predict_proba(val_pool)[:, 1]))
        return np.mean(cv_scores)

    study_cat = optuna.create_study(direction='maximize')
    study_cat.optimize(cat_objective, n_trials=15)
    best_cat_params = study_cat.best_params
    best_cat_params.update({'iterations': 1000, 'eval_metric': 'AUC', 'random_seed': 42, 'task_type': 'GPU'})

    # ==========================================
    # 5. 최적화된 새로운 파라미터로 최종 실전 학습
    # ==========================================
    print("\n🚀 [5/6] 새로 찾은 파라미터로 최종 실전 학습을 진행합니다!")
    oof_lgb = np.zeros(len(X_train_cat))
    oof_cat = np.zeros(len(X_train_cat))
    test_preds_lgb = np.zeros(len(X_test_cat))
    test_preds_cat = np.zeros(len(X_test_cat))

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_int, y_train)):
        print(f"--- Final Fold {fold + 1} / {N_SPLITS} ---")
        
        # LGBM
        X_tr_cat, X_va_cat = X_train_cat.iloc[train_idx], X_train_cat.iloc[val_idx]
        y_tr, y_va = y_train.iloc[train_idx], y_train.iloc[val_idx]
        model_lgb = lgb.LGBMClassifier(**best_lgb_params, n_jobs=-1)
        model_lgb.fit(X_tr_cat, y_tr, eval_set=[(X_va_cat, y_va)], eval_metric='auc', callbacks=[lgb.early_stopping(50, verbose=False)])
        oof_lgb[val_idx] = model_lgb.predict_proba(X_va_cat)[:, 1]
        test_preds_lgb += model_lgb.predict_proba(X_test_cat)[:, 1] / N_SPLITS

        # CatBoost
        X_tr_int, X_va_int = X_train_int.iloc[train_idx], X_train_int.iloc[val_idx]
        train_pool = Pool(X_tr_int, y_tr, cat_features=cat_cols_remaining)
        val_pool = Pool(X_va_int, y_va, cat_features=cat_cols_remaining)
        model_cat = CatBoostClassifier(**best_cat_params, verbose=False)
        model_cat.fit(train_pool, eval_set=val_pool, early_stopping_rounds=50)
        oof_cat[val_idx] = model_cat.predict_proba(val_pool)[:, 1]
        test_preds_cat += model_cat.predict_proba(X_test_int)[:, 1] / N_SPLITS

    # ==========================================
    # 6. 황금 비율 계산 및 최종 제출
    # ==========================================
    print("\n" + "="*45)
    print("⚖️ [6/6] AI가 새로운 데이터셋에 최적화된 앙상블 비율을 재계산합니다...")
    def auc_objective(w):
        mixed_pred = w[0] * oof_lgb + (1 - w[0]) * oof_cat
        return -roc_auc_score(y_train, mixed_pred)

    res = minimize(auc_objective, [0.5], bounds=[(0.0, 1.0)], method='L-BFGS-B')
    best_w_lgb = res.x[0]
    best_w_cat = 1.0 - best_w_lgb
    best_auc = -res.fun 

    print(f"🌟 [통합 코드 풀튜닝 완료] 새로운 최적 가중치 발견!!")
    print(f"   ▶ LightGBM 비중 : {best_w_lgb * 100:.2f}%")
    print(f"   ▶ CatBoost 비중 : {best_w_cat * 100:.2f}%")
    print(f"🏆 최종 앙상블 AUC Score: {best_auc:.5f}")
    print("="*45)

    final_test_preds = (best_w_lgb * test_preds_lgb) + (best_w_cat * test_preds_cat)
    submission = pd.DataFrame({'ID': test_id, 'probability': final_test_preds})
    submission.to_csv('final_unified_submission.csv', index=False)
    print("\n🎉 조장님 제출용 'final_unified_submission.csv' 파일이 저장되었습니다!")

if __name__ == "__main__":
    main()