import pandas as pd
import os

# ==========================================
# 1. 파일 이름 설정 (Seed 앙상블 지원)
# ==========================================
# 💡 시드 앙상블: ML 코드(CatBoost/LightGBM)에서 random_state(시드)만 
# 42, 2024, 1234 등으로 바꿔서 뽑은 CSV 파일들의 이름을 리스트 안에 모두 적어주세요!
# (만약 지금 1개밖에 없다면 그냥 1개만 둬도 문제없이 돌아갑니다.)
ml_files = [
    'final_unified_submission.csv',  # 시드 42로 뽑은 메인 파일
    'ml_seed_2024.csv',            # 추가 시드 파일이 있다면 주석(#) 지우고 파일명 입력
    'ml_seed_1234.csv',
    'ml_seed_777.csv'
]

dl_file = 'dl_baseline_submission.csv'

print(f"🔄 [1/3] 앙상블을 준비합니다...")
for f in ml_files + [dl_file]:
    if not os.path.exists(f):
        print(f"\n❌ [에러] '{f}' 파일을 찾을 수 없습니다! 폴더를 확인해주세요.")
        exit()

# ==========================================
# 2. 머신러닝 Seed Averaging (시드 평균)
# ==========================================
print(f"\n🌱 [2/3] 총 {len(ml_files)}개의 머신러닝 예측값을 하나로 합칩니다(Seed Averaging)...")
ml_base = pd.read_csv(ml_files[0]) # 뼈대
ml_probs = 0.0

for f in ml_files:
    temp_df = pd.read_csv(f)
    ml_probs += temp_df['probability']

# 다 더한 확률값을 파일 개수로 나눠서 평균(초강력 ML 확률)을 만듭니다.
ml_base['probability'] = ml_probs / len(ml_files)

# ==========================================
# 3. Rank Averaging (등수 평균) 블렌딩
# ==========================================
dl_sub = pd.read_csv(dl_file)
ml_weight = 0.8
dl_weight = 0.2

print(f"\n🧠 [3/3] 랭크 평균(Rank Averaging)으로 황금 비율({ml_weight*100}% : {dl_weight*100}%) 블렌딩을 진행합니다...")

final_sub = ml_base.copy()

# 🚀 0.0005점 상승의 핵심 비밀: 확률값을 그대로 더하지 않고, 등수(퍼센티지)로 변환!
ml_rank = ml_base['probability'].rank(pct=True)
dl_rank = dl_sub['probability'].rank(pct=True)

# 등수끼리 곱해서 최종 점수 산출
final_sub['probability'] = (ml_rank * ml_weight) + (dl_rank * dl_weight)

# ==========================================
# 4. 최종 파일 저장
# ==========================================
output_name = 'final_rank_seed_ensemble.csv'
final_sub.to_csv(output_name, index=False)

print(f"\n🎉 앙상블 완료! 1등을 잡기 위한 최종 병기 '{output_name}' 파일이 생성되었습니다! 🚀")