import pandas as pd
import os

# ==========================================
# 1. 합칠 두 개의 파일 이름 설정
# ==========================================
# 💡 주의: 아래 'best_ml.csv' 자리에 본인이 예전에 0.74점 이상을 냈던 
# CatBoost/LightGBM 결과 파일의 정확한 이름을 적어주세요!
ml_file = 'final_unified_submission.csv'  
dl_file = 'dl_baseline_submission.csv'

print(f"🔄 [1/3] 앙상블을 준비합니다...")
print(f"   ▶ 머신러닝 파일: {ml_file}")
print(f"   ▶ 딥러닝 파일: {dl_file}")

if not os.path.exists(ml_file) or not os.path.exists(dl_file):
    print("\n❌ [에러] 파일을 찾을 수 없습니다! 두 파일이 현재 폴더에 있는지 확인해주세요.")
    exit()

ml_sub = pd.read_csv(ml_file)
dl_sub = pd.read_csv(dl_file)

# ==========================================
# 2. 황금 비율 설정 (Weighted Blending)
# ==========================================
# ML이 기본적으로 점수가 더 높으므로 가중치를 더 많이 줍니다.
# 가장 대중적인 8:2 비율로 섞어봅니다.
ml_weight = 0.8
dl_weight = 0.2

print(f"\n🧠 [2/3] 황금 비율({ml_weight*100}% : {dl_weight*100}%)로 두 인공지능의 뇌를 섞습니다...")

final_sub = ml_sub.copy()
# 확률값(probability)을 가중치만큼 곱해서 더해줍니다.
final_sub['probability'] = (ml_sub['probability'] * ml_weight) + (dl_sub['probability'] * dl_weight)

# ==========================================
# 3. 최종 파일 저장
# ==========================================
output_name = 'final_ensemble_submission.csv'
final_sub.to_csv(output_name, index=False)

print(f"\n🎉 [3/3] 앙상블 완료! 최종 결과물 '{output_name}' 파일이 생성되었습니다.")
print("이 파일을 대회 플랫폼에 제출해보세요. 0.74 벽을 뚫을 확률이 아주 높습니다! 🚀")