# Pregnancy Prediction Hackathon

난임 환자 데이터를 분석하여 임신 성공 여부를 예측하는 DACON 해커톤 프로젝트입니다.

단순히 리더보드 점수를 높이는 데 그치지 않고, 데이터 이해부터 검증, 모델링, Feature Engineering까지의 과정을 팀원이 함께 학습하고 재현할 수 있도록 만드는 것을 목표로 합니다.

## 대회 정보

- 대회: [난임 환자 대상 임신 성공 여부 예측 AI 온라인 해커톤](https://dacon.io/competitions/official/236733/overview/description)
- 예측 대상: `임신 성공 여부`
- 임신 성공의 정의: 출산까지 성공적으로 진행된 임신
- 문제 유형: 이진 분류
- 평가산식: ROC-AUC
- 평가 데이터: Public에서 Test 데이터 100% 사용
- 일일 제출 제한: 3회
- 외부 데이터: 사용 불가
- 사전 학습 모델: 사용 가능

## 데이터 구성

| 파일 | 크기 | 설명 |
|---|---:|---|
| `data/train.csv` | 256,351 × 69 | Feature, ID, target이 포함된 학습 데이터 |
| `data/test.csv` | 90,067 × 68 | target이 없는 평가 데이터 |
| `data/sample_submission.csv` | 90,067 × 2 | 제출 파일 형식 |
| `data/데이터 명세.xlsx` | - | Feature 설명 자료 |

- ID 컬럼: `ID`
- Target 컬럼: `임신 성공 여부`
- 제출 예측 컬럼: `probability`

대회 데이터는 Git으로 공유하지 않습니다. 각 팀원은 DACON에서 데이터를 내려받아 `data/`에 직접 저장해야 합니다.

## 검증 및 Leakage 원칙

기본 검증은 클래스 비율을 유지하는 `StratifiedKFold`를 사용하고, 공식 평가산식과 동일한 ROC-AUC로 OOF 성능을 측정합니다.

- 모든 실험은 가능하면 같은 fold와 random seed를 사용합니다.
- ROC-AUC 계산에는 `predict_proba()[:, 1]`의 임신 성공 확률을 사용합니다.
- 결측치 대체, 인코딩, 스케일링은 각 fold의 train 데이터로만 `fit`합니다.
- validation과 test에는 학습된 전처리기를 이용해 `transform`만 수행합니다.
- train과 test를 합쳐 전처리하거나 범주를 학습하지 않습니다.
- test의 통계량이나 분포를 모델 학습과 Feature Engineering 기준으로 사용하지 않습니다.
- test 데이터에 별도로 `pd.get_dummies()`를 적용하지 않습니다.

## 프로젝트 구조

```text
pregnancy-predict-hackathon/
├─ data/                         # 대회 원본 데이터, Git 추적 제외
│  ├─ train.csv
│  ├─ test.csv
│  ├─ sample_submission.csv
│  └─ 데이터 명세.xlsx
├─ docs/                         # 협업 규칙과 프로젝트 문서
│  └─ git_branch_전략.md
├─ experiments/                  # 모델링 실험 코드
│  ├─ core/                      # 학습 및 재사용 가치가 높은 주요 실험
│  │  └─ EXP001_DummyClassifier.py
│  └─ archive/                   # 중복되거나 제외한 실험
├─ reports/                      # 실험 점수와 분석 결과, Git 추적 제외
├─ submissions/                  # 제출용 CSV, Git 추적 제외
│  ├─ selected/                  # 실제 제출 후보
│  └─ archive/                   # 이전 또는 제외한 제출 파일
├─ .python-version               # 프로젝트 Python 버전
├─ pyproject.toml                # 직접 사용하는 패키지와 프로젝트 설정
├─ uv.lock                       # 전체 의존성 버전 잠금 파일
└─ README.md                     # 프로젝트 안내 문서
```

실험이 완료되면 검증된 코드를 최종 실행 파일로 통합하고, 마지막 단계에서 제출용 Notebook을 제작합니다. 초기 EDA와 모델링 실험은 `.py` 파일로 진행합니다.

## 폴더 및 파일 관리 규칙

### 실험 코드

```text
EXP001_DummyClassifier.py
EXP002_ModelName.py
EXP003_FeatureName.py
```

- 파일명은 `EXP번호_핵심변경사항.py` 형식을 사용합니다.
- 한 실험에서는 가능하면 하나의 핵심 요소만 변경합니다.
- 주요 실험은 `experiments/core/`에 저장합니다.
- 단순 실패나 중복 실험은 `experiments/archive/`로 이동합니다.

### 결과 및 제출 파일

```text
exp001_results.csv
oof_exp001_model_name.csv
submission_exp001_model_name.csv
```

결과 CSV, OOF 예측, 모델 파일, submission은 로컬에서 생성하며 Git에 올리지 않습니다. 주요 점수와 실험 해석은 팀 실험 로그에 별도로 기록합니다.

## 개발 환경 설정

이 프로젝트는 Python 3.11과 `uv`를 사용합니다.

### 최초 환경 구성

```bash
git clone <repository-url>
cd pregnancy-predict-hackathon
uv sync
```

`uv sync`를 실행하면 `.venv`가 생성되고 `uv.lock`에 기록된 패키지가 설치됩니다.

### 실험 실행

```bash
uv run python experiments/core/EXP001_DummyClassifier.py
```

### 패키지 추가 및 공유

```bash
uv add <package-name>
```

패키지를 변경하면 `pyproject.toml`과 `uv.lock`을 함께 커밋합니다. 다른 팀원은 변경 사항을 받은 뒤 `uv sync`를 실행합니다.

## 진행 순서

1. 주제 및 Feature 파악
2. 도메인 조사 및 관련 논문·공식 자료 탐색
3. 평가산식과 검증 전략 설계
4. 모델 후보 선정
5. 모델 학습 및 예측
6. Feature Engineering, 튜닝, 앙상블을 통한 모델 개선
7. 최종 재현 코드와 제출용 Notebook 작성

## 실험 원칙

- baseline을 먼저 만들고 이후 실험의 기준점으로 사용합니다.
- 모든 모델은 동일한 검증 조건에서 비교합니다.
- OOF 점수와 Public 점수를 구분해서 기록합니다.
- 좋은 결과뿐 아니라 실패한 실험과 원인도 기록합니다.
- 리더보드 점수만 보고 반복적으로 모델을 변경하지 않습니다.
- 최종 코드는 처음부터 실행해 같은 submission을 재현할 수 있어야 합니다.
