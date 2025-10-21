# 아파트 실거래가 예측 경진대회
## 8조 ITelligence

| ![김주형](https://avatars.githubusercontent.com/u/156163982?v=4) | ![이재용](https://avatars.githubusercontent.com/u/156163982?v=4) | ![최지희](https://avatars.githubusercontent.com/u/156163982?v=4) | ![김재덕](https://avatars.githubusercontent.com/u/156163982?v=4) | ![김재덕](https://avatars.githubusercontent.com/u/156163982?v=4) |
| :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: |
|            [김주형](https://github.com/UpstageAILab)             |            [이재용](https://github.com/UpstageAILab)             |            [최지희](https://github.com/UpstageAILab)             |            [김재덕](https://github.com/UpstageAILab)             |            [강연경](https://github.com/UpstageAILab)             |
|                            팀장, 모델 하이퍼파라미터 조정                             |                            데이터 전처리/모델학습                             |                            데이터 전처리/모델학습                             |                            데이터 전처리/모델학습                             |                            데이터 전처리/모델학습                             |

## 0. Overview
### Environment
- AI Stages GPU Server
- VS Code
- Anaconda

### Requirements
- Machine Learning 프로세스에 익숙해지기

## 1. Competiton Info

### Overview

- 아파트 실거래가 예측 : 서울시 아파트 실거래가 매매 데이터를 기반으로 아파트 가격을 예측하는 대회

### Timeline

- 2025.05.01 : Start Date
- 2025.05.01 ~ 2025.05.11 : 개별 강의 수강 및 EDA
- 2025.05.12 : 데이터 분석 및 전처리 결과 공유
- 2025.05.13 : 코드 및 데이터 공유
- 2025.05.14 ~ 2025.05.15 : Best Score 데이터 및 방법론 적용
- 2025.05.15 : Final submission deadline

## 2. Components

### Directory

```
├── code
│   └── Step01.preprocessing.py
│   └── models
│       └── BaseTrainer.py
│       └── CatBoostTrainer.py
│       └── LightGBMTrainer.py
│       └── XGBoostTrainer.py
│   └── util
│       └── korean_matplot_setting.py
│       └── missing_utils.py
│       └── outlier_utils.py
│       └── translation.py
│       └── util_function.py
│       └── visualization.py
├── docs
│   ├── pdf
│   │   └── [패스트캠퍼스] AI 부트캠프 13기 ML경진대회 발표템플릿(ml-8조).pdf
```

## 3. Data descrption

### Dataset overview

- 데이터는 공공기관에서 제공한 아파트 실거래가 데이터를 기반으로 구성됨
- 주요 컬럼: `시군구`, `도로명`, `아파트명`, `전용면적`, `층`, `건축년도`, `계약년월`, `거래금액` 등

```
├── data
│   └── apt_loc_info_2511.pkl
│   └── apt_price_index.pkl
│   └── homLoanRate_20250511.pkl
```

### EDA

- 전용면적의 분포, 층 수, 연식, 거래 시기별 패턴 분석
- 이상치 및 결측치 탐색, 거래금액의 로그 분포 확인

### Data Processing

- **Data Cleaning**: 결측치 처리, 필요 없는 컬럼 삭제
- **Feature Engineering**:
    - 날짜 → 계약년, 계약월 분리
    - 전용면적 → 전용면적 구간 생성
    - 건축년도 → 연식 파생
    - 주소 → 위경도 변환 및 지하철거리/학교거리/편의시설 파생
- **Encoding & Scaling**: 범주형 라벨 인코딩, 수치형 스케일링, 거리/타겟값 로그 변환

## 4. Modeling

### Model descrition

- **XGBoost, LightGBM, CatBoost, Random Forest** 모델을 비교 분석
- 트리 기반 모델들은 비선형 데이터 처리에 강하고, 성능이 뛰어나며 feature importance 확인이 용이함
- LightGBM은 속도와 성능에서 균형이 좋고, 대규모 데이터셋에도 효과적이므로 주요 선택 모델로 고려됨
- CatBoost는 범주형 처리 자동화가 강점이며, XGBoost는 튜닝을 통해 안정적 성능 확보 가능

### Modeling Process

- **시계열 기반 K-Fold**를 적용해 시간 순서에 따라 데이터를 학습/검증 분리
- 각 모델을 5-Fold 방식으로 학습 후, 평균 성능을 비교 평가함
- 최종 모델은 전체 데이터로 다시 학습하여 테스트 데이터 예측 수행

### Fold별 성능 시각화

- 각 모델의 Fold별 RMSE, MAE, R2 Score를 시각화하여 비교 분석
  
## 5. Result

### Leader Board

 ![image](https://github.com/user-attachments/assets/42b85230-f49a-4651-8f7e-caa385fbe6a6)

### Presentation

- [link](https://docs.google.com/presentation/d/1b4Q7Tu1kr4uKn8NHG-qnMV_Qyjzd3iwCEZka-92mTiA/edit?usp=sharing)
