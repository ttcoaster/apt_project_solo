__all__ = ["plot_histogram", "plot_scatter", "plot_outliers_boxplot", "plot_missing_data_var", "view_eval_result", "plot_feature_importances_general"]
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_histogram(data, column, bins=50):
    """ 히스토그램을 그리는 함수 """
    plt.figure(figsize=(10, 5))
    sns.histplot(data[column], bins=bins, alpha=0.7, kde=True)
    plt.title(f"Histogram of {column}")
    plt.show()
    

def plot_scatter(x, y, xlabel="X", ylabel="Y"):
    """ 산점도를 그리는 함수 """
    plt.figure(figsize=(8, 5))
    plt.scatter(x, y, alpha=0.5)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title("Scatter Plot")
    plt.show()
    

def plot_outliers_boxplot(x, y=None):
    """ 
    이상치 탐지를 위한 박스플롯을 그리는 함수 
    Arguments:
    - x: x축 데이터 (Series)
    - y: y축 데이터 (Series) (default: None)
    """
    plt.figure(figsize=(7, 3))
    sns.boxplot(x=x, y=y, color="lightgreen")
    plt.title(f"Boxplot of {x.name}")
    plt.xlabel("Area")
    plt.show()
    

def plot_missing_data_var(data):
    """ 
    변수별 결측치 비율을 시각화하는 함수 
    Arguments:
    - data: 데이터프레임
    """
    plt.figure(figsize=(13, 2))
    data.plot.bar( color='orange')
    plt.title("변수별 결측치 비율")
    plt.show()
    

def view_eval_result(evals_result, model_name="Model"):
    """
    학습 과정의 평가 지표를 시각화하는 함수.
    evals_result는 {'train': {'metric_name': [...]}, 'val': {'metric_name': [...]}} 형태이며,
    'train'과 'val' 키 및 해당 메트릭 데이터가 항상 존재하고 유효하다고 가정합니다.
    """
    # 사용자의 가정에 따라 'train' 데이터에서 첫 번째 메트릭 이름을 가져옵니다.
    # 이 메트릭 이름이 'val' 데이터에도 존재하며, 각 데이터 리스트가 비어있지 않다고 가정합니다.
    metric_name = list(evals_result['train'].keys())[0]
    
    # 전체 Figure 크기 설정 (두 개의 그래프를 양옆으로 배치하므로 가로 길이를 늘림)
    plt.figure(figsize=(16, 5))

    # 첫 번째 subplot (Train 데이터 그래프)
    plt.subplot(1, 2, 1) # 1행 2열의 첫 번째 위치
    plt.plot(evals_result['train'][metric_name], label=f"Train {metric_name.upper()}", marker="o", color='blue')
    plt.xlabel("Iteration")
    plt.ylabel(metric_name.upper())
    plt.title(f"{model_name} Training Progress - Train Data")
    plt.legend()
    plt.grid(True)

    # 두 번째 subplot (Validation 데이터 그래프)
    plt.subplot(1, 2, 2) # 1행 2열의 두 번째 위치
    plt.plot(evals_result['val'][metric_name], label=f"Validation {metric_name.upper()}", marker="s", color='orange')
    plt.xlabel("Iteration")
    plt.ylabel(metric_name.upper())
    plt.title(f"{model_name} Training Progress - Validation Data")
    plt.legend()
    plt.grid(True)

    plt.tight_layout() # subplot들이 겹치지 않도록 레이아웃 조정
    plt.show()
    
    
    
def plot_feature_importances_general(importances, feature_names, model_name_str, importance_type_str="Importance"):
    """
    피처 중요도를 시각화하는 범용 함수입니다.

    Args:
        importances (array-like): 피처 중요도 값 배열.
        feature_names (list of str): 피처 이름 리스트.
        model_name_str (str): 그래프 제목에 사용될 모델 이름.
        importance_type_str (str): 중요도 타입 (예: "Gain", "Weight", "PredictionValuesChange").
    """
    if len(importances) != len(feature_names):
        raise ValueError(f"중요도 값의 개수({len(importances)})와 피처 이름의 개수({len(feature_names)})가 일치하지 않습니다.")

    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    })

    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False).reset_index(drop=True)

    plt.figure(figsize=(12, max(6, len(feature_importance_df['Feature']) / 2.5)))
    sns.barplot(x='Importance', y='Feature', data=feature_importance_df, palette='viridis')
    plt.title(f'{model_name_str} Feature Importance', fontsize=16)
    plt.xlabel(f'Importance Score ({importance_type_str})', fontsize=14)
    plt.ylabel('Feature', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
