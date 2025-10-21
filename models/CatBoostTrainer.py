__all__ = ['CatBoostTrainer']
import numpy as np
import catboost as cb
from catboost import CatBoostRegressor
from .BaseTrainer import BaseTrainer
from sklearn.metrics import mean_squared_error
from util.util_function import SEED

class CatBoostTrainer(BaseTrainer):
    """ CatBoost를 활용한 회귀 모델 학습 및 평가 클래스 """

    def __init__(self, params=None, early_stopping_rounds=10, verbose:'bool | int'=True, categorical_features: list = None):
        """ 
            CatBoost 학습을 위한 클래스
            - params: CatBoost의 하이퍼파라미터 딕셔너리
            - early_stopping_rounds: 조기 종료 기준
            - verbose: 학습 과정 중 로그 출력 레벨 또는 빈도 제어
            - categorical_features: 범주형 특성 이름의 리스트
        """
        self.params = params if params else self.default_params()
        self.early_stopping_rounds = early_stopping_rounds
        self.verbose = verbose
        self.categorical_features = categorical_features
        self.model = CatBoostRegressor(**self.params)
        self.evals_result = {}
        
    def default_params(self):
        """ 기본 하이퍼파라미터 설정 """
        return {
            "iterations": 100, # 학습 횟수임.
            "learning_rate": 0.05,
            "depth": 6,
            "loss_function": "RMSE",
            "eval_metric": "RMSE", # 명시적으로 평가 지표 설정
            "random_seed": SEED,
            "use_best_model": True, # 조기 종료 시 최적 성능을 보인 모델을 자동으로 선택 기본값은 마지막 모델을 사용함.
        }
        
    def save_model_info(self):
        # evals 값 저장
        raw_eval_results = self.model.get_evals_result()
        self.evals_result['train'] = raw_eval_results.get('learn', {})
        self.evals_result['val'] = raw_eval_results.get('validation', {})
        

    def fit(self, X_train, y_train, X_val, y_val):
        """ 모델 학습 """

        train_data = cb.Pool(
            data=X_train, 
            label=y_train, 
            cat_features=self.categorical_features,
            feature_names=list(X_train.columns)
        )
        val_data = cb.Pool(
            data=X_val, 
            label=y_val, 
            cat_features=self.categorical_features,
            feature_names=list(X_val.columns)
        )
        self.model.fit(train_data, eval_set=val_data, early_stopping_rounds=self.early_stopping_rounds, verbose=self.verbose)
        

    def predict(self, X_test):
        """ 예측 수행 """
        return np.expm1(self.model.predict(X_test)) # 로그 변환된 예측값을 다시 원래 값으로 변환

    def evaluate(self, X_test, y_test):
        """ 모델 평가 (RMSE 계산) """
        y_pred = self.predict(X_test)
        y_test_original = np.expm1(y_test)  # 실제 Validation 데이터도 로그 변환을 풀어줘야 함
        rmse = np.sqrt(mean_squared_error(y_test_original, y_pred))
        print(f'Test 데이터 RMSE: {rmse}')
        return rmse