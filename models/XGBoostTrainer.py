__all__ = ['XGBoostTrainer']
import numpy as np
from sklearn.metrics import mean_squared_error
from .BaseTrainer import BaseTrainer
import xgboost as xgb
from xgboost import Booster
from util.util_function import SEED


class XGBoostTrainer(BaseTrainer):
    """ XGBoost를 활용한 모델 학습 및 평가 클래스 """

    def __init__(self, params=None, num_boost_round=100, early_stopping_rounds=50, enable_categorical=False, verbose=True):
        """
            XGBoost 학습을 위한 클래스
            - params: XGBoost 하이퍼파라미터 딕셔너리
            - early_stopping_rounds: 조기 종료 기준
            - enable_categorical: 범주형 변수 사용 여부
            - verbose: 학습 과정 출력 여부
        """
        self.params = params if params else self.default_params()
        self.num_boost_round = num_boost_round
        self.early_stopping_rounds = early_stopping_rounds
        self.enable_categorical = enable_categorical
        self.verbose = verbose
        self.model: Booster = None
        self.evals_result = {}
        
    def default_params(self):
        """ 기본 하이퍼파라미터 설정 """
        return {
            "eval_metric": "rmse",
            "random_state": SEED,
            "learning_rate": 0.1,
            "max_depth": 3,
            "subsample": 0.8, # 샘플 사용 비율 과적합 방지용. default=1.0
            "colsample_bytree": 0.8,
        
        }
        
    def save_model_info(self):
        # feature importance 값 저장
        self.features_importance = self.model.get_score(importance_type="gain").values()
        self.features_name = self.model.get_score(importance_type="gain").keys()

    def fit(self, X_train, y_train, X_val, y_val):
        """ 
            모델 학습
            Args:
                X_train: 훈련 데이터
                y_train: 훈련 데이터의 타겟
                X_val: 검증 데이터
                y_val: 검증 데이터의 타겟
            Returns:
                None
        """
        train_data = xgb.DMatrix(X_train, label=y_train, enable_categorical=self.enable_categorical)
        val_data = xgb.DMatrix(X_val, label=y_val, enable_categorical=self.enable_categorical)

        self.model = xgb.train(self.params, train_data, num_boost_round=self.num_boost_round,
                               evals=[(train_data, "train"), (val_data, "val")],
                               early_stopping_rounds=self.early_stopping_rounds,
                               verbose_eval=self.verbose,
                               evals_result=self.evals_result)


    def predict(self, X_test):
        """ 예측 수행 """
        if not self.model:
            raise ValueError("먼저 학습을 진행해주세요!")
        dtest = xgb.DMatrix(X_test, enable_categorical=getattr(self, 'enable_categorical', False))
        raw_predictions = self.model.predict(dtest)
        return np.expm1(raw_predictions) # 로그 변환된 예측값을 다시 원래 값으로 변환

    def evaluate(self, X_test, y_test):
        """ 모델 평가 """
        y_pred = self.predict(X_test)
        y_test_original = np.expm1(y_test)  # 실제 Validation 데이터도 로그 변환을 풀어줘야 함
        rmse = np.sqrt(mean_squared_error(y_test_original, y_pred))
        print(f'Test 데이터 RMSE: {rmse}')
        return rmse
