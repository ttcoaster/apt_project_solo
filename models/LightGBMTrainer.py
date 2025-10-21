__all__ = ['LightGBMTrainer']
import lightgbm as lgb
from lightgbm import Booster
from sklearn.metrics import mean_squared_error
import numpy as np
from .BaseTrainer import BaseTrainer # BaseTrainer를 상속받는 클래스
from util.util_function import SEED


class LightGBMTrainer(BaseTrainer):
    """ LightGBM 모델 학습을 위한 Trainer 클래스 """
    def __init__(self, params=None, num_boost_round=500, early_stopping_rounds=50, log_period=0, categorical=None, verbose=True):
        """
        LightGBM 학습을 위한 클래스
        - params: LightGBM 하이퍼파라미터 딕셔너리
        - num_boost_round: 부스팅 라운드 횟수
        - early_stopping_rounds: 조기 종료 기준
        - log_period: 로그 기록 주기
        - categorical: 범주형 변수 리스트
        - verbose: 학습 과정 출력 여부
        """
        self.params = params if params else self.default_params()
        self.num_boost_round = num_boost_round
        self.early_stopping_rounds = early_stopping_rounds
        self.log_period = log_period
        self.categorical = categorical
        self.verbose = verbose
        self.evals_result = {}
        self.setting_callback_function()
        self.model:Booster = None

    def default_params(self):
        """ 기본 하이퍼파라미터 설정 """
        return {
            'seed': SEED,
            'objective': 'regression',
            'metric': 'rmse',
            'learning_rate': 0.1,
            'num_leaves': 80,
            'boosting_type': 'gbdt',
            'min_data_in_leaf': 20,
            'max_depth': -1,
            'feature_fraction': 0.8
        }
        
    def setting_callback_function(self):
        """ 콜백 함수 설정 """
        self.callbacks = [
            lgb.early_stopping(self.early_stopping_rounds, verbose=self.verbose),
            lgb.log_evaluation(period=self.log_period),
            lgb.record_evaluation(eval_result=self.evals_result)
        ]
        
    def save_model_info(self):
        self.features_importance = self.model.feature_importance("gain")
        self.features_name = self.model.feature_name()
        

    def fit(self, X_train, y_train, X_val, y_val):
        """ 모델 학습 """
        train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=self.categorical)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data, categorical_feature=self.categorical)

        self.model = lgb.train(self.params, train_data, num_boost_round=self.num_boost_round,
                               valid_sets=[train_data, val_data], valid_names=['train', 'val'],
                               callbacks=self.callbacks)

    def predict(self, X_test):
        """ 예측 수행 """
        if not self.model:
            raise ValueError("먼저 학습을 진행해주세요!")
        return np.expm1(self.model.predict(X_test)) # 로그 변환된 예측값을 다시 원래 값으로 변환

    def evaluate(self, X_test, y_test):
        """ RMSE 평가 수행 """
        y_pred = self.predict(X_test)
        y_test_original = np.expm1(y_test)  # 실제 Validation 데이터도 로그 변환을 풀어줘야 함
        rmse = np.sqrt(mean_squared_error(y_test_original, y_pred))
        print(f'Test 데이터 RMSE: {rmse}')
        return rmse