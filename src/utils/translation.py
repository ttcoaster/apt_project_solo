__all__ = ['translation_log', 'multi_translation_log', 'categorize_data', 'haversine_distance']
import math
import numpy as np
import pandas as pd


# 이거 고민해봐야하는게, 이렇게 하면, 속도라던가, 메모리라던가, 문제가 없으려나??
def translation_log(df:pd.DataFrame, col_name:str):
    """
    로그 변환 함수
    :param df: DataFrame
    :param col_name: 변환할 컬럼 이름
    :return: 변환된 DataFrame
    """
    df_copy = df.copy()
    return np.log1p(df_copy.loc[:, col_name])


def multi_translation_log(df:pd.DataFrame, cols:'list[str]'):
    df_copy = df.copy()
    for col in cols:
        df_copy.loc[:, col] = translation_log(df, col)
    
    return df_copy


def categorize_data(df: pd.DataFrame, col: str, bins: 'list[int]', labels: 'list[str]') -> pd.Series:
    """
        특정 변수를 범주화시키는 함수
        Arguments:
        - df: 데이터
        - col: 사용하려는 컬럼
        - bins: 어떤 범위로 나눌 것인지
        - labels: 나눈 범위의 이름들
    """
    return pd.cut(df[col], bins=bins, labels=labels).astype(str)


# 위경도를 이용해 두 지점간의 거리를 구하는 함수를 생성합니다.
def haversine_distance(lat1, lon1, lat2, lon2):
    radius = 6371.0

    lat1 = math.radians(lat1)
    lon1 = math.radians(lon1)
    lat2 = math.radians(lat2)
    lon2 = math.radians(lon2)

    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    distance = radius * c
    return distance