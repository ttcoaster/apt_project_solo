__all__ = ['set_seed', 'split_columns_by_type', 'SEED', 'SQUARE_FOOTAGE']
import os
import random
import numpy as np
import pandas as pd


# util const
SEED = 1
SQUARE_FOOTAGE = 305785


# 시드 고정
def set_seed(seed: int = SEED) -> None:
    """시드 고정"""
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    print(f"시드 고정 완료: {seed}")
    
    
def split_columns_by_type(df: pd.DataFrame) -> 'tuple[list[str], list[str]]':
    continuous_columns = []
    categorical_columns = []

    for column in df.columns:
        if pd.api.types.is_numeric_dtype(df[column]):
            continuous_columns.append(column)
        else:
            categorical_columns.append(column)

    return continuous_columns, categorical_columns