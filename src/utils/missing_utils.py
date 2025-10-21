__all__ = ['missing_info']
import pandas as pd


def missing_info(df: pd.DataFrame):
    missing = df.isnull().sum() / df.shape[0]
    missing = missing[missing > 0]
    missing.sort_values(inplace=True)
    
    return missing