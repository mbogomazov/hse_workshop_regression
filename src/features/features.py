import pandas as pd
import numpy as np
from src.config import *


def gen_pool_availability(df: pd.DataFrame) -> pd.DataFrame:
    for index, row in df.iterrows():
        has_pool = 0 if row['PoolQC'] == 'None' else 1
        df.loc[index, HAS_POOL] = has_pool

    df[HAS_POOL] = df[HAS_POOL].astype(np.int32)
    return df


def gen_fence_availability(df: pd.DataFrame) -> pd.DataFrame:
    for index, row in df.iterrows():
        has_fence = 0 if row['Fence'] == 'None' else 1
        df.loc[index, HAS_FENCE] = has_fence

    df[HAS_FENCE] = df[HAS_FENCE].astype(np.int32)
    return df


def gen_features(df: pd.DataFrame) -> pd.DataFrame:
    df = gen_fence_availability(df)
    df = gen_pool_availability(df)
    return df
