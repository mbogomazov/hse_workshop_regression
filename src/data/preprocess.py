import pandas as pd
import numpy as np
from src.config import *
import datetime as dt


# use it only for generate cols for params.yaml
def get_type_cols(df: pd.DataFrame) -> list:
    REAL_COLS = []
    CAT_COLS = []
    results = df.dtypes
    for el in results.index:
        if results[el] == np.int64 or results[el] == np.float64:
            REAL_COLS.append(el)
        elif results[el] == object:
            CAT_COLS.append(el)
    return REAL_COLS, CAT_COLS


def fill_na(df: pd.DataFrame) -> pd.DataFrame:
    df[CAT_COLS] = df[CAT_COLS].fillna('None')
    df = df.fillna(0)
    return df


def set_idx(df: pd.DataFrame, idx_col: str) -> pd.DataFrame:
    df = df.set_index(idx_col)
    return df


def cast_types(df: pd.DataFrame) -> pd.DataFrame:
    df[CAT_COLS] = df[CAT_COLS].astype('category')
    df[REAL_COLS] = df[REAL_COLS].astype(np.int64)
    return df


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    df = set_idx(df, ID_COL)
    df = fill_na(df)
    df = cast_types(df)
    return df
