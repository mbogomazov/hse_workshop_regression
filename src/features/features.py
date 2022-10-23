import pandas as pd
import numpy as np
from src.config import *


def calc_building_age(df: pd.DataFrame) -> pd.DataFrame:
    for index, row in df.iterrows():
        building_age = int(row[YEAR_SOLD] - row[YEAR_BUILT])
        df.loc[index, BUILDING_AGE] = building_age

    df[BUILDING_AGE] = df[BUILDING_AGE].astype(np.int32)
    return df


def gen_features(df: pd.DataFrame) -> pd.DataFrame:
    # df = calc_building_age(df)
    return df
