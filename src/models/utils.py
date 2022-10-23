import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
from src.models.hyperopt.pipelines import *
from sklearn.utils import indexable, _safe_indexing
from sklearn.utils.validation import _num_samples
from sklearn.model_selection._split import _validate_shuffle_split
from sklearn.model_selection import *
from itertools import chain
from src.config import *


def extract_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    df, target = df.drop(TARGET_COL, axis=1), df[TARGET_COL]
    return df, target
