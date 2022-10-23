# -*- coding: utf-8 -*-
import logging
import joblib
import pandas as pd
from sklearn.pipeline import *
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from catboost import CatBoostRegressor
from src.config import *
from src.models.utils import extract_target
from sklearn.model_selection import train_test_split


def main():
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('Train Catboost model')

    train = pd.read_pickle(featurized_train_data_pkl)

    train, target = extract_target(train)

    train_data, val_data, train_target, val_target = train_test_split(
        train, target, train_size=train_test_split_train_size, random_state=seed)

    model = CatBoostRegressor(
        iterations=catboost_iterations,
        learning_rate=catboost_learning_rate,
        cat_features=CAT_COLS,
        eval_metric=catboost_eval_metric,
    )

    pipeline_castboost = Pipeline([
        ('model', model)])

    best_model = pipeline_castboost.fit(train_data, train_target)
    joblib.dump(best_model, catboost_best_model_file)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
