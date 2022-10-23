# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from src.features.features import gen_features
from src.utils import save_as_pickle
from src.config import *
import pandas as pd


def main():
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('Generate features')

    train = pd.read_pickle(preprocessed_train_data_pkl)
    test = pd.read_pickle(preprocessed_test_data_pkl)

    train = gen_features(train)
    test = gen_features(test)

    save_as_pickle(train, featurized_train_data_pkl)
    save_as_pickle(test, featurized_test_data_pkl)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
