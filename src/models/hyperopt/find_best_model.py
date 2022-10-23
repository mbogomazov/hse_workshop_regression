from hpsklearn import HyperoptEstimator
from hpsklearn import any_regressor, any_preprocessing
from hyperopt import tpe
from src.models.hyperopt.pipelines import *
from sklearn.model_selection import *
from src.config import *


def find_best_model_by_hyperopt(train_data, train_target):
    X = preprocess_pipe.fit_transform(train_data)

    model = HyperoptEstimator(
        regressor=any_regressor('reg'),
        preprocessing=any_preprocessing('pre'),
        algo=tpe.suggest,
        max_evals=max_evals,
        trial_timeout=trial_timeout,
    )

    # perform the search
    model.fit(X, train_target)

    return model.best_model()['learner']
