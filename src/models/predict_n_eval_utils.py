from sklearn.metrics import *
import numpy as np
import json


def save_metrics_to_json(y_true, y_pred, file_name,):
    metrics = {
        'Explained variance': explained_variance_score(y_true, y_pred),
        'R2 score':  r2_score(y_true, y_pred),
        'MSE': mean_squared_error(y_true, y_pred),
    }

    with open(file_name, 'w') as f:
        f.write(json.dumps(metrics))
