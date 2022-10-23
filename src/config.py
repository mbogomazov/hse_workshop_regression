import yaml

with open('params.yaml') as conf_file:
    config = yaml.safe_load(conf_file)

# Base
seed = config['base']['seed']

# Data load paths
train_csv = config['load_data']['data_train_path']
test_csv = config['load_data']['data_test_path']

# Pre-process category
TARGET_COL = config['preprocess_data']['cols']['TARGET_COL']
ID_COL = config['preprocess_data']['cols']['ID_COL']
CAT_COLS = config['preprocess_data']['cols']['CAT_COLS']
REAL_COLS = config['preprocess_data']['cols']['REAL_COLS']
# DATA_COLS = config['preprocess_data']['cols']['DATA_COLS']

# BASEMENT_COLS = config['preprocess_data']['cols']['BASEMENT_COLS']
# MASONRY_COLS = config['preprocess_data']['cols']['MASONRY_COLS']
# GARAGE_COLS = config['preprocess_data']['cols']['GARAGE_COLS']
# GARAGE_COL = config['preprocess_data']['cols']['GARAGE_COL']

# Data preprocess filepaths
preprocessed_train_data_pkl = config['preprocess_data']['paths']['preprocessed_train_data_pkl']
preprocessed_target_data_pkl = config['preprocess_data']['paths']['preprocessed_target_data_pkl']
preprocessed_test_data_pkl = config['preprocess_data']['paths']['preprocessed_test_data_pkl']

# Generate features step
YEAR_SOLD = config['generate_features']['cols']['YEAR_SOLD']
YEAR_BUILT = config['generate_features']['cols']['YEAR_BUILT']
BUILDING_AGE = config['generate_features']['cols']['BUILDING_AGE']

featurized_train_data_pkl = config['generate_features']['paths']['featurized_train_data_pkl']
featurized_test_data_pkl = config['generate_features']['paths']['featurized_test_data_pkl']

# Model train step
train_test_split_train_size = config['train_models']['train_test_split_train_size']
# Hyperopt
max_evals = config['train_models']['hyperopt']['max_evals']
trial_timeout = config['train_models']['hyperopt']['trial_timeout']
hyperopt_best_model_file = config['train_models']['paths']['hyperopt_best_model_path']

# Catboost
catboost_best_model_file = config['train_models']['paths']['catboost_best_model_path']
catboost_iterations = config['train_models']['catboost']['iterations']
catboost_learning_rate = config['train_models']['catboost']['learning_rate']
catboost_eval_metric = config['train_models']['catboost']['eval_metric']


# Predict and eval

# Hyperopt
hyperopt_metrics = config['predict_n_eval']['hyperopt']['metrics']
hyperopt_inference_predict = config['predict_n_eval']['hyperopt']['inference_predict']

# Catboost
catboost_metrics = config['predict_n_eval']['catboost']['metrics']
catboost_inference_predict = config['predict_n_eval']['catboost']['inference_predict']
