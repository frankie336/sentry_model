# OLD_config.py

import os

# Helper function to create paths
def create_path(base, subpath):
    return os.path.abspath(os.path.join(base, subpath))

# Base Directories
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
NEW_DATA_DIR = '/media/primethanos/sql/ForexMastermind/ML/data'
UNSEEN_DATA_DIR = '/media/primethanos/sql/ForexMastermind/ML/unseen_data'

MODEL_DIR = '/media/primethanos/sql/ForexMastermind/ML/models'
SCALER_DIR = '/media/primethanos/sql/ForexMastermind/ML/scaler'
PLOT_DIR = '/media/primethanos/sql/ForexMastermind/ML/models/plots'
DATA_DIR = NEW_DATA_DIR
SQL_DIR = '/media/primethanos/sql/ForexMastermind/ML'

# File Paths
PATH_TO_MIN_DATA = create_path(DATA_DIR, 'ForexData_20180101_to_20231121_1Min_v3.0.1.csv')
PATH_TO_MIN_UNSEEN_DATA = create_path(UNSEEN_DATA_DIR, 'ForexData_20231120_to_20231124_1Min_v3.0.1.csv')

PATH_TO_MIN_DATA_NEXT = create_path(DATA_DIR, 'ForexData_20180101_to_20231121_1Min_v5.csv')
PATH_TO_MIN_UNSEEN_DATA_NEXT = create_path(UNSEEN_DATA_DIR, 'ForexData_20231120_to_20231124_1Min_v5.csv')

PATH_TO_NEWS_DATA = create_path(DATA_DIR, 'ForexNewsSentiment_20180101_to_20231121_1Min_v3.csv')
PATH_TO_UNSEEN_NEWS_DATA = create_path(UNSEEN_DATA_DIR, 'ForexNewsSentiment_20231120_to_20231124_1Min_v3.csv')

PATH_TO_HOUR_DATA = create_path(DATA_DIR, 'seen/Merged_EURUSD_Candlestick_1_Hour_BID_01.01.2019-30.11.2021.csv')
PATH_TO_TEST_LIB = '/media/primethanos/sql/ForexMastermind/ML/processed_data_chunks/Merged_EURUSD_MIN/temp_sequence_28000_to_28999_processed.joblib'
PATH_TO_TEST_LIB2 = '/media/primethanos/sql/ForexMastermind/ML/processed_data_chunks/Merged_EURUSD_MIN/temp_sequence_27000_to_27999_processed.joblib'


# Generate Paths
PATHS = {
    'MIN': {
        'TRAINING_DATA': PATH_TO_MIN_DATA, 'UNSEEN_TRAINING_DATA': PATH_TO_MIN_UNSEEN_DATA,
        'NEWS_DATA': PATH_TO_NEWS_DATA, 'UNSEEN_NEWS_DATA': PATH_TO_UNSEEN_NEWS_DATA,
        'NEXT_TRAINING_DATA': PATH_TO_MIN_DATA_NEXT, 'UNSEEN_NEXT_TRAINING_DATA': PATH_TO_MIN_UNSEEN_DATA_NEXT,
        'MODEL_DIR': MODEL_DIR, 'PLOT_DIR': PLOT_DIR, 'SCALER_DIR': SCALER_DIR
    },
    'HOUR': {
        'TRAINING_DATA': PATH_TO_HOUR_DATA,
        'MODEL_DIR': MODEL_DIR
    }
}


# Logs
PATH_TO_LOGS = create_path(BASE_DIR, 'logs/results.log')
PATH_WRITER_LOGS = create_path(BASE_DIR, 'logs')








