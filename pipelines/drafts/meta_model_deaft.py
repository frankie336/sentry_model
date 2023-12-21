import csv
import json
import os
import time
from datetime import datetime

# Third party imports
import joblib
import optuna
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from sklearn.metrics import (accuracy_score, auc, classification_report,
                             confusion_matrix, precision_recall_curve,
                             roc_curve, average_precision_score)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import StackingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
# Local application imports
from ForexMastermind.ML.config import ForexMastermindConfig
from ForexMastermind.ML.tools.tools import (SampleData, DataFrameSaver, ModelVersionService, Balance, DataLoader,
                                      ListDataFrameColumns)
from ForexMastermind.ML.collections import top_seven_features, top_seven_features_indexes
# Other imports
import hashlib
import re
from imblearn.over_sampling import SMOTE


class ModelTrainingService:
    def __init__(self, X, y, X_test, y_test,  config, model_version_service):
        self.X = X
        self.X_test = X_test
        self.y_test = y_test
        self.y = y
        self.config = config
        self.model_version_service = model_version_service

    def train_model(self):
        """
        Trains a stacked ensemble machine learning model.

        Returns:
            tuple: The trained ensemble machine learning model and its training time.
        """
        start_time = time.time()

        # Define base models
        estimators = [
            ('rf', RandomForestClassifier(n_estimators=10, random_state=42)),
            ('mlp', MLPClassifier(hidden_layer_sizes=(100, 100),
                                  activation='relu',
                                  solver='adam',
                                  alpha=0.0001,
                                  batch_size='auto',
                                  learning_rate='constant',
                                  learning_rate_init=0.001,
                                  max_iter=500,
                                  random_state=42)),
            ('xgb', XGBClassifier(use_label_encoder=False, eval_metric='logloss')),
            ('gbm', GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)),
            # ('ada', AdaBoostClassifier(n_estimators=100, random_state=42)),
            # ('svm', SVC(kernel='rbf', probability=True, random_state=42))  # SVM added here
        ]

        # Define the stacking ensemble with LogisticRegression as the final estimator
        model = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression(), verbose=True)

        # Train the ensemble model
        model.fit(self.X, self.y)

        # Calculate the training time
        training_time = time.time() - start_time

        # Save the trained model
        model_version = self.model_version_service.get_next_model_version()
        model_directory = self.config.get_model_directory()
        model_filename = f"IntraDayForexPredictor_v{model_version}.sav"
        model_path = os.path.join(model_directory, model_filename)
        joblib.dump(model, model_path)
        print(f"Ensemble model saved as {model_path}")

        return model, training_time, len(self.X)

class TrainingPipeline:
    def __init__(self, file_path, version=None):
        self.file_path = file_path
        self.version = version
        self.config = ForexMastermindConfig()
        self.model_version_service = ModelVersionService(self.config)

        self.data_preprocessor = DataPreprocessor(config=self.config,
                                                  model_version_service=self.model_version_service)

    def run(self, model_notes='None', sample_size=4):

        # Load and process data
        data_loader = DataLoader()
        unprocessed_training_data = data_loader.load_data(data_path=self.file_path)

        # Use this block to cut a subset of the total number of samples for training
        # You might want to cut a small percentage of very large sample sizes to
        # conserve computational resources, and or test concepts in a shorter time.
        # Alternatively, leave the default value which is 100 for all samples.
        sample_the_data = SampleData()
        sampled_training_data = sample_the_data.sample_data(n_percent=sample_size, data=unprocessed_training_data)

        (processed_training_data, features, X_train_res_scaled, y_train_res,
         X_test_scaled, y_test) = self.data_preprocessor.preprocess_data(
            unprocessed_training_data=sampled_training_data)

        print(processed_training_data.head())
        print(f'There are {len(processed_training_data)} samples in data about to be trained!')

        model_training = ModelTrainingService(X=X_train_res_scaled, X_test=X_test_scaled,
                                              y=y_train_res, y_test=y_test, config=self.config,
                                              model_version_service=self.model_version_service)

        # We need to return the model & training time here  for later use in evaluation
        model, training_time, fitted_samples_count = model_training.train_model()

        # Evaluate the model
        evaluation = ModelEvaluationService(model, X_test=X_test_scaled, y_test=y_test,
                                            config=self.config, training_time=training_time,
                                            feature_list=features, model_notes=model_notes,
                                            fitted_samples_count=fitted_samples_count)

        evaluation_results = evaluation.evaluate()

        print("Evaluation Results:")
        for key, value in evaluation_results.items():
            if isinstance(value, pd.DataFrame):
                print(f"\n{key}:\n", value)
            else:
                print(f"{key}: {value}")


# Testing & example usage. Uncomment to test
if __name__ == '__main__':
    config = ForexMastermindConfig()
    file_path = config.get_next_data_path(data_type='ForexData', training_data_version='6')
    train_test = TrainingPipeline(file_path=file_path)
    train_test.run(sample_size=1)
