# neural_network_training_pipeline.py

# Standard library imports
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
from ForexMastermind.ML.pipelines.pipeline_feature_engineering import Balance, DataLoader
from ForexMastermind.ML.config import ForexMastermindConfig
from ForexMastermind.ML.tools import DataFrameExplorer, ModelVersionService, DataFrameSaver, DataLoader, SampleData
# Other imports
import hashlib
import re
from imblearn.over_sampling import SMOTE

# services/feature_engineering.py


class DataPreprocessor:
    def __init__(self, config, model_version_service):
        self.config = config
        self.model_version_service = model_version_service

    def preprocess_data(self, data):

        data = self._drop_redundant_columns(data)
        data = self._balance_dataset(data)
        data = self._drop_nan_values(data)

        features, X, y = self._separate_features_and_target(data)
        X_train, X_test, y_train, y_test = self._split_data(X, y)

        X_train_res, y_train_res, X_test_scaled = self._process_features(X_train, X_test, y_train)

        print(data.head())

        return data, features, X_train_res, y_train_res, X_test_scaled, y_test

    def _drop_redundant_columns(self, data):

        try:
            drop_cols = ['sentiment_score_x', 'news_headline_x', 'sentiment_score_y', 'news_headline_y']
            return data.drop(drop_cols, axis=1)
        except KeyError as e:
            # Handle the case where one or more columns to drop don't exist in the DataFrame
            print(f"Warning: One or more columns to drop not found in the DataFrame: {e}")
            return data

    def _balance_dataset(self, data):

        balance = Balance(data=data)
        return balance.balance_samples()

    def _drop_nan_values(self, data):
        return data.dropna()

    def _separate_features_and_target(self, data):

        non_feature_cols = ['country', 'impact', 'event', 'Target', 'date', 'Pair', 'currency', 'news_headline']
        features = [col for col in data.columns if col not in non_feature_cols]
        X = data[features].copy()
        y = data['Target'].copy()
        return features, X, y

    def _split_data(self, X, y):

        return train_test_split(X, y, test_size=0.2, random_state=42)

    def _process_features(self, X_train, X_test, y_train):

        self._validate_features(X_train)
        X_train_res, y_train_res = self._apply_smote(X_train, y_train)
        X_train_res_scaled, X_test_scaled = self._scale_features(X_train_res, X_test)
        return X_train_res, y_train_res, X_test_scaled

    def _validate_features(self, X):

        if X.isnull().any().any():
            raise ValueError("NaN values found in dataset.")
        if not all(X[col].dtype.kind in 'fi' for col in X.columns):
            non_numeric_cols = [col for col in X.columns if X[col].dtype.kind not in 'fi']
            raise ValueError(f"Non-numeric data types found in features: {non_numeric_cols}")

    def _apply_smote(self, X, y):

        sm = SMOTE(random_state=42)
        return sm.fit_resample(X, y)

    def _scale_features(self, X_train_res, X_test):

        scaler = StandardScaler()
        X_train_res_scaled = pd.DataFrame(scaler.fit_transform(X_train_res), columns=X_train_res.columns)
        X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

        model_version = self.model_version_service.get_next_model_version()
        scaler_path = os.path.join(self.config.get_scaler_directory(),
                                   f"IntraDayForexPredictor_v{model_version}_scaler.sav")
        joblib.dump(scaler, scaler_path)

        return X_train_res_scaled, X_test_scaled


class DataPreprocessingService:
    def __init__(self, data, features, scaler):
        self.data = data
        self.features = features
        self.scaler = scaler

    def execute(self):
        """
        Execute data preprocessing steps.
        """
        # Drop NaN values from the dataset
        self.data.dropna(inplace=True)

        # Optionally sample the data if it's very large
        if len(self.data) >= 10000:
            self.data = self.data.sample(n=10000, random_state=3)

        # Extract features and target from the data
        X = self.data[self.features].copy()
        y = self.data['Target'].copy()

        # Diagnostics: Compare feature sets
        print("Number of features in X_test:", len(X.columns))
        print("Number of features in scaler:", len(self.scaler.feature_names_in_))
        print("Number of features in the feature list:", len(self.features))

        # Find features in scaler that are not in X
        missing_features = set(self.scaler.feature_names_in_) - set(X.columns)
        print("Features in the scaler but not in X:", missing_features)

        # Ensure the feature names match those used during model training
        X.columns = self.scaler.feature_names_in_

        # Scale the features using the loaded scaler
        X_scaled = pd.DataFrame(self.scaler.transform(X), columns=self.scaler.feature_names_in_)

        return X_scaled, y


class FeatureEngineeringService:
    def __init__(self, data, feature_list):
        """
        Initialize the FeatureEngineeringService with the dataset and a list of features to be used.

        Parameters:
            data (pd.DataFrame): The dataset to be processed.
            feature_list (list): A list of feature names to be used for model training or evaluation.
        """
        self.data = data
        self.feature_list = feature_list

    def execute(self):
        """
        Execute feature engineering steps on the provided dataset.

        Returns:
            tuple: A tuple containing the features (X), target variable (y), and the final list of feature names used.
        """
        # Identify features and target variable
        X = self.data[self.feature_list].copy()
        y = self.data['Target'].copy()

        # Remove non-numeric columns and any other preprocessing as required
        # Example: X = X.select_dtypes(include=['number'])

        # Optionally, more feature engineering steps can be added here
        # Example: X = self._additional_feature_engineering(X)

        return X, y, self.feature_list


class OmegaForexSentinel:

    def __init__(self, file_path, version=None):

        self.file_path = file_path
        self.data = DataLoader(data_path=file_path).load_data()
        self.price_movement_threshold = 0.0001
        self.version = version if version is not None else "v0.0"
        self.model_name = f"ForexModel_{self.version}"
        self.model = None
        self.model_version = None
        self.start_time = time.time()
        self.training_time = 0
        self.X_features = None
        self.features = None

        self.config = ForexMastermindConfig()
        self.model_version_service = ModelVersionService(config=self.config)
        self.data_preprocessor = DataPreprocessor(self.config, self.model_version_service)

    def model_predict(self, X):
        """ Wrapper function to make predictions with the stacked model """
        return self.model.predict_proba(X)

    def get_shap_values(self, X):
        # Create an explainer with the model's predict_proba function and a sample of the input X
        # The following uses KernelExplainer as an example, but TreeExplainer could be more appropriate
        # for tree-based models for efficiency.
        explainer = shap.KernelExplainer(self.model.predict_proba, shap.sample(X, 100))

        # Calculate SHAP values for all samples in X (this can be computationally intensive)
        shap_values = explainer.shap_values(X, nsamples=100)

        # SHAP values are returned as a list of arrays for multi-class outputs. For binary classification,
        # you can just take the first element.
        return shap_values[1] if isinstance(shap_values, list) else shap_values

    def optimize_hyperparameters(self, trial):
        """
        Optimize hyperparameters using Optuna.
        """
        # Define the hyperparameter search space
        n_layers = trial.suggest_int('n_layers', 1, 3)
        layers = [trial.suggest_int(f'n_units_l{i}', 50, 500) for i in range(n_layers)]

        # Create the classifier with the suggested hyperparameters
        classifier = MLPClassifier(
            hidden_layer_sizes=tuple(layers),
            activation=trial.suggest_categorical('activation', ['relu', 'tanh']),
            solver=trial.suggest_categorical('solver', ['adam', 'sgd']),
            alpha=trial.suggest_float('alpha', 1e-5, 1e-1, log=True),
            batch_size=trial.suggest_categorical('batch_size', [64, 128, 256]),
            learning_rate_init=trial.suggest_float('learning_rate_init', 1e-5, 1e-1, log=True),
            max_iter=10000,  # Adjust as needed
            random_state=42,
            verbose=True
        )

        # Use a subset of data for efficiency
        X_subsample, _, y_subsample, _ = train_test_split(
            self.X_train_res_scaled, self.y_train_res, test_size=0.95, random_state=42
        )

        # Train the model
        classifier.fit(X_subsample, y_subsample)

        # Evaluate the model's accuracy on the validation set
        y_pred = classifier.predict(self.X_test_scaled)
        accuracy = accuracy_score(self.y_test, y_pred)

        return accuracy

    def run_hyperparameter_optimization(self):

        model_directory = self.config.get_model_directory()

        study = optuna.create_study(direction='maximize')
        study.optimize(self.optimize_hyperparameters, n_trials=100)  # Adjust the number of trials as needed

        # Save the best hyperparameters
        best_hyperparams = study.best_trial.params

        model_version = self._get_next_model_version()
        best_hyperparams_path = os.path.join(model_directory, f'best_hyperparams_{model_version}.json')

        with open(best_hyperparams_path, 'w') as f:
            json.dump(best_hyperparams, f)

        print(f"Best hyperparameters: {best_hyperparams}")
        self.best_hyperparams = best_hyperparams  # Save the best hyperparameters to the instance for later use

    def train_best_model(self):
        # Extract layer sizes from the best hyperparameters
        n_layers = self.best_hyperparams['n_layers']
        hidden_layer_sizes = tuple(self.best_hyperparams[f'n_units_l{i}'] for i in range(n_layers))

        # Prepare the classifier parameters, excluding 'n_layers' and layer unit keys
        classifier_params = {k: v for k, v in self.best_hyperparams.items() if
                             not k.startswith('n_units_l') and k != 'n_layers'}
        classifier_params['hidden_layer_sizes'] = hidden_layer_sizes

        # Use the best hyperparameters to train the final model
        self.model = MLPClassifier(**classifier_params)
        self.model.fit(self.X_train_res_scaled, self.y_train_res)

    def _generate_data_hash(self):
        """
        Generate a hash for the current dataset to use as a data model_version identifier.
        """
        return hashlib.md5(pd.util.hash_pandas_object(self.data).values).hexdigest()

    def _get_existing_versions(self):
        """
        Scan the model directory for existing model files and retrieve all versions.
        """

        model_dir = self.config.get_model_directory()

        pattern = re.compile(r'IntraDayForexPredictor_v(\d+)\.(\d+)\.(\d+)\.sav')
        versions = []

        for filename in os.listdir(model_dir):
            match = pattern.match(filename)
            if match:
                version_numbers = tuple(map(int, match.groups()))
                versions.append(version_numbers)

        return versions

    def _get_next_model_version(self):
        """
        Generate the next model model_version number based on existing models in the registry.
        """
        existing_versions = self._get_existing_versions()
        if not existing_versions:
            return '1.0.0'

        latest_version = max(existing_versions)
        major, minor, patch = latest_version
        next_version = f"{major}.{minor}.{patch + 1}"
        self.model_version = next_version

        return next_version

    def preprocess_data(self):
        # Drop redundant columns

        #drop_cols = ['sentiment_score_x', 'news_headline_x', 'sentiment_score_y', 'news_headline_y']
        #self.data.drop(drop_cols, axis=1, inplace=True)

        # Balance the dataset
        balance = Balance(data=self.data)
        self.data = balance.balance_samples()

        # Drop any remaining NaN values
        self.data.dropna(inplace=True)

        # Define the feature set, excluding non-feature columns
        non_feature_cols = ['country', 'impact', 'event', 'Target', 'date', 'Pair', 'currency', 'news_headline']
        self.features = [col for col in self.data.columns if col not in non_feature_cols]
        print(f'There are {len(self.features)} features in the training set.')

        # Separate features (X) and target (y)
        X = self.data[self.features].copy()
        y = self.data['Target'].copy()

        # Check for NaN values and non-numeric data types in features
        if X.isnull().any().any():
            raise ValueError("NaN values found in dataset.")
        if not all(X[col].dtype.kind in 'fi' for col in X.columns):
            non_numeric_cols = [col for col in X.columns if X[col].dtype.kind not in 'fi']
            raise ValueError(f"Non-numeric data types found in features: {non_numeric_cols}")

        # Split the data into training and test sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Apply SMOTE to balance the training set
        sm = SMOTE(random_state=42)
        self.X_train_res, self.y_train_res = sm.fit_resample(self.X_train, self.y_train)

        # Scale the features
        scaler = StandardScaler()
        self.X_train_res_scaled = pd.DataFrame(scaler.fit_transform(self.X_train_res), columns=self.X_train_res.columns)
        self.X_test_scaled = pd.DataFrame(scaler.transform(self.X_test), columns=self.X_test.columns)

        # Save the scaler
        model_version = self._get_next_model_version()
        scaler_path = os.path.join(self.config.get_scaler_directory(),
                                   f"IntraDayForexPredictor_v{model_version}_scaler.sav")
        joblib.dump(scaler, scaler_path)

    def train_model(self):
        """
        Trains a stacked ensemble machine learning model using various base classifiers and a meta-classifier.
        This method now includes SVM as a base model, along with RandomForest, MLPClassifier, XGBoost, GBM, and AdaBoost,
        and uses Logistic Regression as the meta-classifier.
        After training, the ensemble model is saved to the filesystem with a unique model_version identifier.
        """

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
            #('ada', AdaBoostClassifier(n_estimators=100, random_state=42)),
            #('svm', SVC(kernel='rbf', probability=True, random_state=42))  # SVM added here
        ]

        # Define the stacking ensemble
        self.model = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression(), verbose=True)

        # Train the ensemble model
        self.model.fit(self.X_train_res_scaled, self.y_train_res)

        # Save the trained ensemble model
        model_version = self._get_next_model_version()
        model_directory = self.config.get_model_directory()
        model_filename = f"IntraDayForexPredictor_v{model_version}.sav"
        model_path = os.path.join(model_directory, model_filename)
        self.training_time = time.time() - self.start_time  # timing
        joblib.dump(self.model, model_path)
        print(f"Ensemble model saved as {model_path}")

    @staticmethod
    def _serialize_ensemble_hyperparameters(ensemble_model):

        serialized_hyperparams = {}
        for idx, base_model in enumerate(ensemble_model.estimators_):
            model_name = f"model_{idx}_{type(base_model).__name__}"
            serialized_hyperparams[model_name] = json.dumps(base_model.get_params(deep=True))
        return json.dumps(serialized_hyperparams)

    @staticmethod
    def calculate_total_parameters(stacked_model):
        total_params = 0
        for _, estimator in stacked_model.estimators:
            if hasattr(estimator, 'coef_'):
                total_params += estimator.coef_.size
            if hasattr(estimator, 'intercept_'):
                total_params += estimator.intercept_.size

        final_estimator = stacked_model.final_estimator_
        if hasattr(final_estimator, 'coef_'):
            total_params += final_estimator.coef_.size
        if hasattr(final_estimator, 'intercept_'):
            total_params += final_estimator.intercept_.size

        return total_params

    def _save_model_metadata(self, accuracy, classification_rep, training_time):
        """
        Save the metadata for the trained model.
        """

        # Serialize model hyperparameters
        model_hyperparams = self._serialize_ensemble_hyperparameters(self.model)

        total_params = OmegaForexSentinel.calculate_total_parameters(self.model)
        next_min_data_path = self.config.get_next_data_path('ForexData', '6')

        metadata = {
            "training_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "model_version": self.model_version,
            "data_hash": self._generate_data_hash(),
            "total_parameters": total_params,
            "total_samples": len(self.data),
            "model_type": type(self.model).__name__,  # Updated to reflect the actual model type
            "convergence_time": training_time,
            "accuracy": accuracy,

            "classification_report": classification_rep,
            "model_hyperparameters": model_hyperparams,
            "data_version": next_min_data_path,
            "feature_list": self.features
        }

        model_directory = self.config.get_model_directory()
        model_registry_path = os.path.join(model_directory, 'model_registry.csv')

        # Check if the registry file exists, if not, create it with headers
        if not os.path.isfile(model_registry_path):
            with open(model_registry_path, mode='w', newline='') as file:
                writer = csv.DictWriter(file, fieldnames=metadata.keys())
                writer.writeheader()

        # Append the metadata to the registry file
        with open(model_registry_path, mode='a', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=metadata.keys())
            writer.writerow(metadata)

        print(f"Model metadata saved for model_version {self.model_version}")

    def _save_feature_importance(self, shap_df):
        """
        Save the evaluation plots to a specified directory.
        """
        feature_importance_directory = self.config.get_feature_importance_directory()
        model_version = self.model_version_service.get_current_model_version()
        filename = f"IntraDayForexPredictor_v{model_version}_FeatureImportance.csv"
        feature_importance_path = os.path.join(feature_importance_directory, filename)

        data_frame_saver = DataFrameSaver()
        data_frame_saver.save_df(df=shap_df, path=feature_importance_path)

    def evaluate_model(self):
        # Model predictions

        y_pred = self.model.predict(self.X_test_scaled)
        y_proba = self.model.predict_proba(self.X_test_scaled)[:, 1]

        accuracy = accuracy_score(self.y_test, y_pred)
        classification_report_str = classification_report(self.y_test, y_pred)

        # Call the method to save model metadata
        self._save_model_metadata(
            accuracy=accuracy,
            classification_rep=classification_report_str,
            training_time=self.training_time
        )

        # Create figure for subplots with a 3x3 layout to accommodate the SHAP plot
        fig, axs = plt.subplots(3, 2, figsize=(20, 18))  # This was the original size

        # Plot each chart in its subplot
        self.plot_confusion_matrix(self.y_test, y_pred, axs[0, 0])
        self.plot_roc_curve(self.y_test, y_proba, axs[0, 1])
        self.plot_precision_recall_curve(self.y_test, y_proba, axs[1, 0])
        self.plot_lift_curve(self.y_test, y_proba, axs[1, 1])
        self.plot_correlation_heatmap(axs[2, 0])
        self.plot_feature_interactions(axs[2, 1], [0, 1])

        # Classification report and accuracy
        self.print_classification_report(self.y_test, y_pred)
        print("Accuracy:", accuracy)

        plot_directory = self.config.get_plot_directory()
        plot_filename = f"IntraDayForexPredictor_v{self.model_version}_plot.png"
        plot_path = os.path.join(plot_directory, plot_filename)

        # Adjust layout and save the plot
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.show()  # Show the plots
        # self.plot_shap_summary()

        data, features, X_train_res_scaled, y_train_res, X_test_scaled, y_test = self.data_preprocessor.preprocess_data(
            self.data)

        # Get SHAP values in numerical form
        shap_values = self.get_shap_values(self.X_test)
        # Aggregate SHAP values across all samples
        shap_sum = np.abs(shap_values).mean(axis=0)  # Take the mean of absolute values for each feature


        shap_df = pd.DataFrame({
            'Feature': features,
            'SHAP Importance': shap_sum
        })

        # Sort the DataFrame by importance score in descending order
        shap_df = shap_df.sort_values(by='SHAP Importance', ascending=False)

        self._save_feature_importance(shap_df=shap_df)
        # Print the aggregated SHAP values
        print(shap_df)  # Get SHAP values in numerical form

    def print_classification_report(self, y_true, y_pred):

        report = classification_report(y_true, y_pred)
        print("\nClassification Report:\n", report)

    def plot_confusion_matrix(self, y_true, y_pred, ax):

        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, ax=ax, annot=True, fmt='d', cmap='Blues')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title('Confusion Matrix')

    def plot_roc_curve(self, y_true, y_score, ax):

        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {roc_auc:.2f})')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curve')
        ax.legend(loc='lower right')

    def plot_precision_recall_curve(self, y_true, y_score, ax):

        precision, recall, _ = precision_recall_curve(y_true, y_score)
        average_precision = average_precision_score(y_true, y_score)
        ax.plot(recall, precision, color='blue', lw=2, label=f'Precision-Recall Curve (AP={average_precision:.2f})')
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('Precision-Recall Curve')
        ax.legend(loc='lower left')

    def plot_lift_curve(self, y_true, y_score, ax):

        y_scores = self.model.predict_proba(self.X_test_scaled)[:, 1]
        pos_label_idx = (y_true == 1).sum()
        sorted_indices = np.argsort(y_scores)[::-1]
        sorted_y_true = y_true.iloc[sorted_indices]
        x_cumulative = np.arange(1, len(sorted_y_true) + 1)
        y_cumulative = np.cumsum(sorted_y_true) / pos_label_idx
        lift = y_cumulative / (x_cumulative / len(sorted_y_true))

        ax.plot(x_cumulative, lift, marker='.', color='blue', label='Lift Curve')
        ax.plot(x_cumulative, np.linspace(1, 1, len(sorted_y_true)), color='red', linestyle='--', label='Baseline')
        ax.set_xlabel('Number of cases examined')
        ax.set_ylabel('Lift')
        ax.set_title('Lift Curve')
        ax.legend(loc='upper right')

    def plot_correlation_heatmap(self, ax, threshold=0.8):
        """
        Plots a heatmap for feature correlations, filtering out correlations below the threshold.
        """
        # Calculate the correlation matrix for the selected features
        corr_matrix = self.X_train_res_scaled.corr()

        # Filter out correlations below the threshold
        filtered_corr_matrix = corr_matrix.where(np.abs(corr_matrix) >= threshold, np.nan)

        # Plot the filtered heatmap
        sns.heatmap(filtered_corr_matrix, ax=ax, cmap='coolwarm',
                    annot=False)  # Turn off annotations by setting `annot=False`
        ax.set_title('Filtered Correlation Heatmap')
        ax.tick_params(axis='x', rotation=90)
        ax.tick_params(axis='y', rotation=0)

    def plot_feature_interactions(self, ax, features):
        top_features = self.X_train_res_scaled.columns[features]
        ax.scatter(self.X_train_res_scaled[top_features[0]], self.X_train_res_scaled[top_features[1]],
                   c=self.y_train_res)
        ax.set_xlabel(str(top_features[0]))
        ax.set_ylabel(str(top_features[1]))
        ax.set_title('Feature Interactions')

    def plot_shap_summary(self):
        """
        Generate and save a standalone SHAP summary plot.
        """
        # Assuming self.X_test_scaled and self.model are already defined and valid

        # Create a SHAP explainer with the model and test data
        explainer = shap.KernelExplainer(self.model.predict, self.X_test_scaled)

        # Calculate SHAP values - this may take some time for larger datasets
        shap_values = explainer.shap_values(self.X_test_scaled)

        # Generate the SHAP summary plot
        shap.summary_plot(shap_values, self.X_test_scaled, show=False)

        # Define the plot filename using established naming conventions
        model_version = self._get_next_model_version()  # Method to get the next model model_version
        plot_filename = f"IntraDayForexPredictor_v{model_version}_shap.png"
        plot_path = self.config.get_plot_path(model_version)  # Method to get the full plot path

        # Save the plot
        plt.savefig(plot_path)
        plt.close()  # Close the plot to free memory

        print(f"SHAP summary plot saved as {plot_path}")
        return plot_path


class TrainingPipeline:
    def __init__(self, file_path, version=None):
        self.file_path = file_path

    def run(self, sample_size=100):

        # Use this block to cut a subset of the total number of samples for training
        # You might want to cut a small percentage of very large sample sizes to
        # conserve computational resources, and or test concepts in a shorter time.
        # Alternatively, leave the default value which is 100 for all samples.
        training = OmegaForexSentinel(file_path=self.file_path)
        sample_data = SampleData()
        training.data = sample_data.sample_data(n_percent=sample_size, data=training.data)

        training.preprocess_data()
        training.train_model()
        training.evaluate_model()
        #training.calculate_and_save_shap_values()


class OptimizationPipeline:
    def __init__(self, file_path, version=None):
        self.file_path = file_path

    def run(self, sample_size=100):

        training = OmegaForexSentinel(file_path=self.file_path)
        sample_data = SampleData()
        training.data = sample_data.sample_data(n_percent=sample_size, data=training.data)
        training.preprocess_data()
        training.run_hyperparameter_optimization()  # Optimize hyperparameters
        training.train_best_model()  # Train the model using the best hyperparameters
        training.evaluate_model()  # Evaluate the trained model



