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
from ForexMastermind.ML.tools import SampleData, DataFrameSaver, ModelVersionService
from ForexMastermind.ML.tools import DataFrameExplorer
# Other imports
import hashlib
import re
from imblearn.over_sampling import SMOTE


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


# services/feature_engineering.py
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

    def _additional_feature_engineering(self, X):
        """
        Apply additional feature engineering steps if required.

        Parameters:
            X (pd.DataFrame): Dataframe containing features.

        Returns:
            pd.DataFrame: Transformed dataframe.
        """
        # Implement additional feature engineering logic here
        # Example: X['new_feature'] = X['existing_feature1'] / X['existing_feature2']
        return X


class DataHashService:
    @staticmethod
    def generate_data_hash(data):
        """
        Generate a hash for the dataset to use as a data model_version identifier.
        """
        return hashlib.md5(pd.util.hash_pandas_object(data).values).hexdigest()


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


class ModelTrainingService:
    def __init__(self, X, y, config, model_version_service):
        self.X = X
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

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)

        # Define base models for the stacking ensemble
        estimators = [
            ('rf', RandomForestClassifier(n_estimators=10, random_state=42)),
            ('mlp', MLPClassifier(hidden_layer_sizes=(100, 100), activation='relu', solver='adam', alpha=0.0001,
                                  batch_size='auto', learning_rate='constant', learning_rate_init=0.001,
                                  max_iter=500, random_state=42)),
            ('xgb', XGBClassifier(use_label_encoder=False, eval_metric='logloss')),
            ('gbm', GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)),
            # Add more models as needed
        ]

        # Define the stacking ensemble with LogisticRegression as the final estimator
        model = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression(), verbose=True)

        # Train the ensemble model
        model.fit(X_train, y_train)

        # Evaluate the model
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Model Accuracy: {accuracy}")

        # Calculate the training time
        training_time = time.time() - start_time
        print(f"Training time: {training_time} seconds")

        # Save the trained model
        model_version = self.model_version_service.get_next_model_version()
        model_directory = self.config.get_model_directory()
        model_filename = f"IntraDayForexPredictor_v{model_version}.sav"
        model_path = os.path.join(model_directory, model_filename)
        joblib.dump(model, model_path)
        print(f"Ensemble model saved as {model_path}")

        return model, training_time


class ModelEvaluationService:
    def __init__(self, model, X_test, y_test, config, training_time, feature_list, model_notes='None',
                 engineered_version='6'):
        """
        Initialize the ModelEvaluationService with the model, test data, and configuration.
        """
        self.model = model
        self.X_test = X_test
        self.y_test = y_test
        self.config = config
        self.model_version_service = ModelVersionService(self.config)
        self.training_time = training_time
        self.feature_list = feature_list
        self.notes = model_notes
        self.engineered_version = engineered_version

    def _calculate_total_parameters(self, stacked_model):
        """
        Calculate total number of parameters in the stacked model.
        """
        total_params = 0
        for estimator in stacked_model.estimators_:
            # Count parameters for each estimator
            if hasattr(estimator, 'coef_'):
                total_params += estimator.coef_.size
            if hasattr(estimator, 'intercept_'):
                total_params += estimator.intercept_.size
        # Count parameters for the final estimator
        final_estimator = stacked_model.final_estimator_
        if hasattr(final_estimator, 'coef_'):
            total_params += final_estimator.coef_.size
        if hasattr(final_estimator, 'intercept_'):
            total_params += final_estimator.intercept_.size

        return total_params

    def _serialize_ensemble_hyperparameters(self, ensemble_model):
        """
        Serialize hyperparameters of the ensemble model.
        """
        serialized_hyperparams = {}
        for idx, base_model in enumerate(ensemble_model.estimators_):
            model_name = f"model_{idx}_{type(base_model).__name__}"
            serialized_hyperparams[model_name] = json.dumps(base_model.get_params(deep=True))
        return json.dumps(serialized_hyperparams)

    def _save_model_metadata(self, accuracy, classification_report_str):
        """
        Save metadata for the trained model.
        """
        model_hyperparams = self._serialize_ensemble_hyperparameters(self.model)
        total_params = self._calculate_total_parameters(self.model)
        data_hash = DataHashService.generate_data_hash(self.X_test)
        model_version = self.model_version_service.get_current_model_version()

        # Include feature_list in the metadata
        metadata = {
            "training_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "model_version": model_version,
            "data_hash": data_hash,
            "total_parameters": total_params,
            "total_samples": len(self.X_test),
            "model_type": type(self.model).__name__,
            "convergence_time": self.training_time,
            "accuracy": accuracy,
            "notes": self.notes,
            "classification_report": classification_report_str,
            "model_hyperparameters": model_hyperparams,
            "trained_with": self.engineered_version,
            "feature_list": self.feature_list
        }
        self._write_metadata_to_registry(metadata)

    def _write_metadata_to_registry(self, metadata):
        model_registry_path = os.path.join(self.config.get_model_directory(), 'model_registry.csv')
        # Check if the registry file exists, if not, create it with headers
        if not os.path.isfile(model_registry_path):
            with open(model_registry_path, mode='w', newline='') as file:
                writer = csv.DictWriter(file, fieldnames=metadata.keys())
                writer.writeheader()

        # Append the metadata to the registry file
        with open(model_registry_path, mode='a', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=metadata.keys())
            writer.writerow(metadata)

        print(f"Model metadata saved for model_version {metadata['model_version']}")

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

    def evaluate(self):
        """
        Evaluate the model and generate various metrics and plots.
        """
        y_pred = self.model.predict(self.X_test)
        y_proba = self.model.predict_proba(self.X_test)[:, 1]
        accuracy = accuracy_score(self.y_test, y_pred)
        classification_report_str = classification_report(self.y_test, y_pred)

        self._save_model_metadata(accuracy, classification_report_str)
        self._generate_evaluation_plots(y_pred, y_proba)

        # Get SHAP values in numerical form
        shap_values = self.get_shap_values(self.X_test)

        # Aggregate SHAP values across all samples
        shap_sum = np.abs(shap_values).mean(axis=0)  # Take the mean of absolute values for each feature

        # Create a DataFrame with the aggregated SHAP values
        shap_df = pd.DataFrame({
            'Feature': self.feature_list,
            'SHAP Importance': shap_sum
        })

        # Sort the DataFrame by importance score in descending order
        shap_df = shap_df.sort_values(by='SHAP Importance', ascending=False)

        self._save_feature_importance(shap_df=shap_df)
        # Print the aggregated SHAP values
        print(shap_df)  # Get SHAP values in numerical form
        shap_values = self.get_shap_values(self.X_test)

        # Aggregate SHAP values across all samples
        shap_sum = np.abs(shap_values).mean(axis=0)  # Take the mean of absolute values for each feature

        # Sort the DataFrame by importance score in descending order
        shap_df = shap_df.sort_values(by='SHAP Importance', ascending=False)

        self._save_feature_importance(shap_df=shap_df)
        # Print the aggregated SHAP values
        print(shap_df)

        return {
            "accuracy": accuracy,
            "classification_report": classification_report_str,
            "plot_path": self._save_evaluation_plots()
        }

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

    def _save_evaluation_plots(self):
        """
        Save the evaluation plots to a specified directory.
        """
        plot_directory = self.config.get_plot_directory()
        model_version = self.model_version_service.get_current_model_version()
        plot_filename = f"IntraDayForexPredictor_v{model_version}_evaluation.png"
        plot_path = os.path.join(plot_directory, plot_filename)

        plt.tight_layout()
        plt.savefig(plot_path)
        plt.show()

        return plot_path

    def _generate_evaluation_plots(self, y_pred, y_proba):
        """
        Generate and display evaluation plots.
        """
        fig, axs = plt.subplots(3, 2, figsize=(20, 18))
        self._plot_confusion_matrix(self.y_test, y_pred, axs[0, 0])
        self._plot_roc_curve(self.y_test, y_proba, axs[0, 1])
        self._plot_precision_recall_curve(self.y_test, y_proba, axs[1, 0])
        self._plot_lift_curve(self.y_test, y_proba, axs[1, 1])
        self._plot_correlation_heatmap(axs[2, 0], self.X_test)
        self._plot_feature_interactions(axs[2, 1], self.X_test, [0, 1])
        plt.tight_layout()
        self._save_evaluation_plots()  # Save the plots

    def _plot_confusion_matrix(self, y_true, y_pred, ax):
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, ax=ax, annot=True, fmt='d', cmap='Blues')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title('Confusion Matrix')

    def _plot_roc_curve(self, y_true, y_score, ax):

        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {roc_auc:.2f})')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curve')
        ax.legend(loc='lower right')

    def _plot_precision_recall_curve(self, y_true, y_score, ax):
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        average_precision = average_precision_score(y_true, y_score)
        ax.plot(recall, precision, color='blue', lw=2, label=f'Precision-Recall Curve (AP={average_precision:.2f})')
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('Precision-Recall Curve')
        ax.legend(loc='lower left')

    def _plot_lift_curve(self, y_true, y_score, ax):

        # Assuming y_score is already computed as the model's probability predictions
        pos_label_idx = (y_true == 1).sum()
        sorted_indices = np.argsort(y_score)[::-1]
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

    def _plot_correlation_heatmap(self, ax, X, threshold=0.8):

        corr_matrix = X.corr()
        filtered_corr_matrix = corr_matrix.where(np.abs(corr_matrix) >= threshold, np.nan)
        sns.heatmap(filtered_corr_matrix, ax=ax, cmap='coolwarm', annot=False)
        ax.set_title('Filtered Correlation Heatmap')
        ax.tick_params(axis='x', rotation=90)
        ax.tick_params(axis='y', rotation=0)

    def _plot_feature_interactions(self, ax, X, features):

        top_features = X.columns[features]
        ax.scatter(X[top_features[0]], X[top_features[1]], c=self.y_test)
        ax.set_xlabel(str(top_features[0]))
        ax.set_ylabel(str(top_features[1]))
        ax.set_title('Feature Interactions')

    def _plot_shap_summary(self, X):
        explainer = shap.KernelExplainer(self.model.predict, X)
        shap_values = explainer.shap_values(X)
        shap.summary_plot(shap_values, X, show=False)

        plot_filename = f"IntraDayForexPredictor_v{self.config.get_current_model_version()}_shap.png"
        plot_path = os.path.join(self.config.get_plot_directory(), plot_filename)
        plt.savefig(plot_path)
        plt.close()

        return plot_path


class TrainingPipeline:
    def __init__(self, file_path, version=None):
        self.file_path = file_path
        self.version = version
        self.config = ForexMastermindConfig()
        self.model_version_service = ModelVersionService(self.config)
        self.data_preprocessor = DataPreprocessor(self.config, self.model_version_service)

    def run(self, sample_size=100, model_notes='None'):
        # Load and process data
        data_loader = DataLoader(self.file_path)
        data = data_loader.load_data()
        if sample_size < 100:
            sampler = SampleData()
            data = sampler.sample_data(n_percent=sample_size, data=data)
        data, features, X_train_res_scaled, y_train_res, X_test_scaled, y_test = self.data_preprocessor.preprocess_data(data)

        # Train the model
        model_training = ModelTrainingService(X=X_train_res_scaled, y=y_train_res, config=self.config,
                                              model_version_service=self.model_version_service)

        model, training_time = model_training.train_model()

        # Evaluate the model
        evaluation = ModelEvaluationService(model, X_test=X_test_scaled, y_test=y_test,
                                            config=self.config, training_time=training_time,
                                            feature_list=features, model_notes=model_notes)

        evaluation_results = evaluation.evaluate()

        print("Evaluation Results:")
        for key, value in evaluation_results.items():
            if isinstance(value, pd.DataFrame):
                print(f"\n{key}:\n", value)
            else:
                print(f"{key}: {value}")



