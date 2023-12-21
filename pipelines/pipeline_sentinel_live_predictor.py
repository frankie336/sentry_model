# pipeline_sentinel_live_predictor.py
import os
import time
import numpy as np
from ForexMastermind.ML.tools.tools import (FeatureFetcherSingleton, DataLoader, ModelVersionService,
                                      DataFrameColumnToListService)
from ForexMastermind.ML.config import ForexMastermindConfig
import pandas as pd


class OmegaForexSentinelPredictor:
    def __init__(self, model, scaler, model_version):

        self.model_version = model_version
        self.model = model
        self.scaler = scaler
        self.config = ForexMastermindConfig()
        self.model_version_service = ModelVersionService(config=self.config)
        self.feature_engineered_data = None
        self.X_test = None

    def receive_processed_data(self, feature_engineered_data):
        self.feature_engineered_data = feature_engineered_data

    def preprocess_data(self):

        feature_fetcher = FeatureFetcherSingleton()
        features = feature_fetcher.get_features(model_version=self.model_version)

        if self.feature_engineered_data is not None:
            self.X_test = self.feature_engineered_data[features]

    def predict(self):

        if self.X_test is not None:
            self.X_test_scaled = pd.DataFrame(self.scaler.transform(self.X_test), columns=self.X_test.columns)

            # Use predict_proba to get probability estimates
            probabilities = self.model.predict_proba(self.X_test_scaled)

            # Extract probabilities for both negative and positive classes
            probability_negative_class = probabilities[:, 0]
            probability_positive_class = probabilities[:, 1]

            # Get predictions from each base model
            base_model_predictions = {estimator_name: estimator.predict(self.X_test_scaled)
                                      for estimator_name, estimator in self.model.named_estimators_.items()}

            # Calculate ensemble disagreement
            # Example: Voting disagreement
            voting_disagreement = np.sum([preds != np.round(self.model.predict(self.X_test_scaled))
                                          for preds in base_model_predictions.values()], axis=0)

            predictions = self.model.predict(self.X_test_scaled)

            # Additional data to collect for training the meta-model
            model_confidence = probabilities.max(axis=1)  # Model's confidence in its prediction
            model_variance = probabilities.var(axis=1)  # Variance in probability estimates

            prediction_results = pd.DataFrame({
                # original model features here
                "date": self.feature_engineered_data["date"],
                "Pair": self.feature_engineered_data["Pair"],
                "open": self.feature_engineered_data["open"],
                "close": self.feature_engineered_data["close"],

                # meta feature here
                "Probability_Negative": probability_negative_class,
                "Probability_Positive": probability_positive_class,
                "Model_Confidence": model_confidence,
                "Model_Variance": model_variance,
                "Voting_Disagreement": voting_disagreement,
                "Prediction": predictions

                # Add other relevant columns here

            })

            return prediction_results
        else:
            print("No data to predict on")
            return None

