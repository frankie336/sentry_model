import time
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from ForexMastermind.ML.tools.tools import LoadModelAndScaler

from ForexMastermind.ML.tools.tools import DataLoader, FeatureFetcherSingleton


class ModelEvaluationPipeline:
    def __init__(self, file_path, model_path, scaler_path, features, version=None):
        """
        Initialize the pipeline with file paths and model model_version.
        """
        self.file_path = file_path
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.version = version if version is not None else "v0.0"
        self.data = None
        self.X_test = None
        self.y_test = None
        self.model = None
        self.scaler = None
        self.features = features
        self.model_loader = LoadModelAndScaler(model_path, scaler_path, version)

    def preprocess_data(self):
        """
        Preprocess the unseen data for evaluation.
        """
        # Assuming DataLoader is a class that loads and preprocesses the data
        data_loader = DataLoader()
        self.data = data_loader.load_data(data_path=self.file_path)

        self.data.dropna(inplace=True)

        if len(self.data) >= 10000:
            self.data = self.data.sample(n=10000, random_state=3)

        # Extract the features and target from the data
        features = [col for col in self.data.columns if col not in ['Target', 'country', 'impact', 'event', 'date',
                                                                    'Pair', 'event', 'currency', 'news_headline']]

        # Filter features
        features = [feature for feature in features if feature in self.features]

        self.X_test = self.data[features].copy()
        self.y_test = self.data['Target'].copy()

        # Ensure the feature names match those used during model training
        self.X_test.columns = self.scaler.feature_names_in_

        # Scale the features using the loaded scaler
        self.X_test = pd.DataFrame(self.scaler.transform(self.X_test), columns=self.scaler.feature_names_in_)

    def evaluate_model(self):
        """
        Evaluate the model on unseen data.
        """
        y_pred = self.model.predict(self.X_test)
        y_proba = self.model.predict_proba(self.X_test)[:, 1]

        # Generate and print evaluation metrics
        accuracy = accuracy_score(self.y_test, y_pred)
        report = classification_report(self.y_test, y_pred)
        print(f"Accuracy: {accuracy}\n")
        print(f"Classification Report:\n{report}")
        print(f"A Sample size of {len(self.data)} of unseen data independent of the training process.")

    def run(self):
        """
        Run the model evaluation pipeline.
        """
        start_time = time.time()
        self.model, self.scaler = self.model_loader.load()
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Model And Scaler Load Time: {elapsed_time} seconds")
        self.preprocess_data()
        self.evaluate_model()
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Evaluation Time: {elapsed_time} seconds")

