import os
import csv
from dotenv import load_dotenv

load_dotenv()


class ForexMastermindConfig:

    def __init__(self, start_date='2018-01-01', end_date='2023-11-21', unseen_start_date='2023-11-20', unseen_end_date='2023-11-24'):
        self.BASE_DIR = os.path.abspath(os.path.dirname(__file__))
        self.NEW_DATA_DIR = os.getenv('FOREXMASTERMIND_NEW_DATA_DIR', '/media/primethanos/sql/ForexMastermind/ML/data')
        self.UNSEEN_DATA_DIR = os.getenv('FOREXMASTERMIND_UNSEEN_DATA_DIR', '/media/primethanos/sql/ForexMastermind/ML/unseen_data')

        self.LIVE_DATA_DIR = os.getenv('FOREXMASTERMIND_LIVE_DATA_DIR',
                                         '/media/primethanos/sql/ForexMastermind/ML/live_data')

        self.MODEL_DIR = os.getenv('FOREXMASTERMIND_MODEL_DIR', '/media/primethanos/sql/ForexMastermind/ML/models')
        self.SCALER_DIR = os.getenv('FOREXMASTERMIND_SCALER_DIR', '/media/primethanos/sql/ForexMastermind/ML/scaler')
        self.PLOT_DIR = os.path.join(self.MODEL_DIR, 'plots')
        self.FEATURE_IMPORTANCE_DIR = os.path.join(self.MODEL_DIR, 'feature_importance')
        self.SQL_DIR = os.getenv('FOREXMASTERMIND_SQL_DIR', '/media/primethanos/sql/ForexMastermind/ML')

        self.DATA_PROCESSING_TEMP_DIR = os.getenv('FOREXMASTERMIND_DATA_PROCESSING_TEMP_DIR',
                                                  '/media/primethanos/sql/ForexMastermind/ML/data_processing_temp')

        # Date ranges
        self.start_date = start_date.replace('-', '')
        self.end_date = end_date.replace('-', '')
        self.unseen_start_date = unseen_start_date.replace('-', '')
        self.unseen_end_date = unseen_end_date.replace('-', '')

    def get_live_data_dir(self):
        return os.getenv('FOREXMASTERMIND_LIVE_DATA_DIR', '/media/primethanos/sql/ForexMastermind/ML/live_data')

    def get_model_directory(self):
        return self.MODEL_DIR

    def get_scaler_directory(self):
        return self.SCALER_DIR

    def get_plot_directory(self):
        return self.PLOT_DIR

    def get_feature_importance_directory(self):
        return self.FEATURE_IMPORTANCE_DIR

    def get_feature_importance_directory(self):
        return self.FEATURE_IMPORTANCE_DIR

    def get_data_processing_temp_directory(self):
        return self.DATA_PROCESSING_TEMP_DIR






    def get_prediction_log_path(self):

        live_data_dir = self.get_live_data_dir()
        prediction_log_filename = 'prediction_log.csv'
        return os.path.join(live_data_dir, prediction_log_filename)

    def create_path(self, base, subpath):
        return os.path.abspath(os.path.join(base, subpath))

    def get_data_path(self, data_type, version, is_unseen=False, live_data=False):

        dir = self.UNSEEN_DATA_DIR if is_unseen else self.NEW_DATA_DIR

        if live_data:
            dir = self.LIVE_DATA_DIR

        # Including date range in the filename
        if data_type == 'ForexData':
            file_name = f"ForexData_{self.start_date}_to_{self.end_date}_v{version}.csv" if not is_unseen else f"ForexData_{self.unseen_start_date}_to_{self.unseen_end_date}_v{version}.csv"
        else:
            file_name = f"{data_type}_v{version}.csv"

        return self.create_path(dir, file_name)

    def get_inverted_data_path(self, data_type, version, is_unseen=False):

        dir = self.UNSEEN_DATA_DIR if is_unseen else self.NEW_DATA_DIR
        file_name = f"{data_type}_v{version}_inverted.csv"
        return self.create_path(dir, file_name)

    def get_next_data_path(self, data_type, training_data_version, is_unseen=False, live_data=False):

        if live_data:
            dir = self.LIVE_DATA_DIR
        elif is_unseen:
            dir = self.UNSEEN_DATA_DIR
        else:
            dir = self.NEW_DATA_DIR

        if data_type == 'ForexData':
            file_name = f"ForexData_{self.start_date}_to_{self.end_date}_1Min_v{training_data_version}.csv" if not is_unseen else f"ForexData_{self.unseen_start_date}_to_{self.unseen_end_date}_1Min_v{training_data_version}.csv"
        else:
            file_name = f"{data_type}_v{training_data_version}.csv"
        return self.create_path(dir, file_name)

    def get_news_data_path(self, version, is_unseen=False, is_live_data=False):

        if is_live_data:
            dir = self.LIVE_DATA_DIR
        elif is_unseen:
            dir = self.UNSEEN_DATA_DIR
        else:
            dir =  self.NEW_DATA_DIR

        file_name = f"ForexNewsSentiment_{self.start_date}_to_{self.end_date}_1Min_v{version}.csv" if not is_unseen else f"ForexNewsSentiment_{self.unseen_start_date}_to_{self.unseen_end_date}_1Min_v{version}.csv"

        return self.create_path(dir, file_name)

    def get_economic_indicators_data_path(self, version, is_unseen=False, is_live_data=False):
        """
        Generates the path for economic indicators data file.

        Args:
            version (str): Version of the economic indicators data.
            is_live_data (bool): Flag to determine if the data is live. Defaults to False
            is_unseen (bool): Flag to determine if the data is unseen. Defaults to False.

        Returns:
            str: The absolute path to the economic indicators data file.
        """
        if is_live_data:
            dir = self.LIVE_DATA_DIR
        elif is_unseen:
            dir = self.UNSEEN_DATA_DIR
        else:
            dir = self.NEW_DATA_DIR

        date_range = f"{self.unseen_start_date}_to_{self.unseen_end_date}" if is_unseen else f"{self.start_date}_to_{self.end_date}"
        file_name = f"EconomicIndicators_{date_range}_v{version}.csv"
        return self.create_path(dir, file_name)

    def get_standard_model_path(self, model_version):
        return self.create_path(self.MODEL_DIR, f"IntraDayForexPredictor_v{model_version}.sav")

    def get_inverted_model_path(self, version):
        return self.create_path(self.MODEL_DIR, f"IntraDayForexPredictor_v{version}_inv.sav")

    def get_scaler_path(self, model_version):
        return self.create_path(self.SCALER_DIR, f"IntraDayForexPredictor_v{model_version}_scaler.sav")

    def get_plot_path(self, version):
        return self.create_path(self.PLOT_DIR, f"IntraDayForexPredictor_v{version}_plot.png")

    def get_hyperparams_path(self):
        return self.create_path(self.MODEL_DIR, 'best_hyperparams.json')

    def get_shap_plot_path(self, version):
        return os.path.join(self.PLOT_DIR, f"IntraDayForexPredictor_v{version}_shap.png")

    def get_model_registry_path(self):
        """
        Generates the path for the model registry file and ensures its existence.

        Returns:
            str: The absolute path to the model registry file.
        """
        registry_path = os.path.join(self.MODEL_DIR, 'model_registry.csv')

        # Check if the registry file exists, if not, create it with headers
        if not os.path.isfile(registry_path):
            with open(registry_path, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['training_date', 'model_version', 'data_hash', 'total_parameters', 'total_samples',
                                 'model_type', 'convergence_time', 'accuracy', 'classification_report',
                                 'model_hyperparameters', 'trained_with', 'feature_list'])
        return registry_path


# Example usage
if __name__ == '__main__':
    config = ForexMastermindConfig()

    # Testing various data paths
    min_data_path = config.get_data_path('ForexData', '3.0.1')
    min_unseen_data_path = config.get_data_path('ForexData', '3.0.1', is_unseen=True)
    next_min_data_path = config.get_next_data_path('ForexData', '5')
    next_min_unseen_data_path = config.get_next_data_path('ForexData', '5', is_unseen=True)
    # Testing model, scaler, and plot paths
    model_path = config.get_standard_model_path('3.0.1')
    scaler_path = config.get_scaler_path('3.0.1')
    plot_path = config.get_plot_path('3.0.1')
    # Printing the paths for verification
    print("Min Data Path:", min_data_path)
    print("Min Unseen Data Path:", min_unseen_data_path)
    print("Next Min Data Path:", next_min_data_path)
    print("Next Min Unseen Data Path:", next_min_unseen_data_path)
    print("Model Path:", model_path)
    print("Scaler Path:", scaler_path)
    print("Plot Path:", plot_path)
    # Testing best hyperparameters path
    hyperparams_path = config.get_hyperparams_path()
    print("Hyperparameters Path:", hyperparams_path)
    # Testing news data paths
    news_data_path = config.get_news_data_path('3')
    unseen_news_data_path = config.get_news_data_path('3', is_unseen=True)
    live_news_data_path = config.get_news_data_path('3', is_live_data=True)

    print("News Data Path:", news_data_path)
    print("Unseen News Data Path:", unseen_news_data_path)
    print("Live News Data Path:", live_news_data_path)

    """
    # Getting model directory
    model_directory = config.get_model_directory()
    print("Model Directory:", model_directory)
    # Getting scaler directory
    scaler_directory = config.get_scaler_directory()
    print("Scaler Directory:", scaler_directory)
    # Getting plot directory
    plot_directory = config.get_plot_directory()
    print("Plot Directory:", plot_directory)
    # Testing inverted model path
    inverted_model_path = config.get_inverted_model_path('3.0.1')
    print("Inverted Model Path:", inverted_model_path)
    # Testing inverted data path
    inverted_data_path = config.get_inverted_data_path('ForexData', '5')
    print("Inverted Data Path:", inverted_data_path)
    economic_data_path = config.get_economic_indicators_data_path(model_version='1')
    unseen_economic_data_path = config.get_economic_indicators_data_path(model_version='1', is_unseen=True)
    print("Economic indicator Data Path:", economic_data_path)
    print("Unseen Economic indicator Data Path:", unseen_economic_data_path)
    prediction_log_path = config.get_prediction_log_path()
    print("Prediction Log Path:", prediction_log_path)
    """
    config.get_economic_indicators_data_path(version='1', is_unseen=True)
    temp_data_dir = config.get_data_processing_temp_directory()
    print(temp_data_dir)
