# Standard library imports
import os
import time

# Related third-party imports
import pandas as pd
from dotenv import load_dotenv

# Local application/library-specific imports
from ForexMastermind.ML.config import ForexMastermindConfig
from ForexMastermind.ML.pipelines.forex_data_fetcher_pipeline import FetchDataPipeline
from ForexMastermind.ML.pipelines.pipeline_indicator_data_merger import EconomicIndicatorDataMergerPipeline
from ForexMastermind.ML.pipelines.pipeline_sentinel_live_predictor import OmegaForexSentinelPredictor
from ForexMastermind.ML.pipelines.pipeline_news_data_merger import NewsMergerPipeline
from ForexMastermind.ML.tools.constants import CURRENCY_PAIRS
from ForexMastermind.ML.tools.tools import (AtomicClockSynchronizer, DataLoader, GetADay, LoadModelAndScaler,
                                            ModelVersionService)


# Load environment variables from .env file
load_dotenv()
API_KEY = os.getenv('FINANCIAL_MODELING_PREP_API_KEY')

# 3.0.73 <-- previous
def set_live_model_version(live_model_version='3.0.86'):
    return live_model_version


class FetchDataService:
    def __init__(self, fetch_data_pipeline):
        self.api_key = API_KEY
        self.start_date, self.end_date = GetADay.get_days()
        self.fetch_data_pipeline = fetch_data_pipeline

    def fetch_live_data(self):

        # uncomment these for data diagnostics
        # data = self.fetch_data_pipeline.run()
        # data_frame_explorer = DataFrameExplorer(dataframe=data)
        # basic_info = data_frame_explorer.get_basic_info()
        # print(basic_info)
        # time.sleep(1000000)

        # Initialize the fetch_data_pipeline
        try:
            fetched_data = self.fetch_data_pipeline

            fetched_data = fetched_data.run()
            return fetched_data
        except Exception as e:
            print(f"An error occurred while fetching live data: {e}")


class NewsDataMergingService:
    def __init__(self, news_merger_pipeline):
        self.news_merger_pipeline = news_merger_pipeline

    def merge_with_news(self, live_data):

        if self.news_merger_pipeline is None:
            raise ValueError("NewsMergerPipeline is not initialized.")

        return self.news_merger_pipeline.run(live_trading_df=live_data)


class IndicatorsDataMergingService:
    def __init__(self, indicators_merger_pipeline):
        self.indicators_merger_pipeline = indicators_merger_pipeline

    def merge_data(self, is_live_data, live_data_feed):

        print(f'{is_live_data} from the live prediction pipeline')

        config = ForexMastermindConfig()
        indicator_data_path = config.get_economic_indicators_data_path(version='1', is_live_data=True)

        data_loader = DataLoader()
        indicator_data = data_loader.load_data(data_path=indicator_data_path)

        if self.indicators_merger_pipeline is None:
            raise ValueError("IndicatorsDataMergingService is not initialized.")

        return self.indicators_merger_pipeline.merge_data(live_data_feed=live_data_feed, indicator_data=indicator_data)


class PredictionService:
    def __init__(self):

        self.model_version = set_live_model_version()
        model_path = config.get_standard_model_path(model_version=self.model_version)
        scaler_path = config.get_scaler_path(model_version=self.model_version)
        model_loader = LoadModelAndScaler(model_path, scaler_path)
        self.model, self.scaler = model_loader.load()

    def make_predictions(self, data):

        sentinel_predictor = OmegaForexSentinelPredictor(model=self.model, scaler=self.scaler,
                                                         model_version=self.model_version)

        sentinel_predictor.receive_processed_data(feature_engineered_data=data)
        sentinel_predictor.preprocess_data()

        return sentinel_predictor.predict()


class DataSavingService:
    def __init__(self, config):
        self.config = config

    def concatenate_prediction_with_features(self, prediction_results, most_recent_data):
        # Align the dataframes by a common key if needed, e.g., 'date' or 'Pair'
        # This assumes most_recent_data has a 'Pair' or 'date' column that is also present in prediction_results

        combined = pd.merge(prediction_results, most_recent_data, on='Pair', suffixes=('', '_drop'))

        # Drop duplicate columns by filtering out columns with '_drop' suffix
        combined = combined[[c for c in combined.columns if not c.endswith('_drop')]]

        # Now combined has the prediction results and the features from most_recent_data, without duplicates
        return combined

    # Use this function to append combined data to CSV
    def append_to_csv(self, prediction_results, most_recent_data):
        try:
            combined_data = self.concatenate_prediction_with_features(prediction_results, most_recent_data)
            print('After combining data')
            prediction_log_path = self.config.get_prediction_log_path()
            if os.path.exists(prediction_log_path):
                combined_data.to_csv(prediction_log_path, mode='a', header=False, index=False)
            else:
                combined_data.to_csv(prediction_log_path, mode='w', header=True, index=False)
            print('Data saved to CSV')
        except Exception as e:
            print(f"Error in appending to CSV: {e}")


class PredictiveAnalyticsPipeline:
    def __init__(self, model_version, fetch_service, news_merger_service, indicators_merger_service, predictor_service, saver_service):

        self.model_version = model_version
        self.fetch_service = fetch_service
        self.news_merger_service = news_merger_service
        self.indicators_merger_service = indicators_merger_service
        self.predictor_service = predictor_service
        self.saver_service = saver_service
        self.config = ForexMastermindConfig()
        self.model_version_service = ModelVersionService(config=self.config)

    @staticmethod
    def get_most_recent_data(data):
        """
        Filter to retain only the most recent data for each pair.
        """
        # Convert 'date' column to datetime if it's not already
        data['date'] = pd.to_datetime(data['date'])

        # Sort data by 'date' in ascending order
        data_sorted = data.sort_values('date', ascending=True)

        # Now group by 'Pair' and take the last occurrence
        most_recent_data = data_sorted.groupby('Pair').last().reset_index()
        return most_recent_data

    def run(self):

        while True:
            # Wait for the next minute to fetch new live data
            AtomicClockSynchronizer.sleep_until_next_minute()

            # Attempt to fetch live data
            live_data = self.fetch_service.fetch_live_data()

            # If live data is not fetched, exit the pipeline
            if live_data is None:
                print("Failed to fetch live data. Aborting pipeline.")
                break

            # Merge news data into the live data
            news_merged_data = self.news_merger_service.merge_with_news(live_data=live_data)

            # Ensure that news merging was successful before proceeding
            if news_merged_data is None:
                print("Failed to merge news data. Aborting pipeline.")
                break

            # Merge indicator data into the news merged data
            indicator_merged_data = self.indicators_merger_service.merge_data(
                is_live_data=True, live_data_feed=news_merged_data)

            # Ensure that indicator merging was successful before proceeding
            if indicator_merged_data is None:
                print("Failed to merge indicator data. Aborting pipeline.")
                break

            # Get the most recent data after merging indicators
            most_recent_data = self.get_most_recent_data(data=indicator_merged_data)
            most_recent_data.fillna(-1, inplace=True)

            # Make predictions using the most recent data
            prediction_results = self.predictor_service.make_predictions(data=most_recent_data)
            print(prediction_results)
            self.saver_service.append_to_csv(prediction_results, most_recent_data)


if __name__ == '__main__':
    start_date, end_date = GetADay.get_days()
    fetch_data_pipeline = FetchDataPipeline(
                    pairs=CURRENCY_PAIRS.copy(),
                    start_date=start_date,
                    end_date=end_date,
                    api_key=API_KEY,
                    live_data=True
                )

    fetch_service = FetchDataService(fetch_data_pipeline=fetch_data_pipeline)

    news_merger_pipeline = NewsMergerPipeline(start_date=start_date, end_date=end_date, live_data=True)
    news_merger_service = NewsDataMergingService(news_merger_pipeline=news_merger_pipeline)

    get_a_day = GetADay()
    _, end_date = get_a_day.get_days()
    start_date = '2018-01-01'
    indicators_merger_pipeline = EconomicIndicatorDataMergerPipeline(is_live_data=True)
    indicators_merger_service = IndicatorsDataMergingService(indicators_merger_pipeline=indicators_merger_pipeline)

    config = ForexMastermindConfig()
    predictor_service = PredictionService()
    saver_service = DataSavingService(config=config)
    predictive_analytics_pipeline = PredictiveAnalyticsPipeline(
        model_version=set_live_model_version(),
        fetch_service=fetch_service,
        news_merger_service=news_merger_service,
        indicators_merger_service=indicators_merger_service,

        predictor_service=predictor_service,
        saver_service=saver_service
    )
    predictive_analytics_pipeline.run()

