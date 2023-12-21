from ForexMastermind.ML.tools.tools import DataFrameSaver, DataLoader, GetADay
from ForexMastermind.ML.pipelines.draft_forex_data_fetcher_pipeline import (
    ForexDataAPIFetcherService, SentimentAnalyzerService, ForexPriceDataChunkFetcher,
    EconomicCalendarFetcherService, NewsFetcherService, EconomicIndicatorsFetcherService,
    EconomicCalendarPreprocessingPipeline
)
from ForexMastermind.ML.pipelines.pipeline_feature_engineering import FeatureEngineeringPipeline
from ForexMastermind.ML.pipelines.pipeline_news_data_merger import NewsMergerPipeline
from ForexMastermind.ML.pipelines.pipeline_indicator_data_merger import EconomicIndicatorDataMergerPipeline
from ForexMastermind.ML.pipelines.draft_forex_data_fetcher_pipeline import FetchDataPipeline, DataMergerService


class DataPipelineBuilder:
    def __init__(self, api_key, config):
        self.api_key = api_key
        self.config = config
        self.data_fetcher_service = ForexDataAPIFetcherService()
        self.sentiment_analyzer_service = SentimentAnalyzerService()
        self.data_frame_saver = DataFrameSaver()
        self.data_loader = DataLoader()

    def get_dates(self):
        get_a_day = GetADay()
        _, end_date = get_a_day.get_days()
        start_date = '2018-01-01'
        return start_date, end_date

    def build_fetcher_services(self, start_date, end_date):
        forex_data_fetcher_service = ForexPriceDataChunkFetcher(
            forex_data_fetcher_service=self.data_fetcher_service, api_key=self.api_key,
            start_date=start_date, end_date=end_date)

        calendar_data_fetcher_service = EconomicCalendarFetcherService(
            data_fetcher_service=self.data_fetcher_service, api_key=self.api_key,
            start_date=start_date, end_date=end_date)

        news_fetcher_service = NewsFetcherService(
            self.sentiment_analyzer_service, api_key=self.api_key,
            start_date=start_date, end_date=end_date)

        economic_indicators_fetcher_service = EconomicIndicatorsFetcherService(
            data_fetcher_service=self.data_fetcher_service, api_key=self.api_key,
            start_date=start_date, end_date=end_date)

        return (
            forex_data_fetcher_service,
            calendar_data_fetcher_service,
            news_fetcher_service,
            economic_indicators_fetcher_service
        )

    def build_feature_engineering_services(self):
        feature_engineering_pipeline = FeatureEngineeringPipeline()
        calendar_feature_engineering_pipeline = EconomicCalendarPreprocessingPipeline()
        return feature_engineering_pipeline, calendar_feature_engineering_pipeline

    def build_merger_services(self, start_date, end_date):
        news_merger_pipeline = NewsMergerPipeline(start_date=start_date, end_date=end_date, live_data=True)
        indicators_merger_pipeline = EconomicIndicatorDataMergerPipeline(is_live_data=False)
        return news_merger_pipeline, indicators_merger_pipeline

    def build_fetch_data_pipeline(self):
        start_date, end_date = self.get_dates()

        (forex_data_fetcher_service, calendar_data_fetcher_service, news_fetcher_service,
         economic_indicators_fetcher_service) = self.build_fetcher_services(start_date, end_date)

        feature_engineering_service, calendar_feature_engineering_service = self.build_feature_engineering_services()
        news_merger_service, indicators_merger_service = self.build_merger_services(start_date, end_date)

        return FetchDataPipeline(
            start_date=start_date, end_date=end_date,
            forex_data_fetcher_service=forex_data_fetcher_service,
            calendar_data_fetcher_service=calendar_data_fetcher_service,
            news_fetcher_service=news_fetcher_service,
            economic_indicators_fetcherService=economic_indicators_fetcher_service,
            price_feature_engineering_service=feature_engineering_service,
            calendar_feature_engineering_service=calendar_feature_engineering_service,
            trading_news_data_merger_service=news_merger_service,
            data_merger_service=DataMergerService(),
            indicators_merger_service=indicators_merger_service,
            config=self.config,
            data_frame_saver=self.data_frame_saver,
            data_loader=self.data_loader
        )
