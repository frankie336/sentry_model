#pipeline_feature_engineering.py
import time

from ForexMastermind.ML.pipelines.pipeline_indicator_data_merger import EconomicIndicatorDataMergerPipeline
from ForexMastermind.ML.services.service_feature_engineering import (DateTimeFeatureService, PriceFeatureService,
                                                                     TechnicalIndicatorService,
                                                                     SentimentFeatureService,
                                                                     LaggedFeatureService, TargetFeatureService)

from ForexMastermind.ML.config import ForexMastermindConfig


class FeatureEngineeringPipeline:

    def __init__(self, start_date=None, end_date=None, live_data=False):
        self.start_date = start_date
        self.end_date = end_date
        self.live_data = live_data
        self.config = ForexMastermindConfig()
        self.data = None

    def apply_feature_engineering(self, data):
        # Sequence the feature processing using the services
        datetime_service = DateTimeFeatureService(data)
        self.data = datetime_service.add_datetime_features()

        price_service = PriceFeatureService(self.data)
        self.data = price_service.add_price_features()

        # Create lagged features
        lagged_feature_service = LaggedFeatureService(data=self.data)
        self.data = lagged_feature_service.create_lagged_features()

        technical_service = TechnicalIndicatorService(self.data)
        self.data = technical_service.add_technical_indicators()

        # add_target
        if not self.live_data:
            target_service = TargetFeatureService(data=self.data)
            self.data = target_service.add_binary_target()

        return self.data








