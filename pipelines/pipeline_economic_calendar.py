# pipeline_economic_calendar.py
from ForexMastermind.ML.services.service_feature_engineering import CategoricalFeatureService, DateTimeFeatureService


class EconomicCalendarPreprocessingPipeline:

    def __init__(self, data):

        self.data = data

    def process(self):

        categorical_feature = CategoricalFeatureService(data=self.data)

        date_time = DateTimeFeatureService(data=self.data)
        self.data = date_time.convert_to_datetime()
        self.data = categorical_feature.encode_events_and_impact()

        return self.data
