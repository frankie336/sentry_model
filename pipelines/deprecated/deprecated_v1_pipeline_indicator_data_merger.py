import time

from ForexMastermind.ML.tools import DataLoader
from ForexMastermind.ML.services.service_indicator_data_merger import IndicatorDataMergerService
from ForexMastermind.ML.config import ForexMastermindConfig
from ForexMastermind.ML.tools import GetADay


class EconomicIndicatorDataMergerPipeline:
    def __init__(self, start_date, end_date, unseen_data=False, live_data=False):
        self.start_date = start_date
        self.end_date = end_date
        self.unseen_data = unseen_data
        self.live_data = live_data

        self.config = ForexMastermindConfig(start_date=self.start_date, end_date=self.end_date,
                                            unseen_start_date=self.start_date, unseen_end_date=self.end_date)

        self.data_path = self.config.get_next_data_path(data_type='ForexData', is_unseen=self.unseen_data,
                                                        live_data=self.live_data, training_data_version='7')

        self.indicator_data_path = self.config.get_economic_indicators_data_path(version='1',
                                                                                 is_unseen=self.unseen_data,
                                                                                 live_data=self.live_data)

        data_loader = DataLoader()
        self.indicator_data = data_loader.load_data(data_path=self.indicator_data_path)

        self.data = data_loader.load_data(data_path=self.indicator_data_path)

    def run(self, live_data):

        data_merger = IndicatorDataMergerService(start_date=self.start_date, end_date=self.end_date,
                                                 data=self.data, indicator_data=self.indicator_data,
                                                 live_data=self.live_data, unseen_data=self.unseen_data)

        if self.live_data:
            return data_merger.merge_data(live_data_feed=live_data)

