import os
import time
from ForexMastermind.ML.services.service_indicator_data_merger import IndicatorDataMergerService, IndicatorDataMergerServiceLive
from ForexMastermind.ML.config import ForexMastermindConfig
from ForexMastermind.ML.tools.tools import DataLoader


class EconomicIndicatorDataMergerPipeline:
    def __init__(self, unseen_data=False, is_live_data=False):
        self.unseen_data = unseen_data
        self.is_live_data = is_live_data
        self.config = ForexMastermindConfig()

    def merge_data(self, live_data_feed=None, indicator_data=None):
        # Determine the paths for the price and indicator data based on the type of data
        temp_dir = self.config.get_data_processing_temp_directory()
        data_path = os.path.join(temp_dir, 'price_calendar_merged.csv')
        indicator_data_path = os.path.join(temp_dir, 'econ_indicator_data.csv')

        # Determine the appropriate data target path based on the data type
        if self.unseen_data:
            data_target_path = self.config.get_next_data_path(
                data_type='ForexData', training_data_version='6', is_unseen=True)
        else:
            data_target_path = self.config.get_next_data_path(
                data_type='ForexData', training_data_version='11')

        # Merge logic for live data
        if self.is_live_data:
            if indicator_data.empty:
                pass

                #indicator_data_path = self.config.get_economic_indicators_data_path(version='1', is_live_data=True)
                #indicator_data = DataLoader().load_data(data_path=indicator_data_path)

            data_merger = IndicatorDataMergerServiceLive(live_data=self.is_live_data, unseen_data=self.unseen_data)
            return data_merger.merge_data(live_data_feed=live_data_feed, indicator_data=indicator_data)

        # Merge logic for unseen or historical data
        else:
            data_merger = IndicatorDataMergerService(
                is_live_data=self.is_live_data, unseen_data=self.unseen_data,
                config=self.config, training_data_path=data_target_path)

            return data_merger.merge_data_in_chunks(data_path=data_path, indicator_data_path=indicator_data_path)





