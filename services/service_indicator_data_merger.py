import time
# service_indicator_data_merger.py
import pandas as pd
import os
from tqdm import tqdm
from ForexMastermind.ML.tools.constants import INDICATORS


class IndicatorDataMergerService:
    def __init__(self, training_data_path,  config, unseen_data=False, is_live_data=False):

        self.training_data_path = training_data_path
        self.config = config
        self.unseen_data = unseen_data
        self.is_live_data = is_live_data

    def _normalize_dates(self, df):
        df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None)
        df.sort_values('date', inplace=True)
        df.ffill(inplace=True)

    def merge_data_in_chunks(self, data_path, indicator_data_path, chunk_size=100000):

        indicator_data = pd.read_csv(indicator_data_path)
        self._normalize_dates(indicator_data)

        # Fill NaN values for specific indicators
        indicators = INDICATORS.copy()
        for indicator in indicators:
            if indicator in indicator_data.columns:
                indicator_data[indicator].fillna(method='ffill', inplace=True)  # forward fill the indicator data

        # Get the last date from the indicator data and the first date from the trading data
        last_indicator_date = indicator_data['date'].max()
        trading_data_first_date = pd.read_csv(data_path, nrows=1)['date'].min()
        trading_data_first_date = pd.to_datetime(trading_data_first_date).tz_localize(None)

        # Extend the indicator data if necessary
        if trading_data_first_date > last_indicator_date:
            extension = pd.DataFrame({'date': pd.date_range(start=last_indicator_date + pd.Timedelta(days=1),
                                                            end=trading_data_first_date, freq='D')})
            indicator_data = pd.concat([indicator_data, extension], ignore_index=True)
            indicator_data.ffill(inplace=True)  # forward fill the new rows

        temp_files = []
        temp_dir = self.config.get_data_processing_temp_directory()
        chunks_dir = os.path.join(temp_dir, 'chunks')

        for i, chunk in enumerate(tqdm(pd.read_csv(data_path, chunksize=chunk_size), desc="Merging chunks")):
            self._normalize_dates(chunk)
            chunk['date'] = pd.to_datetime(chunk['date']).dt.tz_localize(None)
            chunk.sort_values('date', inplace=True)

            merged_chunk = pd.merge_asof(chunk, indicator_data, on='date', direction='forward')
            merged_chunk.ffill(inplace=True)

            # Generate a unique file path for each chunk
            temp_file_path = os.path.join(chunks_dir, f'chunk_{i}.csv')
            merged_chunk.to_csv(temp_file_path, index=False)
            temp_files.append(temp_file_path)

        # Concatenate all chunks into a single DataFrame
        concatenated_df = pd.concat([pd.read_csv(f) for f in temp_files], ignore_index=True)

        # Save the concatenated DataFrame
        concatenated_df.to_csv(self.training_data_path, index=False)

        # Clean up the temporary files
        for f in temp_files:
            os.remove(f)

        return self.training_data_path


class IndicatorDataMergerServiceLive:

    def __init__(self, unseen_data=False, live_data=False):

        self.unseen_data = unseen_data
        self.live_data = live_data
        self.data = None
        self.indicator_data = None

    def merge_data(self, is_live_data=True, live_data_feed=None,  indicator_data=None):

        self.data = live_data_feed
        self.indicator_data = indicator_data
        self._normalize_dates()
        self.indicator_data = indicator_data

        return self._merge_indicators()

    def _normalize_dates(self):

        # Convert the 'date' column in both DataFrames to datetime without timezone
        self.data['date'] = pd.to_datetime(self.data['date']).dt.tz_localize(None)
        self.indicator_data['date'] = pd.to_datetime(self.indicator_data['date']).dt.tz_localize(None)

        # Sort indicator_data by date
        self.indicator_data.sort_values('date', inplace=True)

        # Forward fill the indicator_data
        self.indicator_data.ffill(inplace=True)

    def _merge_indicators(self):
        # Normalize the dates
        self.data['date'] = pd.to_datetime(self.data['date']).dt.tz_localize(None)
        self.indicator_data['date'] = pd.to_datetime(self.indicator_data['date']).dt.tz_localize(None)

        # Sort both DataFrames by the 'date' column
        self.data.sort_values('date', inplace=True)
        self.indicator_data.sort_values('date', inplace=True)

        # Fill NaN values for specific indicators in indicator_data
        indicators = INDICATORS.copy()
        for indicator in indicators:
            if indicator in self.indicator_data.columns:
                self.indicator_data[indicator].fillna(method='ffill', inplace=True)  # forward fill the indicator data

        # Extend the indicator data to the current date, if necessary
        last_indicator_date = self.indicator_data['date'].max()
        last_data_date = self.data['date'].max()
        if last_data_date > last_indicator_date:
            # If the last data date is greater than the last indicator date, extend the indicator data
            extension = pd.DataFrame(
                {'date': pd.date_range(start=last_indicator_date + pd.Timedelta(days=1), end=last_data_date, freq='D')})
            self.indicator_data = pd.concat([self.indicator_data, extension], ignore_index=True)
            self.indicator_data.ffill(inplace=True)  # forward fill the new rows

        # Merge with a forward fill
        merged_results = pd.merge_asof(self.data, self.indicator_data, on='date', direction='forward')

        # Forward fill the results to carry last known values
        merged_results.ffill(inplace=True)

        print(merged_results)

        return merged_results


