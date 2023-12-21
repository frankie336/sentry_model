import time

import pandas as pd
import talib
from tqdm import tqdm
from joblib import dump, load
from ForexMastermind.ML.OLD_config import PATHS


class PreprocessForexData:

    def __init__(self, data=None):
        if isinstance(data, str):
            self.data = pd.read_csv(data)
        elif isinstance(data, pd.DataFrame):
            self.data = data
        else:
            raise ValueError("Invalid data format. Expected a filepath or pandas DataFrame.")

    def convert_to_datetime(self):
        """Converts the 'date' column to datetime format."""
        self.data['date'] = pd.to_datetime(self.data['date'])

    def compute_mid_price(self):
        """Computes the mid-price using the average of 'high' and 'low' columns."""
        self.data['Mid-price'] = (self.data['high'].astype(float) + self.data['low'].astype(float)) / 2

    def compute_price_dynamics(self):
        """Computes the price dynamics using 'close' and 'open' columns."""
        self.data['Price Dynamics'] = self.data['close'].astype(float) - self.data['open'].astype(float)

    def create_lagged_features(self, n_lags=60):
        """Creates lagged features for specific columns."""
        columns_to_lag = ['Mid-price', 'volume']
        original_number_of_columns = self.data.shape[1]
        lagged_frames = [self.data]

        for col in columns_to_lag:
            for lag in range(1, n_lags + 1):
                lagged_col_name = f'{col}-lag-{lag}'
                lagged_frame = self.data[col].shift(lag).to_frame(lagged_col_name)
                lagged_frames.append(lagged_frame)

        self.data = pd.concat(lagged_frames, axis=1)

        # Validation
        expected_columns = original_number_of_columns + len(columns_to_lag) * n_lags
        assert self.data.shape[1] == expected_columns, "Mismatch in expected number of columns."

    def add_technical_indicators(self):
        """Adds various technical indicators in an optimized manner to reduce DataFrame fragmentation."""
        indicators = {
            'SMA': self.data['Mid-price'].rolling(window=14).mean(),
            'RSI': talib.RSI(self.data['Mid-price'], timeperiod=14),
            # MACD
            **dict(zip(['MACD', 'MACD_signal', 'MACD_hist'], talib.MACD(self.data['Mid-price']))),
            # Bollinger Bands
            **dict(zip(['upper_band', 'middle_band', 'lower_band'], talib.BBANDS(self.data['Mid-price'], timeperiod=20))),
            # Stochastic Oscillator
            **dict(zip(['slowk', 'slowd'], talib.STOCH(self.data['high'], self.data['low'], self.data['close']))),
            # ATR
            'ATR': talib.ATR(self.data['high'], self.data['low'], self.data['close'], timeperiod=14),
            # OBV
            'OBV': talib.OBV(self.data['close'], self.data['volume']),
            # ADX
            'ADX': talib.ADX(self.data['high'], self.data['low'], self.data['close'], timeperiod=14),
            # CCI
            'CCI': talib.CCI(self.data['high'], self.data['low'], self.data['close'], timeperiod=14),
            # MFI
            'MFI': talib.MFI(self.data['high'], self.data['low'], self.data['close'], self.data['volume'], timeperiod=14)
        }

        # Concatenate all indicators with the original DataFrame
        self.data = pd.concat([self.data, pd.DataFrame(indicators)], axis=1)

    def drop_nan_rows(self):
        """
        Drops rows with NaN values in any column.
        """
        self.data.dropna(inplace=True)

    def add_target(self):
        """
        Adds a target column to the DataFrame.
        The target is 1 if the next period's open is higher than the current period's close, else 0.
        """
        # Shift the 'open' column to align the next period's open with the current row
        next_open = self.data['open'].shift(-1)

        # Compare the next period's open with the current period's close
        target = (next_open > self.data['close']).astype(int)

        # Create a DataFrame for the target and concatenate it to avoid fragmentation
        target_df = pd.DataFrame({'Target': target}, index=self.data.index)
        self.data = pd.concat([self.data, target_df], axis=1)

        # drop last row because there is no prediction possible after this segment
        self.data = self.data.drop(self.data.tail(1).index)

    def encode_events_and_impact(self):
        # Encode events
        unique_events = self.data['event'].dropna().unique()
        event_map = {event: idx for idx, event in enumerate(unique_events, start=1)}
        event_map[None] = -1  # Special code for rows with no events
        self.data['event_encoded'] = self.data['event'].map(event_map).fillna(-1)

        # Encode impact
        unique_impacts = self.data['impact'].dropna().unique()
        impact_map = {impact: idx for idx, impact in enumerate(unique_impacts, start=1)}
        impact_map[None] = -1  # Special code for rows with no impact data
        self.data['impact_encoded'] = self.data['impact'].map(impact_map).fillna(-1)

        # Handle NaN values in other event-dependent columns
        event_dependent_columns = ['actual', 'previous', 'change', 'changePercentage', 'estimate']
        for col in event_dependent_columns:
            self.data[col] = self.data[col].fillna(-1)  # Replace NaN with -1 or another suitable default

    def create_previous_period_features(self):
        # Create a new DataFrame with the shifted data
        shifted_data = pd.DataFrame({
            'prev_Mid-price': self.data['Mid-price'].shift(1),
            'prev_low': self.data['low'].shift(1),
            'prev_high': self.data['high'].shift(1),
            'prev_close': self.data['close'].shift(1),
            'prev_volume': self.data['volume'].shift(1),
            'prev_open': self.data['open'].shift(1),
        })

        # Concatenate the new DataFrame with the original data
        self.data = pd.concat([self.data, shifted_data], axis=1)

    def balance_samples(self):

        # Assuming 'target' is the binary target column in your DataFrame 'df'
        count_class_0, count_class_1 = self.data['Target'].value_counts()

        # Divide by class
        df_class_0 = self.data[self.data['Target'] == 0]
        df_class_1 = self.data[self.data['Target'] == 1]

        df_class_0_under = df_class_0.sample(count_class_1)

        self.data = pd.concat([df_class_0_under, df_class_1], axis=0)

        print('Random under-sampling:')
        print(self.data['Target'].value_counts())

        return self.data

    def aggregate_sentiment_scores(self):

        # Convert 'date' to datetime and remove timezone information for aggregation
        self.data['date'] = pd.to_datetime(self.data['date']).dt.tz_localize(None)

        # Group by 'date' and aggregate sentiment scores
        aggregated_news = self.data.groupby('date').agg({
            'sentiment_score': 'mean',  # Replace 'mean' with 'sum' if you want the total score
            'news_headline': lambda headlines: ' // '.join(headlines)
        }).reset_index()

        return aggregated_news

    def merge_trading_with_news(self, trading_df, news_df):
        # Convert the 'date' column in both DataFrames to datetime without timezone
        trading_df['date'] = pd.to_datetime(trading_df['date']).dt.tz_localize(None)
        news_df['date'] = pd.to_datetime(news_df['date']).dt.tz_localize(None)

        # Sort news_data by date
        news_df = news_df.sort_values('date')

        # Initialize an empty DataFrame to hold the merged results
        merged_results = pd.DataFrame()

        # Assuming 'currency_pair' is the column that identifies each currency pair in trading_df
        for currency_pair, group in trading_df.groupby('currency'):
            # Sort each group by date
            group_sorted = group.sort_values('date')

            # Merge the group with news_data using pd.merge_asof
            merged_group = pd.merge_asof(group_sorted, news_df, left_on='date', right_on='date', direction='forward')

            # Append the result to the merged_results DataFrame
            merged_results = pd.concat([merged_results, merged_group], ignore_index=True)

        # Fill in default values for rows without corresponding news data
        default_values = {'sentiment_score': -1, 'news_headline': 'None'}  # Replace with relevant column names
        merged_results.fillna(value=default_values, inplace=True)

        print(merged_results.head())  # Print the first few rows of the merged dataframe
        self.save_df(df=merged_results, path=PATHS['MIN']['NEXT_TRAINING_DATA'])

        return merged_results

    def save_df(self, df, path):

        df.to_csv(path, index=False)
        print(f"DataFrame saved successfully to {path}")

    def return_final_data(self):

        return self.data

    def _shared_preprocessing_steps(self):
        """Shared preprocessing steps for both training and new data."""
        self.convert_to_datetime()
        self.compute_mid_price()
        self.compute_price_dynamics()
        self.create_lagged_features()
        self.add_technical_indicators()
        self.create_previous_period_features()
        self.drop_nan_rows()
        self.add_target()

        return self.data

    def _economic_data_steps(self):

        self.convert_to_datetime()
        self.encode_events_and_impact()

        return self.data

    def run_shared_steps(self):

        return self._shared_preprocessing_steps()

    def run_economic_steps(self):

        return self._economic_data_steps()

    def run_news_merger_steps(self):

        return self.merge_trading_with_news(news_df=pd.read_csv(PATHS['MIN']['NEWS_DATA']),
                                            trading_df=pd.read_csv(PATHS['MIN']['TRAINING_DATA']))











































