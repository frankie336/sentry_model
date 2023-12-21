import time
from scipy.stats import linregress
import numpy as np
import pandas as pd
import talib
from ForexMastermind.ML.config import ForexMastermindConfig
from sklearn.preprocessing import LabelEncoder
from ForexMastermind.ML.tools import DataLoader, DataFrameSaver
from ForexMastermind.ML.tools import GetADay, TimeOfDayEncoder


class DateTimeFeatureService:
    def __init__(self, data):
        self.data = data

    def add_datetime_features(self):
        self._convert_to_datetime()
        self._add_time_of_day()
        return self.data

    def _convert_to_datetime(self):
        """Converts the 'date' column to datetime format."""
        self.data['date'] = pd.to_datetime(self.data['date'])

    def _add_time_of_day(self):
        """Adds a column with the time of day encoded as a fraction."""
        self.data['time_of_day'] = (self.data['date'].dt.hour * 3600 +
                                    self.data['date'].dt.minute * 60 +
                                    self.data['date'].dt.second) / (24 * 3600)


class PriceFeatureService:
    def __init__(self, data):
        self.data = data

    def add_price_features(self):
        self._compute_mid_price()
        self._compute_price_dynamics()
        self._add_linear_regression_slope(window_size=5)
        return self.data

    def _compute_mid_price(self):
        """Computes the mid-price using the average of 'high' and 'low' columns."""
        self.data['Mid-price'] = (self.data['high'].astype(float) + self.data['low'].astype(float)) / 2

    def _compute_price_dynamics(self):
        """Computes the price dynamics using 'close' and 'open' columns."""
        self.data['Price Dynamics'] = self.data['close'].astype(float) - self.data['open'].astype(float)

    def _add_linear_regression_slope(self, window_size=5):
        """
        Adds the slope of the linear regression line of the 'Mid-price' over a rolling window.
        """
        def get_slope(data):
            """Helper function to apply linear regression and return the slope."""
            regression = linregress(np.arange(len(data)), data.values)
            return regression.slope

        # Use rolling window and apply the helper function
        self.data['LR_Slope'] = self.data['Mid-price'].rolling(window=window_size).apply(get_slope, raw=False)

    def _create_previous_period_features(self):
        """
        Creates new columns in the DataFrame by shifting certain price-related columns by one period.

        This method generates lagged (previous period) features for specific price-related columns such as 'Mid-price',
        'low', 'high', 'close', 'volume', and 'open'. These lagged features provide insight into the previous period's
        values, which can be useful for time-series forecasting and analysis.

        The method appends these new lagged feature columns to the existing DataFrame, enhancing the dataset with
        historical context that may be relevant for predictive modeling.

        The resulting DataFrame includes the following additional columns:
        - 'prev_Mid-price': The mid-price from the previous period.
        - 'prev_low': The low price from the previous period.
        - 'prev_high': The high price from the previous period.
        - 'prev_close': The closing price from the previous period.
        - 'prev_volume': The trading volume from the previous period.
        - 'prev_open': The opening price from the previous period.
        """
        shifted_data = pd.DataFrame({
            'prev_Mid-price': self.data['Mid-price'].shift(1),
            'prev_low': self.data['low'].shift(1),
            'prev_high': self.data['high'].shift(1),
            'prev_close': self.data['close'].shift(1),
            'prev_volume': self.data['volume'].shift(1),
            'prev_open': self.data['open'].shift(1),
        })

        self.data = pd.concat([self.data, shifted_data], axis=1)


class LaggedFeatureService:
    def __init__(self, data):
        self.data = data

    def create_lagged_features(self, n_lags=60):
        """
        Creates lagged features for specific columns.

        Args:
            n_lags (int): Number of lagged features to create.

        Returns:
            pd.DataFrame: DataFrame with lagged features added.
        """
        columns_to_lag = ['Mid-price', 'volume']
        original_number_of_columns = self.data.shape[1]
        lagged_frames = [self.data]

        for col in columns_to_lag:
            for lag in range(1, n_lags + 1):
                lagged_col_name = f'{col}-lag-{lag}'
                lagged_frame = self.data[col].shift(lag).to_frame(lagged_col_name)
                lagged_frames.append(lagged_frame)

        self.data = pd.concat(lagged_frames, axis=1)

        # Validate the number of columns in the resulting DataFrame
        expected_columns = original_number_of_columns + len(columns_to_lag) * n_lags
        assert self.data.shape[1] == expected_columns, "Mismatch in expected number of columns."

        return self.data


class TechnicalIndicatorService:
    def __init__(self, data):
        self.data = data

    def add_technical_indicators(self):

        self._add_sma()
        self._add_rsi()
        self._add_macd()
        self._add_bollinger_bands()
        self._add_stochastic_oscillator()
        self._add_atr()
        self._add_obv()
        self._add_adx()
        self._add_cci()
        self._add_mfi()

        return self.data

    def _add_sma(self):
        # Simple Moving Average
        self.data['SMA'] = self.data['Mid-price'].rolling(window=14).mean()

    def _add_rsi(self):
        # Relative Strength Index
        self.data['RSI'] = talib.RSI(self.data['Mid-price'], timeperiod=14)

    def _add_macd(self):
        # Moving Average Convergence Divergence
        macd, macdsignal, macdhist = talib.MACD(self.data['Mid-price'])
        self.data['MACD'] = macd
        self.data['MACD_signal'] = macdsignal
        self.data['MACD_hist'] = macdhist

    def _add_bollinger_bands(self):
        # Bollinger Bands
        upperband, middleband, lowerband = talib.BBANDS(self.data['Mid-price'], timeperiod=20)

        self.data['upper_band'] = upperband
        self.data['middle_band'] = middleband
        self.data['lower_band'] = lowerband

    def _add_stochastic_oscillator(self):
        # Stochastic Oscillator
        slowk, slowd = talib.STOCH(self.data['high'], self.data['low'], self.data['close'])

        self.data['slowk'] = slowk
        self.data['slowd'] = slowd

    def _add_atr(self):
        # Average True Range
        self.data['ATR'] = talib.ATR(self.data['high'], self.data['low'], self.data['close'], timeperiod=14)

    def _add_obv(self):
        # On-Balance Volume
        self.data['OBV'] = talib.OBV(self.data['close'], self.data['volume'])

    def _add_adx(self):
        # Average Directional Movement Index
        self.data['ADX'] = talib.ADX(self.data['high'], self.data['low'], self.data['close'], timeperiod=14)

    def _add_cci(self):
        # Commodity Channel Index
        self.data['CCI'] = talib.CCI(self.data['high'], self.data['low'], self.data['close'], timeperiod=14)

    def _add_mfi(self):
        # Money Flow Index
        self.data['MFI'] = talib.MFI(self.data['high'], self.data['low'], self.data['close'], self.data['volume'],
                                     timeperiod=14)


class NewsAggregationService:
    def __init__(self, data):
        self.data = data

    def aggregate_sentiment_scores(self):
        """
        Aggregates sentiment scores by date.

        Converts 'date' from string to datetime, groups the data by date,
        and calculates the mean sentiment score for each date. It also aggregates
        all news headlines for the same date into a single string separated by ' // '.

        Returns:
            pd.DataFrame: Aggregated news data with the mean sentiment score and concatenated headlines for each date.
        """
        self.data['date'] = pd.to_datetime(self.data['date']).dt.tz_localize(None)

        # Group by 'date' and aggregate sentiment scores
        aggregated_news = self.data.groupby('date').agg({
            'sentiment_score': 'mean',
            'news_headline': lambda headlines: ' // '.join(headlines)
        }).reset_index()

        return aggregated_news


class CategoricalFeatureService:
    def __init__(self, data):
        self.data = data

    def encode_currency_pairs(self):
        """
        Encodes the 'currency' column with integers using label encoding.

        Returns:
            pd.DataFrame: DataFrame with the encoded currency column.
        """
        label_encoder = LabelEncoder()
        currency_encoded = label_encoder.fit_transform(self.data['currency'])

        # Convert the NumPy array to a DataFrame
        currency_encoded_df = pd.DataFrame({'CurrencyPair_encoded': currency_encoded})

        # Concatenate the new DataFrame with the existing DataFrame
        self.data = pd.concat([self.data, currency_encoded_df], axis=1)

        return self.data

    def _encode_events_and_impact(self):
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

        return self.data


class TargetFeatureService:
    def __init__(self, data):
        self.data = data

    def add_binary_target(self):
        """
        Adds a binary target column to the DataFrame.
        The target is 1 if the next period's open is higher than the current period's close, else 0.

        Returns:
            pd.DataFrame: DataFrame with the target column added.
        """
        target_col = (self.data['open'].shift(-1) > self.data['close']).astype(int)
        self.data = pd.concat([self.data, target_col.rename('Target')], axis=1)
        self.data = self.data.iloc[:-1]

        return self.data


class SentimentFeatureService:
    def __init__(self, data):
        self.data = data

    def add_sentiment_features(self):
        # add methods to process sentiment features
        pass


class FeatureEngineeringPipeline:
    def __init__(self, data, live_data=False):
        self.data = data
        self.live_data = live_data

    def process(self):
        # Sequence the feature processing using the services
        datetime_service = DateTimeFeatureService(self.data)
        self.data = datetime_service.add_datetime_features()

        price_service = PriceFeatureService(self.data)
        self.data = price_service.add_price_features()

        technical_service = TechnicalIndicatorService(self.data)
        self.data = technical_service.add_technical_indicators()

        sentiment_service = SentimentFeatureService(self.data)
        self.data = sentiment_service.add_sentiment_features()

        # Handle live data conditions and others as needed
        # ...

        return self.data
