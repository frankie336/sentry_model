# forex_data_fetcher_pipeline.py
# Standard library imports
import asyncio
import os
import random
import time
from datetime import datetime, timedelta

# Related third-party imports
import pandas as pd
import requests
from textblob import TextBlob
from tqdm import tqdm

# Local application/library-specific imports
from ForexMastermind.ML.tools.constants import CURRENCY_PAIRS, INDICATORS
from ForexMastermind.ML.tools.tools import DataFrameSaver
from ForexMastermind.ML.services.service_feature_engineering import CategoricalFeatureService, DateTimeFeatureService


class EconomicCalendarPreprocessingPipeline:

    def __init__(self, data=None):

        self.data = data

    def apply_feature_engineering(self, data):

        print('This is the calendar method')
        date_time = DateTimeFeatureService(data=data)
        data = date_time.convert_to_datetime()

        categorical_feature = CategoricalFeatureService(data=data)
        data = categorical_feature.encode_events_and_impact()

        print(data)

        return data


class IndicatorsDataMergingService:
    def __init__(self, indicators_merger_pipeline):
        self.indicators_merger_pipeline = indicators_merger_pipeline

    def merge_trading_with_indicators(self, live_data, indicator_data):

        if self.indicators_merger_pipeline is None:
            raise ValueError("IndicatorsDataMergingService is not initialized.")

        return self.indicators_merger_pipeline.run(live_data=live_data, indicator_data=indicator_data)


class DataMergerService:
    def __init__(self):
        pass

    def merge_data_asof(self, df1, df2, on, direction='forward'):
        """
        Merges two dataframes based on a key using an 'asof' merge. This is particularly useful for
        time-series data.

        Args:
            df1 (pd.DataFrame): The first dataframe.
            df2 (pd.DataFrame): The second dataframe.
            on (str): The column name to merge on.
            direction (str, optional): The merge direction, 'forward', 'backward', or 'nearest'. Default is 'forward'.

        Returns:
            pd.DataFrame: A merged dataframe.
        """
        # Ensure both DataFrames have 'date' in datetime format
        df1[on] = pd.to_datetime(df1[on])
        df2[on] = pd.to_datetime(df2[on])

        # Ensure DataFrames are sorted by 'date'
        df1_sorted = df1.sort_values(on)
        df2_sorted = df2.sort_values(on)

        # Perform the merge
        merged_df = pd.merge_asof(df1_sorted, df2_sorted, on=on, direction=direction)

        return merged_df


class SentimentAnalyzerService:
    """
    Performs sentiment analysis on text data.
    """

    @staticmethod
    def analyze_sentiment(text):
        """
        Analyzes the sentiment of the given text.

        Args:
            text (str): The text to analyze.

        Returns:
            float: The sentiment polarity score (-1.0 to 1.0).
        """
        blob = TextBlob(text)

        return blob.sentiment.polarity


class ForexDataAPIFetcherService:
    def __init__(self, api_key=None):
        self.api_key = api_key

    def fetch_data(self, endpoint=None, params=None):
        """
        Fetches data from the specified Financial Modeling Prep API endpoint.

        Args:
            endpoint (str): The complete API endpoint URL.
            params (dict, optional): Additional query parameters for the API request.

        Returns:
            data: A Json abject containing the fetched data, or an empty DataFrame in case of failure.
        """
        params = params or {}
        params['apikey'] = self.api_key  # Add the API key to the parameters

        response = requests.get(endpoint)

        if response.status_code == 200:
            data = response.json()
            return data
        else:
            print(f"Failed to fetch data from {endpoint}: {response.status_code}")
            print('fetch_data(self, endpoint=None, params=None)')
            return pd.DataFrame()


class ForexPriceDataChunkFetcher:
    def __init__(self, forex_data_fetcher_service, start_date, end_date, api_key, live_data=False):
        self.forex_data_fetcher_service = forex_data_fetcher_service
        self.start_date = start_date
        self.end_date = end_date
        self.api_key = api_key
        self.live_data = live_data

    def fetch_data_in_chunks(self, pair):

        all_data = []
        date_range_generator = DateRangeGenerator()

        for start, end in tqdm(
                date_range_generator.generate_date_ranges(self.start_date, self.end_date, timedelta(days=30)),
                desc=f"Fetching {pair}"):

            retry_count = 0
            max_retries = 3
            successful = False

            while retry_count < max_retries and not successful:
                endpoint = f"https://financialmodelingprep.com/api/v3/historical-chart/1min/{pair}?from={start.strftime('%Y-%m-%d')}&to={end.strftime('%Y-%m-%d')}&apikey={self.api_key}"

                try:
                    response = requests.get(endpoint)
                    if response.status_code == 200:
                        forex_data = response.json()
                        if forex_data:
                            df = pd.DataFrame(forex_data)
                            df['Pair'] = pair
                            all_data.append(df)
                        successful = True
                    else:
                        print(f"Failed to fetch data, status code: {response.status_code}")
                        retry_count += 1
                        time.sleep(5)  # Wait for 5 seconds before retrying
                except Exception as e:
                    print(f"Error fetching data: {e}")
                    retry_count += 1
                    time.sleep(5)

        return pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()


class EconomicCalendarFetcherService:
    """
       Fetches economic calendar data within a specified date range.

       Attributes:
           start_date (str): The start date for the data range in 'YYYY-MM-DD' format.
           end_date (str): The end date for the data range in 'YYYY-MM-DD' format.
           api_key (str): The API key for accessing the data source.
    """
    def __init__(self, data_fetcher_service, api_key, start_date, end_date):

        self.data_fetcher_service = data_fetcher_service

        self.start_date = start_date
        self.end_date = end_date

        self.api_key = api_key

    def fetch_calendar_data(self):

        # Sets up the params for the base fetcher service
        # uses instance of the main fetcher to fetch data
        params = {'start_date': self.start_date, 'end_date': self.end_date, 'apikey': self.api_key}
        endpoint = f"https://financialmodelingprep.com/api/v3/economic_calendar?from={self.start_date}&to={self.end_date}&apikey={self.api_key}"
        data = self.data_fetcher_service.fetch_data(params=params, endpoint=endpoint)

        df = pd.DataFrame(data)

        return df


class NewsFetcherService:
    """
    Fetches forex news data relevant to forex markets within a specified date range.

    Attributes:
        start_date (str): The start date for the news data in 'YYYY-MM-DD' format.
        end_date (str): The end date for the news data in 'YYYY-MM-DD' format.
        api_key (str): The API key for accessing the news data source.
    """
    def __init__(self, sentiment_analyzer_service, api_key, start_date, end_date):

        self.start_date = start_date
        self.end_date = end_date
        self.api_key = api_key
        self.news_item = ''

        self.sentiment_analyzer_service = sentiment_analyzer_service

    def fetch_news_data(self):
        page = 0
        all_news = []
        while True:
            print(page)
            # Adding a random sleep here
            #time.sleep(random.uniform(0.5, 2.0))  # Sleep for a random time between 0.5 to 2 seconds

            url = f"https://financialmodelingprep.com/api/v4/forex_news?page={page}&apikey={self.api_key}"
            response = requests.get(url)
            if response.status_code == 200 and response.json():
                news_data = response.json()
                for news_item in news_data:
                    print(news_item)
                    # Adding a random sleep here
                    #time.sleep(random.uniform(0.5, 2.0))  # Sleep for a random time between 0.5 to 2 seconds

                    published_date = datetime.fromisoformat(news_item['publishedDate'][:-1])
                    if published_date < datetime.fromisoformat(self.start_date):
                        return pd.DataFrame(all_news)
                    if published_date <= datetime.fromisoformat(self.end_date):

                        # Calculate sentiment of the news text.
                        sentiment_score = self.sentiment_analyzer_service.analyze_sentiment(text=news_item['text'])

                        all_news.append({
                            'date': news_item['publishedDate'],
                            'sentiment_score': sentiment_score,
                            'news_headline': news_item.get('title', '')  # Ensure this matches the API response
                        })
                page += 1
            else:
                break

        news = pd.DataFrame(all_news)

        return news


class EconomicIndicatorsFetcherService:
    def __init__(self, data_fetcher_service, api_key, start_date, end_date):

        self.api_key = api_key
        self.start_date = start_date
        self.end_date = end_date
        self.data_fetcher_service = data_fetcher_service

    def fetch_indicator(self, name):

        # Sets up the params for the base fetcher service
        # uses instance of the main fetcher to fetch data
        params = {'start_date': self.start_date, 'end_date': self.end_date, 'apikey': self.api_key}
        endpoint = f"https://financialmodelingprep.com/api/v4/economic?name={name}&from={params['start_date']}&to={params['end_date']}&apikey={self.api_key}"

        data = self.data_fetcher_service.fetch_data(params=params, endpoint=endpoint)

        return pd.DataFrame(data)

    def fetch_all_indicators(self):

        indicators = INDICATORS.copy()

        all_data = pd.DataFrame()
        for indicator in indicators:
            data = self.fetch_indicator(indicator)
            all_data = pd.concat([all_data, data.assign(Indicator=indicator)], ignore_index=True)
        return all_data.pivot(index='date', columns='Indicator', values='value').reset_index()


class DateRangeGenerator:
    """
    This class is responsible for generating date ranges.

    Methods:
    generate_date_ranges(start, end, delta): Generates date ranges between the start and end dates with a specified interval.
    """

    @staticmethod
    def generate_date_ranges(start, end, delta):
        """
        Generates a sequence of date ranges from a specified start date to an end date with a given interval.

        Parameters:
        start (str): The start date in 'YYYY-MM-DD' format.
        end (str): The end date in 'YYYY-MM-DD' format.
        delta (timedelta): The time interval for each date range.

        Yields:
        tuple: A tuple of two datetime objects representing the start and end of each interval within the specified date range.

        Example:
        >>> date_gen = DateRangeGenerator()
        >>> for start, end in date_gen.generate_date_ranges('2020-01-01', '2020-01-10', timedelta(days=1)):
        >>>     print(start, end)
        """
        current = datetime.strptime(start, '%Y-%m-%d')
        end = datetime.strptime(end, '%Y-%m-%d')
        while current < end:
            yield current, min(current + delta, end)
            current += delta


class FeatureEngineeringService:
    def __init__(self, feature_engineering_service, live_data=False):

        self.live_data = live_data
        self.feature_engineering_service = feature_engineering_service

    def apply_feature_engineering(self, data):

        return self.feature_engineering_service.apply_feature_engineering(data=data)


class DataMergingService:
    @staticmethod
    def merge_data(forex_df, econ_df):
        forex_df_sorted = forex_df.sort_values('date')
        econ_df_sorted = econ_df.sort_values('date')
        return pd.merge_asof(forex_df_sorted, econ_df_sorted, on='date', direction='forward')


class FetchDataPipeline:

    def __init__(self, start_date, end_date, forex_data_fetcher_service, calendar_data_fetcher_service, news_fetcher_service,
                 economic_indicators_fetcherService, price_feature_engineering_service,
                 calendar_feature_engineering_service,trading_news_data_merger_service, data_merger_service,
                 indicators_merger_service, config, data_frame_saver, data_loader, is_live_data=False):

        self.start_date = start_date
        self.end_date = end_date
        self.forex_data_fetcher_service = forex_data_fetcher_service
        self.calendar_data_fetcher_service = calendar_data_fetcher_service
        self.news_fetcher_service = news_fetcher_service
        self.economic_indicators_fetcherService = economic_indicators_fetcherService
        self.price_feature_engineering_service = price_feature_engineering_service
        self.calendar_feature_engineering_service = calendar_feature_engineering_service
        self.trading_news_data_merger_service = trading_news_data_merger_service
        self.data_merger_service = data_merger_service
        self.is_live_data = is_live_data
        self.indicators_merger_service = indicators_merger_service
        self.config = config
        self.data_frame_saver = data_frame_saver
        self.data_loader = data_loader

    def fetch_price_data(self):

        currency_pairs = CURRENCY_PAIRS.copy()

        combined_data = pd.DataFrame()
        for pair in currency_pairs:
            pair_data = self.forex_data_fetcher_service.fetch_data_in_chunks(pair)
            combined_data = pd.concat([combined_data, pair_data], ignore_index=True)
        return combined_data

    def fetch_calendar_data(self):
        return self.calendar_data_fetcher_service.fetch_calendar_data()

    def fetch_news_data(self):

        return self.news_fetcher_service.fetch_news_data()

    def fetch_all_indicators(self):

        return self.economic_indicators_fetcherService.fetch_all_indicators()

    def price_feature_engineering(self, data):

        engineering = FeatureEngineeringService(
            feature_engineering_service=self.price_feature_engineering_service, live_data=False)

        return engineering.apply_feature_engineering(data=data)

    def calendar_feature_engineering(self, data):

        engineering = FeatureEngineeringService(
            feature_engineering_service=self.calendar_feature_engineering_service, live_data=False)

        return engineering.apply_feature_engineering(data=data)

    def merge_data_asof(self, df1, df2, on):

        return self.data_merger_service.merge_data_asof(df1, df2, on)

    def merge_news(self, data, news_data):

         return self.trading_news_data_merger_service.merge_news(data=data, news_data=news_data)

    def saver_service(self, combined_df):

        data_path = self.config.get_next_data_path(data_type='ForexData', is_unseen=False,
                                                        live_data=False, training_data_version='11')

        self.data_frame_saver.save_df(df=combined_df, path=data_path)

    def load_data(self, data_path):

        return self.data_loader.load_data(data_path=data_path)

    def save_intermediate_data(self, df, filename):

        temp_dir = self.config.get_data_processing_temp_directory()
        path = os.path.join(temp_dir, filename)

        self.data_frame_saver.save_df(df=df, path=path)

        print(f"Intermediate data saved at {path}")

    def load_intermediate_data(self, filename):

        temp_dir = self.config.get_data_processing_temp_directory()
        path = os.path.join(temp_dir, filename)

        return self.data_loader.load_data(data_path=path)

    def fetch_and_save_data(self):
        # Data Fetching and Feature Engineering

        price_data = self.fetch_price_data()
        price_data_processed = self.price_feature_engineering(data=price_data)
        self.save_intermediate_data(price_data_processed, 'price_data_processed.csv')

        calendar_data = self.calendar_data_fetcher_service.fetch_calendar_data()
        calendar_data_processed = self.calendar_feature_engineering(data=calendar_data)
        self.save_intermediate_data(calendar_data_processed, 'calendar_data_processed.csv')

        news_data = self.news_fetcher_service.fetch_news_data()
        self.save_intermediate_data(news_data, 'news_data.csv')

        #econ_indicator_data = self.fetch_all_indicators()
        #self.save_intermediate_data(econ_indicator_data, 'econ_indicator_data.csv')

    def merge_data_and_save(self):
        # Sequential Merging
        """
        price_calendar_merged = self.merge_data_asof(
            self.load_intermediate_data('price_data_processed.csv'),
            self.load_intermediate_data('calendar_data_processed.csv'),
            'date'
        )
        self.save_intermediate_data(price_calendar_merged, 'price_calendar_merged.csv')

        news_merged = self.merge_news(
            data=self.load_intermediate_data('price_calendar_merged.csv'),
            news_data=self.load_intermediate_data('news_data.csv')
        )
        self.save_intermediate_data(news_merged, 'news_merged.csv')
        """
        # A special case solution to chunk the merger of very large training files.
        self.indicators_merger_service.merge_data()

    async def fetch_news_data_async(self):

        # Asynchronously fetch news data
        news_data = await self.fetch_news_data()
        return news_data

    async def _live_news(self):
        # Semantic versioning
        major_version = 1
        minor_version = 0
        patch_version = 0

        while True:
            # Construct the path for storing news data
            live_news_data_path = self.config.get_news_data_path(
                version=f'{major_version}.{minor_version}.{patch_version}',
                is_live_data=True)

            # Fetch news data asynchronously
            news_data = await self.fetch_news_data_async()
            if news_data:
                # Save the fetched news data
                data_frame_saver = DataFrameSaver()
                data_frame_saver.save_df(df=news_data, path=live_news_data_path)
                print(f"News data saved to {live_news_data_path}")

            # Wait for 3 minutes before next fetch
            await asyncio.sleep(180)  # 180 seconds = 3 minutes

    def live_chain(self):

        # Continuously fetch and update live news data
        asyncio.run(self._live_news())

        # Continuously fetch live data + steps
        # Merge with existing indicator data, refresh every 3 days
        # Make a prediction
        # The rest of the logic in pipeline_live_prediction.py
