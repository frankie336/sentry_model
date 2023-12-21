# forex_data_fetcher_pipeline.py

from tqdm import tqdm
from textblob import TextBlob
import random
import time
import requests
import pandas as pd
from datetime import datetime, timedelta

from ForexMastermind.ML.config import ForexMastermindConfig
from ForexMastermind.ML.tools.tools import DataFrameSaver, GetADay
# Pipes
from ForexMastermind.ML.pipelines.pipeline_economic_calendar import EconomicCalendarPreprocessingPipeline

from ForexMastermind.ML.pipelines.pipeline_feature_engineering import FeatureEngineeringPipeline
from ForexMastermind.ML.pipelines.pipeline_news_data_merger import NewsMergerPipeline

# Services
from ForexMastermind.ML.services.service_feature_engineering import CategoricalFeatureService


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


class ForexDataFetcher:
    """
      Fetches historical forex data for a specified currency pair and date range.

      Attributes:
          pair (str): The currency pair to fetch data for.
          start_date (str): The start date for the data range in 'YYYY-MM-DD' format.
          end_date (str): The end date for the data range in 'YYYY-MM-DD' format.
          api_key (str): The API key for accessing the data source.
      """
    def __init__(self, data_fetcher_service, pair, api_key, start_date=None, end_date=None):

        self.data_fetcher_service = data_fetcher_service
        self.pair = pair
        self.api_key = api_key

    def fetch_data(self, pair, start, end):

        # Sets up the params for the base fetcher service
        # uses instance of the main fetcher to fetch data
        params = {'pair': pair, 'start_date': start, 'end_date': end, 'apikey': self.api_key}
        endpoint = f"https://financialmodelingprep.com/api/v3/historical-chart/1min/{params['pair']}?from={params['start_date']}&to={params['end_date']}&apikey={params['apikey']}"
        data = self.data_fetcher_service.fetch_data(params=params, endpoint=endpoint)

        df = pd.DataFrame(data)
        df['Pair'] = pair

        return df


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


class FetchedDataProcessing:

    def __init__(self, forex_df=None, econ_df=None, live_data=False):

        self.forex_df = forex_df
        self.econ_df = econ_df
        self.live_data = live_data

    def preprocess_forex_data(self):

        preprocessing = FeatureEngineeringPipeline(data=self.forex_df, live_data=self.live_data)

        preprocessed_forex = preprocessing.apply_feature_engineering()

        return preprocessed_forex

    def preprocess_economic_calendar_data(self):

        economic_calendar_preprocessing_pipeline = EconomicCalendarPreprocessingPipeline(data=self.econ_df)
        preprocessed_econ = economic_calendar_preprocessing_pipeline.process()

        return preprocessed_econ


    @staticmethod
    def merge_data(forex_df, econ_df):

        forex_df_sorted = forex_df.sort_values('date')
        econ_df_sorted = econ_df.sort_values('date')
        return pd.merge_asof(forex_df_sorted, econ_df_sorted, on='date', direction='forward')





    def preprocess(self, forex_df, econ_df):

        preprocessing = FeatureEngineeringPipeline(live_data=self.live_data)
        preprocessed_forex = preprocessing.apply_feature_engineering(data=forex_df)

        economic_calendar_preprocessing_pipeline = EconomicCalendarPreprocessingPipeline(data=econ_df)
        preprocessed_econ = economic_calendar_preprocessing_pipeline.process()

        data = FetchedDataProcessing.merge_data(preprocessed_forex, preprocessed_econ)

        # TODO
        #data_preprocessor = DataPreprocessor(data=data)

        feature_engineer = CategoricalFeatureService(data=data)
        data_feature_engineered = feature_engineer.encode_currency_pairs()
        #data_with_encoded_curr = data_preprocessor.encode_currency_pairs()

        return data_feature_engineered


class EconomicIndicatorsFetcher:
    def __init__(self, start_date, end_date, api_key):

        self.api_key = api_key
        self.start_date = start_date
        self.end_date = end_date

    def fetch_indicator(self, name):

        url = f"https://financialmodelingprep.com/api/v4/economic?name={name}&from={self.start_date}&to={self.end_date}&apikey={self.api_key}"
        response = requests.get(url)
        if response.status_code == 200:
            return pd.DataFrame(response.json())
        else:
            print(f"Failed to fetch data for {name}: {response.status_code}")
            return pd.DataFrame()

    def fetch_all_indicators(self, indicators):

        all_data = pd.DataFrame()
        for indicator in indicators:
            data = self.fetch_indicator(indicator)
            all_data = pd.concat([all_data, data.assign(Indicator=indicator)], ignore_index=True)
        return all_data.pivot(index='date', columns='Indicator', values='value').reset_index()


class FetchAndProcessForexData:

    def __init__(self, pair, start_date, end_date, api_key, live_data=False):

        self.pair = pair
        self.start_date = start_date
        self.end_date = end_date
        self.api_key = api_key
        self.live_data = live_data

    def fetch_and_preprocess_pair_data(self, pair, econ_calendar_data):

        all_data = []

        date_range_generator = DateRangeGenerator()

        for start, end in tqdm(date_range_generator.generate_date_ranges(self.start_date, self.end_date, timedelta(days=30)),
                               desc=f'Fetching {pair}'):

            data_fetcher_service = ForexDataAPIFetcherService()
            forex_data_fetcher = ForexDataFetcher(pair=self.pair, start_date=self.start_date, end_date=self.end_date,
                                                  api_key=self.api_key, data_fetcher_service=data_fetcher_service)

            forex_data = forex_data_fetcher.fetch_data(pair, start.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d'))

            #print(forex_data)
            #time.sleep(100000)

            if not forex_data.empty:

                fetched_data_processing = FetchedDataProcessing(live_data=self.live_data)

                preprocessed_df = fetched_data_processing.preprocess(forex_data, econ_calendar_data)
                all_data.append(preprocessed_df)

        return pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()


class EconomicCalendarFetcher:
    """
       Fetches economic calendar data within a specified date range.

       Attributes:
           start_date (str): The start date for the data range in 'YYYY-MM-DD' format.
           end_date (str): The end date for the data range in 'YYYY-MM-DD' format.
           api_key (str): The API key for accessing the data source.
    """
    def __init__(self, start_date, end_date, api_key):
        self.start_date = start_date
        self.end_date = end_date
        self.api_key = api_key

    def fetch_data(self):
        """
        Initializes the EconomicCalendarFetcher with specified parameters.
        """
        url = f"https://financialmodelingprep.com/api/v3/economic_calendar?from={self.start_date}&to={self.end_date}&apikey={self.api_key}"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            return pd.DataFrame(data)
        else:
            print("Failed to fetch economic calendar data")
            return pd.DataFrame()


class SentimentAnalyzer:
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


class ForexNewsFetcher:
    """
    Fetches forex news data relevant to forex markets within a specified date range.

    Attributes:
        start_date (str): The start date for the news data in 'YYYY-MM-DD' format.
        end_date (str): The end date for the news data in 'YYYY-MM-DD' format.
        api_key (str): The API key for accessing the news data source.
    """
    def __init__(self, start_date, end_date, api_key):

        self.start_date = start_date
        self.end_date = end_date
        self.api_key = api_key

    def fetch_forex_news_data(self):
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
                        sentiment_score = SentimentAnalyzer.analyze_sentiment(news_item['text'])
                        all_news.append({
                            'date': news_item['publishedDate'],
                            'sentiment_score': sentiment_score,
                            'news_headline': news_item.get('title', '')  # Ensure this matches the API response
                        })
                page += 1
            else:
                break

        #TODO: BROKEN
        forex_news = pd.DataFrame(all_news)
        processed_news_data = NewsMergerPipeline(start_date=self.start_date, end_date=self.end_date)
        processed_news_data.merge_news(data=processed_news_data, start_date=self.start_date, end_date=self.end_date)

        return processed_news_data


class FetchNewsPipeline:
    def __init__(self, start_date, end_date, config, api_key, unseen_data=False, live_data=False):

        self.unseen_data = unseen_data
        self.start_date = start_date
        self.end_date = end_date
        self.api_key = api_key
        self.live_data = live_data
        self.config = config

        self.news_data_path = self.config.get_news_data_path(version='3', is_unseen=self.unseen_data,
                                                             is_live_data=self.live_data)

    def run(self):

        forex_news_fetcher = ForexNewsFetcher(start_date=self.start_date, end_date=self.end_date, api_key=self.api_key)

        data_frame_saver = DataFrameSaver()
        economic_data_df = forex_news_fetcher.fetch_forex_news_data()
        data_frame_saver.save_df(df=economic_data_df, path=self.news_data_path)


class FetchEconomicIndicatorsPipeline:
    def __init__(self, start_date, end_date, config, api_key, unseen_data=False, live_data=False):

        self.unseen_data = unseen_data
        self.live_data = live_data
        self.start_date = start_date
        self.end_date = end_date
        self.api_key = api_key
        self.config = config

        self.economic_data_path = self.config.get_economic_indicators_data_path(version='1', is_unseen=self.unseen_data,
                                                                                live_data=self.live_data)

        if self.live_data:
            get_a_day = GetADay()
            self.start_date = get_a_day.get_days()
            self.start_date = start_date

    def run(self):

        indicators = ['GDP', 'realGDP', 'nominalPotentialGDP', 'realGDPPerCapita',
                      'federalFunds', 'CPI', 'inflationRate',
                      'unemploymentRate', 'totalNonfarmPayroll',
                      'consumerSentiment', 'retailSales',
                      'durableGoods',
                      '3MonthOr90DayRatesAndYieldsCertificatesOfDeposit']

        fetcher = EconomicIndicatorsFetcher(start_date=self.start_date, end_date=self.end_date, api_key=self.api_key)

        # The file save path will be dynamic according to the Bool condition of unseen_data and is_live_data
        economic_data_df = fetcher.fetch_all_indicators(indicators=indicators)
        data_frame_saver = DataFrameSaver()
        data_frame_saver.save_df(df=economic_data_df, path=self.economic_data_path)


class FetchDataPipeline:
    def __init__(self, pairs, start_date, end_date, api_key, unseen_data=False, live_data=False):

        self.pairs = pairs
        self.start_date = start_date
        self.end_date = end_date
        self.api_key = api_key
        self.combined_df = pd.DataFrame()
        self.unseen_data = unseen_data
        self.live_data = live_data
        print(self.live_data)

    def run(self):

        economic_calendar_fetcher = EconomicCalendarFetcher(start_date=self.start_date,
                                                            end_date=self.end_date, api_key=self.api_key)

        econ_calendar_data = economic_calendar_fetcher.fetch_data()

        # Process each pair
        for pair in self.pairs:
            self._process_pair(pair, econ_calendar_data)

        # Save or return the combined data
        return self._save_or_return_data()

    def _process_pair(self, pair, econ_calendar_data):

        pair_processor = FetchAndProcessForexData(pair=pair, start_date=self.start_date,
                                                                end_date=self.end_date,
                                                                api_key=self.api_key,
                                                                live_data=self.live_data
                                                  )

        pair_data = pair_processor.fetch_and_preprocess_pair_data(pair, econ_calendar_data)

        self.combined_df = pd.concat([self.combined_df, pair_data], ignore_index=True)

    def _save_or_return_data(self):

        data_frame_saver = DataFrameSaver()
        config = ForexMastermindConfig(start_date=self.start_date, end_date=self.end_date)

        if self.live_data:
            # For live data, simply return the combined DataFrame
            print('returned self.combined_df')
            return self.combined_df
        else:
            # For non-live data, save the combined DataFrame
            if self.unseen_data:
                # Save as unseen data
                data_path = config.get_data_path(data_type='ForexData', version='5', is_unseen=self.unseen_data)
            else:
                # Save as regular data
                data_path = config.get_data_path(data_type='ForexData', version='5')

            data_frame_saver.save_df(df=self.combined_df, path=data_path)

































