from tqdm import tqdm
from textblob import TextBlob
import random
import time
import requests
import pandas as pd
from datetime import datetime, timedelta

from ForexMastermind.ML.pipelines.pipeline_feature_engineering import PreprocessForexData
from ForexMastermind.ML.OLD_config import PATHS
# pipes
from ForexMastermind.ML.pipelines.pipeline_feature_engineering import (EconomicCalendarPreprocessingPipeline,
                                                                       SharedPreprocessingStepsPipeline)


class ForexDataFetcher:

    def __init__(self, pairs, start_date, end_date, api_key, econ_calendar_start_date=None, econ_calendar_end_date=None):
        self.pairs = pairs
        self.start_date = start_date
        self.end_date = end_date
        self.api_key = api_key
        self.econ_calendar_start_date = econ_calendar_start_date
        self.econ_calendar_end_date = econ_calendar_end_date
        self.combined_df = pd.DataFrame()

    def fetch_data_for_pair(self, pair, start, end):

        url = f"https://financialmodelingprep.com/api/v3/historical-chart/1min/{pair}?from={start}&to={end}&apikey={self.api_key}"
        response = requests.get(url)
        if response.status_code == 200:
            pair_data = response.json()
            df = pd.DataFrame(pair_data)
            df['Pair'] = pair
            return df
        else:
            print(f"Failed to fetch data for {pair}: {response.status_code}")
            return pd.DataFrame()

    def fetch_and_preprocess_pair_data(self, pair, econ_calendar_data):

        all_data = []
        for start, end in tqdm(self.generate_date_ranges(self.start_date, self.end_date, timedelta(days=30)),
                               desc=f'Fetching {pair}'):

            forex_data = self.fetch_data_for_pair(pair, start.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d'))

            if not forex_data.empty:
                preprocessed_df = self.preprocess(forex_data, econ_calendar_data)
                all_data.append(preprocessed_df)

        return pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()

    def generate_date_ranges(self, start, end, delta):

        current = datetime.strptime(start, '%Y-%m-%d')
        end = datetime.strptime(end, '%Y-%m-%d')
        while current < end:
            yield current, min(current + delta, end)
            current += delta

    def preprocess(self, forex_df, econ_df):

        preprocessor = PreprocessForexData(data=forex_df)
        preprocessed_forex = preprocessor.run_shared_steps()
        preprocessed_econ = PreprocessForexData(data=econ_df).run_economic_steps()

        # TODO: cut over to the new preprocessing pipe by uncommenting the next four lines
        # shared_preprocessing_pipeline = SharedPreprocessingStepsPipeline(data=forex_df)
        # preprocessed_forex = shared_preprocessing_pipeline.process()

        # economic_calendar_preprocessing_pipeline = EconomicCalendarPreprocessingPipeline(data=econ_df)
        # preprocessed_econ = economic_calendar_preprocessing_pipeline.process()

        return self.merge_data(preprocessed_forex, preprocessed_econ)

    def merge_data(self, forex_df, econ_df):

        forex_df_sorted = forex_df.sort_values('date')
        econ_df_sorted = econ_df.sort_values('date')
        return pd.merge_asof(forex_df_sorted, econ_df_sorted, on='date', direction='forward')

    def fetch_economic_calendar_data(self):
        url = f"https://financialmodelingprep.com/api/v3/economic_calendar?from={self.econ_calendar_start_date}&to={self.econ_calendar_end_date}&apikey={self.api_key}"
        response = requests.get(url)
        if response.status_code == 200:
            econ_data = response.json()
            return pd.DataFrame(econ_data)
        else:
            print("Failed to fetch economic calendar data")
            return pd.DataFrame()

    def sentiment_analysis(self, text):
        blob = TextBlob(text)
        return blob.sentiment.polarity

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
                        sentiment_score = self.sentiment_analysis(news_item['text'])
                        all_news.append({
                            'date': news_item['publishedDate'],
                            'sentiment_score': sentiment_score,
                            'news_headline': news_item.get('title', '')  # Ensure this matches the API response
                        })
                page += 1
            else:
                break

        preprocess_forex_data = PreprocessForexData(data=pd.DataFrame(all_news))

        return preprocess_forex_data.aggregate_sentiment_scores()

    def run_fetch_forex_news_data(self):

        news = self.fetch_forex_news_data()
        self.save_df(df=news, path=PATHS['MIN']['NEWS_DATA'])


    def run(self):

        econ_calendar_data = self.fetch_economic_calendar_data()
        try:
            for pair in tqdm(self.pairs, desc='Processing Pairs'):
                pair_data = self.fetch_and_preprocess_pair_data(pair, econ_calendar_data)
                self.combined_df = pd.concat([self.combined_df, pair_data], ignore_index=True)

        except Exception as e:
            print(f"Error encountered: {e}. Saving data processed so far.")
            self.save_df(df=self.combined_df, path=PATHS['MIN']['TRAINING_DATA'])
            raise  # Optionally re-raise the exception if you want the error to propagate

        # Save the combined data after successful processing of all pairs
        self.save_df(df=self.combined_df, path=PATHS['MIN']['TRAINING_DATA'])

    def save_df(self, df, path):
        df.to_csv(path, index=False)
        print(f"DataFrame saved successfully to {path}")


