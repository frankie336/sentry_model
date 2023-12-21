import pandas as pd
from ForexMastermind.ML.config import ForexMastermindConfig
from ForexMastermind.ML.tools.tools import DataFrameSaver, DataLoader


class NewsMerger:

    def __init__(self, start_date, end_date, trading_df, news_df, unseen_data=False, live_data=False):
        self.live_data = live_data
        self.start_date = start_date
        self.end_date = end_date
        self.trading_df = trading_df
        self.news_df = news_df
        self.unseen_data = unseen_data

        self.config = ForexMastermindConfig(start_date=self.start_date, end_date=self.end_date,
                                            unseen_start_date=self.start_date, unseen_end_date=self.end_date)

    def merge_trading_with_news(self):

        # Convert the 'date' column in both DataFrames to datetime without timezone
        self.trading_df['date'] = pd.to_datetime(self.trading_df['date']).dt.tz_localize(None)
        self.news_df['date'] = pd.to_datetime(self.news_df['date']).dt.tz_localize(None)

        # Sort news_df by date
        news_df = self.news_df.sort_values('date')

        # Initialize an empty DataFrame to hold the merged results
        merged_results = pd.DataFrame()

        # Assuming 'currency_pair' is the column that identifies each currency pair in trading_df
        for currency_pair, group in self.trading_df.groupby('currency'):
            # Sort each group by date
            group_sorted = group.sort_values('date')

            # Merge the group with news_df using pd.merge_asof with a tolerance of 1 day
            merged_group = pd.merge_asof(group_sorted, news_df, on='date', tolerance=pd.Timedelta('1d'),
                                         direction='nearest')

            # Append the result to the merged_results DataFrame
            merged_results = pd.concat([merged_results, merged_group], ignore_index=True)

        # Fill in default values for rows without corresponding news data
        default_values = {'sentiment_score': -1, 'news_headline': 'None'}  # Replace with relevant column names
        merged_results.fillna(value=default_values, inplace=True)

        data_saver = DataFrameSaver()
        next_min_unseen_data_path = self.config.get_next_data_path(data_type='ForexData', training_data_version='5', is_unseen=True)
        next_min_data_path = self.config.get_next_data_path(data_type='ForexData', training_data_version='5')

        if self.live_data:

            return merged_results

        if self.unseen_data:
            data_saver.save_df(df=merged_results, path=next_min_unseen_data_path)
        else:
            data_saver.save_df(df=merged_results, path=next_min_data_path)


class NewsMergerPipeline:
    def __init__(self, start_date, end_date, unseen_data=False, live_data=False):

        self.live_data = live_data
        self.start_date = start_date
        self.end_date = end_date
        self.unseen_data = unseen_data
        self.config = ForexMastermindConfig(start_date=self.start_date, end_date=self.end_date,
                                            unseen_start_date=self.start_date, unseen_end_date=self.end_date)

    def run(self, live_trading_df=None):
        # Check if live data is provided when expected
        if self.live_data and live_trading_df is None:
            raise ValueError("Live trading data is required but not provided.")

        # Set the path for news data
        if self.unseen_data:
            the_data_path = self.config.get_data_path(data_type='ForexData', version='3.0.1', is_unseen=self.unseen_data)
            news_data_path = self.config.get_news_data_path(version='3', is_unseen=self.unseen_data)
        else:
            the_data_path = self.config.get_next_data_path(data_type='ForexData', training_data_version='5')
            news_data_path = self.config.get_news_data_path('3')

        # Assign the appropriate DataFrame based on the mode (live or historical)
        trading_df = live_trading_df if self.live_data else DataLoader(data_path=the_data_path).load_data()

        # For live data, adjust the news data path
        if self.live_data:
            print(f'Merging News from the following path:{news_data_path}')
            config = ForexMastermindConfig(start_date=self.start_date, end_date=self.end_date)
            news_data_path = config.get_news_data_path('3', is_live_data=True)

        # Initialize NewsMerger with the appropriate data
        data_loader = DataLoader()
        news_df = data_loader.load_data(data_path=news_data_path)
        news_merger = NewsMerger(start_date=self.start_date, end_date=self.end_date,
                                 trading_df=trading_df,
                                 news_df=news_df,
                                 unseen_data=self.unseen_data, live_data=self.live_data)

        # Merge trading data with news
        return news_merger.merge_trading_with_news()