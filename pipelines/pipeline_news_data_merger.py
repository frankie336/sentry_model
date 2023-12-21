# pipeline_news_data_merger.py
from ForexMastermind.ML.services.service_news_data_merger import TradingNewsDataMergerService
from ForexMastermind.ML.config import ForexMastermindConfig


class NewsMergerPipeline:
    def __init__(self, data_feed=None, start_date=None, end_date=None, live_data=False):
        self.start_date = start_date
        self.end_date = end_date
        self.live_data = live_data
        self.config = ForexMastermindConfig()
        self.data_feed = data_feed

        if self.live_data:
            self.news_data = self.data_feed

    def merge_news(self, data, news_data):

        # Initialize NewsMerger with the appropriate data
        news_merger = TradingNewsDataMergerService(start_date=self.start_date, end_date=self.end_date,
                                                   data=data, news_data=news_data, config=self.config)

        return news_merger.merge_news()

