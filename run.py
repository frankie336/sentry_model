import os
import argparse
import time
import logging
from ForexMastermind.ML.config import ForexMastermindConfig
# Pipes
# NewsMergerPipeline,
# EconomicIndicatorDataMergerPipeline

from ForexMastermind.ML.pipelines.pipeline_indicator_data_merger import EconomicIndicatorDataMergerPipeline
from ForexMastermind.ML.pipelines.pipeline_news_data_merger import NewsMergerPipeline
from ForexMastermind.ML.pipelines.forex_data_fetcher_pipeline import (FetchNewsPipeline, FetchDataPipeline,
                                                                      FetchEconomicIndicatorsPipeline)

from ForexMastermind.ML.pipelines.standard_nn_forex_model_training import TrainingPipeline
from ForexMastermind.ML.pipelines.inverted_nn_forex_model_training import InvertedTrainingPipeline
from ForexMastermind.ML.pipelines.pipeline_unseen_data_evaluation import ModelEvaluationPipeline
from ForexMastermind.ML.tools.tools import GetTrainedFeatures, FeatureFetcherSingleton, GetADay

# Example of setting up logging
logging.basicConfig(level=logging.INFO)
# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()
API_KEY = os.getenv('FINANCIAL_MODELING_PREP_API_KEY')

START_DATE = '2018-01-01'
END_DATE = '2023-11-21'

UNSEEN_START_DATE = '2016-01-01'
UNSEEN_END_DATE = '2017-11-21'


def parse_args():

    parser = argparse.ArgumentParser(description="Run Forex Mastermind pipeline steps.")
    parser.add_argument('--fetch-data', action='store_true', help="Run Forex data fetcher.")
    parser.add_argument('--fetch-news', action='store_true', help="Run fetch Forex news data.")
    parser.add_argument('--fetch-indicators', action='store_true', help="Run fetch Economic indicators.")

    parser.add_argument('--merge-news', action='store_true', help="Run news merger steps.")
    parser.add_argument('--merge-indicators', action='store_true', help="Run indicators merger steps.")

    parser.add_argument('--train-model', action='store_true', help="Run model training.")
    parser.add_argument('--train-inverted-model', action='store_true', help="Run model training.")

    parser.add_argument('--optimize-model', action='store_true', help="Run model optimization.")
    parser.add_argument('--process-live', action='store_true', help="Run live.")

    parser.add_argument('--eval-model', action='store_true', help="Run model evaluation.")

    # passed args
    parser.add_argument('--unseen-data', action='store_true', help="Use unseen data.")
    parser.add_argument('--live-data', action='store_true', help="Use live data.")
    parser.add_argument('--model-notes', type=str, help="Model notes.")
    parser.add_argument('--sample-size', type=int, help="Sample size.")
    parser.add_argument('--data-version', type=str, help="Data version.")
    parser.add_argument('--model-version', type=str, help="Specify the model version to use.")

    return parser.parse_args()


def run_forex_data_fetcher(unseen_data=False, live_data=False):

    if live_data:
        start_date, end_date = GetADay.get_days()
    elif unseen_data:
        start_date, end_date = UNSEEN_START_DATE, UNSEEN_END_DATE
        print(start_date, end_date)
    else:
        start_date, end_date = START_DATE, END_DATE

    list_of_forex_pairs = ["EURUSD", "USDJPY", "GBPUSD", "USDCHF", "AUDUSD", "USDCAD", "NZDUSD"]
    pipeline = FetchDataPipeline(pairs=list_of_forex_pairs, start_date=start_date, end_date=end_date,
                                 api_key=API_KEY, unseen_data=unseen_data)

    pipeline.run()


def run_fetch_forex_news_data(unseen_data=False, live_data=False):

    try:
        if live_data:
            start_date, end_date = GetADay.get_days()
        elif unseen_data:
            start_date, end_date = UNSEEN_START_DATE, UNSEEN_START_DATE
        else:
            start_date, end_date = START_DATE, END_DATE

        config = ForexMastermindConfig(start_date=start_date, end_date=end_date)

        news_fetch_pipeline = FetchNewsPipeline(start_date=start_date, end_date=end_date, config=config,
                                                api_key=API_KEY, unseen_data=unseen_data, live_data=live_data)

        if live_data:
            while True:
                news_fetch_pipeline.run()
                time.sleep(300)  # Run every 5 minutes

        else:
            news_fetch_pipeline.run()

    except Exception as e:
        logging.error(f"An error occurred: {e}")


def run_fetch_economic_indicators(unseen_data=False, live_data=False):

    if live_data:
        get_a_day = GetADay()
        _, end_date = get_a_day.get_days()
        start_date = '2018-01-01'
    elif unseen_data:
        start_date, end_date = UNSEEN_START_DATE, UNSEEN_START_DATE
    else:
        start_date, end_date = START_DATE, END_DATE

    config = ForexMastermindConfig(start_date=start_date, end_date=end_date)
    pipeline = FetchEconomicIndicatorsPipeline(start_date=start_date, end_date=end_date, config=config,
                                               api_key=API_KEY, unseen_data=unseen_data, live_data=live_data)

    pipeline.run()


def run_indicator_merger_steps(unseen_data=False, live_data=False):

    # Example usage:
    # python -m  ForexMastermind.ML.run --merge-indicators --live-data

    if live_data:
        get_a_day = GetADay()
        _, end_date = get_a_day.get_days()
        start_date = '2018-01-01'
    elif unseen_data:
        start_date, end_date = UNSEEN_START_DATE, UNSEEN_START_DATE
    else:
        start_date, end_date = START_DATE, END_DATE

    pipeline = EconomicIndicatorDataMergerPipeline(start_date=start_date, end_date=end_date,
                                                   unseen_data=unseen_data, live_data=live_data)

    pipeline.run()


def run_news_merger_steps(unseen_data=False, live_data=False):

    if live_data:
        start_date, end_date = GetADay.get_days()
    elif unseen_data:
        start_date, end_date = UNSEEN_START_DATE, UNSEEN_START_DATE
    else:
        start_date, end_date = START_DATE, END_DATE

    pipeline = NewsMergerPipeline(start_date=start_date, end_date=end_date)
    pipeline.merge_news()


def run_standard_training(sample_size=4, model_notes='None.', data_version='12'):

    # Example usage:
    # python -m  ForexMastermind.ML.run --train-model --sample-size 4
    # python -m  ForexMastermind.ML.run --train-model --sample-size 4 --model-notes same as v3.0.36
    # python -m  ForexMastermind.ML.run --train-model --sample-size 4 --model-notes "Same results as old?" --data-version "6"

    if sample_size is None:
        sample_size = 0.1
    if data_version is None:
        data_version = '12'
    if model_notes is None:
        model_notes = 'None.'

    config = ForexMastermindConfig(start_date=START_DATE, end_date=END_DATE)
    next_min_data_path = config.get_next_data_path(data_type='ForexData', training_data_version=data_version)

    pipeline = TrainingPipeline(file_path=next_min_data_path)

    pipeline.run(model_notes=model_notes, sample_size=sample_size)


def run_evaluation_on_unseen_data(unseen_data=False, model_version='3.0.38'):

    # Example usage:
    # python -m  ForexMastermind.ML.run --eval-model --unseen-data --model-version 3.0.36
    # python -m  ForexMastermind.ML.run --eval-model --unseen-data --model-version 3.0.37
    # python -m  ForexMastermind.ML.run --eval-model --unseen-data --model-version 3.0.38
    # python -m  ForexMastermind.ML.run --eval-model --unseen-data --model-version 3.0.73
    if model_version is None:
        model_version ='3.0.36'

    if unseen_data:
        pass
        start_date = UNSEEN_START_DATE
        end_date = UNSEEN_END_DATE
    else:
        start_date = START_DATE
        end_date = END_DATE

    config = ForexMastermindConfig(start_date=start_date, end_date=end_date,
                                   unseen_start_date=start_date, unseen_end_date=end_date)

    feature_fetcher = FeatureFetcherSingleton()
    features = feature_fetcher.get_features(model_version=model_version)

    unseen_data_path = config.get_next_data_path(data_type='ForexData', training_data_version='6', is_unseen=True)
    print(f"Testing on data: {unseen_data_path}")

    model_path = config.get_standard_model_path(model_version=model_version)

    scaler_path = config.get_scaler_path(model_version=model_version)

    pipeline = ModelEvaluationPipeline(file_path=unseen_data_path, model_path=model_path,
                                       scaler_path=scaler_path, features=features)
    pipeline.run()


def run_inverted_training():

    config = ForexMastermindConfig()
    inverted_data_path = config.get_inverted_data_path(data_type='ForexData', version='6')

    pipeline = InvertedTrainingPipeline(file_path=inverted_data_path)
    pipeline.run(sample_size=100)


def run_optimization():

    config = ForexMastermindConfig()
    next_min_data_path = config.get_next_data_path(data_type='ForexData', training_data_version='5')
    #pipeline = OptimizationPipeline(file_path=next_min_data_path)
    #pipeline.process(sample_size=100)


if __name__ == '__main__':
    args = parse_args()

    if args.fetch_data:
        run_forex_data_fetcher(unseen_data=args.unseen_data)

    if args.fetch_news:
        run_fetch_forex_news_data(unseen_data=args.unseen_data, live_data=args.live_data)

    if args.fetch_indicators:
        run_fetch_economic_indicators(unseen_data=args.unseen_data, live_data=args.live_data)

    if args.merge_news:
        run_news_merger_steps(unseen_data=args.unseen_data)

    if args.merge_indicators:
        run_indicator_merger_steps(unseen_data=args.unseen_data, live_data=args.live_data)

    if args.train_model:
        run_standard_training(sample_size=args.sample_size, model_notes=args.model_notes,
                              data_version=args.data_version)

    if args.train_inverted_model:
        run_inverted_training()

    if args.eval_model:
        run_evaluation_on_unseen_data(unseen_data=args.unseen_data, model_version=args.model_version)

    if args.optimize_model:
        run_optimization()








