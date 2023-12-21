import os
from ForexMastermind.ML.config import ForexMastermindConfig
from ForexMastermind.ML.infrastructure.pipeline_builder import DataPipelineBuilder

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

API_KEY = os.getenv('FINANCIAL_MODELING_PREP_API_KEY')


def set_live_model_version(live_model_version='3.0.86'):
    return live_model_version


def run_orchestrate_training_data_pipeline():
    """
    Initiates and orchestrates the live data fetching and processing pipeline for Forex trading data,
    including the gathering of training data.

    This function acts as the central controller for live operations and training data collection in Forex
    trading scenarios. It establishes and coordinates a series of services and pipelines for fetching,
    analyzing, and integrating Forex data from multiple sources. These include Forex price data, economic
    calendar details, news feeds, and economic indicators.

    The primary objective of this function is to aggregate and preprocess data for live trading analysis
    and to compile comprehensive training datasets for machine learning models. It leverages various
    services such as ForexDataAPIFetcherService, EconomicCalendarFetcherService, NewsFetcherService,
    and others to efficiently gather and process data. The function is designed to fetch data from a
    specified start date up to the current day, ensuring that both live operational data and historical
    data for training are up-to-date and accurately processed.

    The final output is meticulously prepared and saved, serving both as input for live trading decision
    systems and as a valuable dataset for training and refining predictive models.
    """
    config = ForexMastermindConfig()
    pipeline_builder = DataPipelineBuilder(API_KEY, config)
    fetch_data_pipeline = pipeline_builder.build_fetch_data_pipeline()
    #TODO: THE REST OF THE TRAINING DATA LOGIC, COME ON, YOU GOT THIS








