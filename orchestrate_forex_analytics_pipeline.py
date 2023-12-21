import os
from ForexMastermind.ML.config import ForexMastermindConfig
from ForexMastermind.ML.infrastructure.pipeline_builder import DataPipelineBuilder

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

API_KEY = os.getenv('FINANCIAL_MODELING_PREP_API_KEY')


# 3.0.73 <-- previous
def set_live_model_version(live_model_version='3.0.86'):
    return live_model_version


def run_orchestrate_forex_analytics_pipeline():
    """
    Executes the primary Forex data analytics pipeline, primarily for live testing of the AI model.

    This function is the central orchestrator for the Forex trading data analytics pipeline, with a focus on
    live testing the AI model. It begins by fetching Forex trading data for specific currency pairs over a
    defined time range. Following this, it merges the data with news and economic indicators to create a
    comprehensive dataset for analysis.

    The process involves several specialized pipelines and services. The FetchDataPipeline handles the initial
    data retrieval, the NewsMergerPipeline integrates news data, and the EconomicIndicatorDataMergerPipeline
    appends economic indicators.

    The heart of the function is the PredictiveAnalyticsPipeline, which utilizes a prediction service to
    analyze the enriched dataset. This stage is crucial for live testing the AI model, providing insights
    and validating the model's performance in real-time market conditions. The results of this analysis are
    subsequently saved for review and further decision-making.
    """
    config = ForexMastermindConfig()
    pipeline_builder = DataPipelineBuilder(API_KEY, config)
    fetch_data_pipeline = pipeline_builder.build_fetch_data_pipeline()
    fetch_data_pipeline.live_chain()
    # TODO: FIX THE INFINITE LOOP ISSUE ON NEWS DATA FETCHING , COME ON, YOU GOT THIS!


if __name__ == '__main__':
    run_orchestrate_forex_analytics_pipeline()
