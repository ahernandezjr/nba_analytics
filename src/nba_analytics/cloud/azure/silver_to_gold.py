# Bussiness-Level Aggregates
# Deliver continuously updated, clean data TODOwnstream users and apps

from collections import namedtuple
import json
import numpy as np
import pyspark.sql.functions as F
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, IntegerType

from ...utils import filename_grabber
from ...utils.config import settings
from ...utils.logger import get_logger

logger = get_logger(__name__)

FILTER_AMT = settings.dataset.FILTER_AMT

# Initialize Spark session
spark = SparkSession.builder.appName("NBA Analysis").getOrCreate()

# Define a named tuple for the return value of create_gold_datasets
GoldDataset = namedtuple('GoldDataset', [
    'df_filtered',
    'dict_filtered',
    'df_continuous',
    'dict_continuous',
    'df_first_continuous',
    'dict_first_continuous',
    'np_overlap'
])

def read_data_from_silver():
    """
    Reads the data from the silver dataset using PySpark.

    Returns:
        pyspark.sql.DataFrame: The input dataframe containing player statistics.
    """
    silver_data_path = "/mnt/nba-data/silver/nba_player_stats.csv"
    df = spark.read.csv(silver_data_path, header=True, inferSchema=True)
    return df

def create_gold_datasets(df):
    """
    Creates a dataset with player statistics for players who have played at least FILTER_AMT years in the NBA.

    Args:
        df (pyspark.sql.DataFrame): The input dataframe containing player statistics.

    Returns:
        GoldDataset: A named tuple containing:
            - df_continuous (pyspark.sql.DataFrame): Filtered dataframe with continuous stretches of at least FILTER_AMT years.
            - df_first_continuous (pyspark.sql.DataFrame): Filtered dataframe with the first continuous stretch of at least FILTER_AMT years.
            - np_overlap (numpy.array): Numpy array with player statistics for each continuous period of FILTER_AMT years.
            - dict_continuous (dict): Dictionary with player statistics for continuous stretches.
            - dict_first_continuous (dict): Dictionary with player statistics for first continuous stretches.
    """
    logger.debug(f"Filtering dataset for players who have played for more than {FILTER_AMT} years...")

    # Filter unnecessary columns
    df_filtered = filtering.filter_columns(df)
    dict_filtered = processing.df_to_dict(df_filtered)

    # Filter the dataset to include only players who have continuous stretches of at least FILTER_AMT years
    df_continuous = filtering.filter_atleast_continuous_years(df_filtered, FILTER_AMT)
    dict_continuous = processing.df_to_dict(df_continuous)

    # Filter the dataset to include only players who have a continuous stretch of at least FILTER_AMT years at the start of their career
    df_first_continuous = filtering.filter_first_continuous_years(df_filtered, FILTER_AMT)
    dict_first_continuous = processing.df_to_first_years_dict(df_first_continuous)

    # Create overlap dataset
    np_overlap = processing.create_overlap_data(df_continuous)

    return GoldDataset(
        df_filtered=df_filtered,
        dict_filtered=dict_filtered,
        df_continuous=df_continuous,
        dict_continuous=dict_continuous,
        df_first_continuous=df_first_continuous,
        dict_first_continuous=dict_first_continuous,
        np_overlap=np_overlap
    )

def save_df_and_dict(gold_dataset):
    """
    Saves the given DataFrame, numpy array, and dictionary to respective files.

    Args:
        gold_dataset (GoldDataset): The named tuple containing the data to save.

    Returns:
        None
    """
    # Define file paths
    filtered_path = "/mnt/nba-data/gold/filtered"
    dict_filtered_path = "/mnt/nba-data/gold/filtered.json"
    continuous_path = "/mnt/nba-data/gold/continuous"
    dict_continuous_path = "/mnt/nba-data/gold/continuous.json"
    first_continuous_path = "/mnt/nba-data/gold/first_continuous"
    dict_first_continuous_path = "/mnt/nba-data/gold/first_continuous.json"
    overlap_path = "/mnt/nba-data/gold/overlap.npy"

    logger.debug(f"Saving filtered dataset to '{filtered_path}'...")
    logger.debug(f"Saving first continuous dataset to '{first_continuous_path}'...")
    logger.debug(f"Saving continuous dataset to '{continuous_path}'...")
    logger.debug(f"Saving overlap dataset to '{overlap_path}'...")

    # Save dataframes
    gold_dataset.df_filtered.write.csv(filtered_path, header=True, mode='overwrite')
    gold_dataset.df_continuous.write.csv(continuous_path, header=True, mode='overwrite')
    gold_dataset.df_first_continuous.write.csv(first_continuous_path, header=True, mode='overwrite')

    # Save numpy array
    np.save(overlap_path, gold_dataset.np_overlap)

    # Save dictionaries as JSON
    with open(dict_filtered_path, 'w') as json_file:
        json.dump(gold_dataset.dict_filtered, json_file, indent=4)
    with open(dict_continuous_path, 'w') as json_file:
        json.dump(gold_dataset.dict_continuous, json_file, indent=4)
    with open(dict_first_continuous_path, 'w') as json_file:
        json.dump(gold_dataset.dict_first_continuous, json_file, indent=4)

    logger.debug(f"Filtered dataset saved to: '{filtered_path}'.")
    logger.debug(f"First-continuous dataset saved to: '{first_continuous_path}'.")
    logger.debug(f"Continuous dataset saved to: '{continuous_path}'.")
    logger.debug(f"Overlap dataset saved to: '{overlap_path}'.")
    logger.debug(f"Filtered dictionaries saved to: '{dict_filtered_path}', '{dict_continuous_path}', and '{dict_first_continuous_path}'.")

def log_summary(df_first_continuous, np_overlap):
    """
    Prints a summary of the given DataFrame and its filtered version.

    Args:
        df_first_continuous (pyspark.sql.DataFrame): The filtered DataFrame.
        np_overlap (numpy.array): The numpy array with player statistics for each continuous period of FILTER_AMT years.

    Returns:
        None
    """
    logger.debug("Printing summary...")

    # Show the head of the filtered DataFrame
    df_first_continuous.show()

    # Reshape the 2D numpy array to its original shape
    reshaped_overlap = np_overlap.reshape(np_overlap.shape[0], FILTER_AMT, -1)

    # Print the number of entries and the number of unique players in the filtered dataframe
    logger.info(f"First Continuous DataFrame: Entries={df_first_continuous.count()}, Unique Players={df_first_continuous.select('slug').distinct().count()}")

    # Print the number of entries and the number of unique players in the overlap dataframe
    logger.info(f"Overlap DataFrame: Entries={reshaped_overlap.shape[0] * reshaped_overlap.shape[1]}, Unique Players={len(np_overlap)}")

def run_processing(df=None):
    """
    Main processing function to create and save datasets.

    Args:
        df (pyspark.sql.DataFrame): The input dataframe containing player statistics.

    Returns:
        GoldDataset: The named tuple containing the filtered DataFrame, numpy array, and dictionaries.
    """
    if df is None:
        df = read_data_from_silver()

    gold_dataset = create_gold_datasets(df)
    save_df_and_dict(gold_dataset)
    log_summary(gold_dataset.df_first_continuous, gold_dataset.np_overlap)

    return gold_dataset

if __name__ == '__main__':
    run_processing()
