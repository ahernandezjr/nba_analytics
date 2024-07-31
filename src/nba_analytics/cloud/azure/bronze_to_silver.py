# Filtered, Cleaned, Augmented
# Define structure, enforce schema, and evolve schema as needed

from collections import namedtuple
import json
import numpy as np
import pyspark.sql.functions as F
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, IntegerType

from ...data.transformation import cleaning

from ...utils import filename_grabber
from ...utils.config import settings
from ...utils.logger import get_logger


silver = settings.dataset.silver

logger = get_logger(__name__)

FILTER_AMT = settings.dataset.FILTER_AMT

# Initialize Spark session
spark = SparkSession.builder.appName("NBA Analysis").getOrCreate()

def read_data_from_bronze():
    """
    Reads the data from the bronze dataset using PySpark.

    Returns:
        pyspark.sql.DataFrame: The input dataframe containing player statistics.
    """
    bronze_data_path = "/mnt/nba-data/bronze/nba_player_stats.csv"
    df = spark.read.csv(bronze_data_path, header=True, inferSchema=True)
    return df

def create_silver_dataset(df):
    """
    Creates a clean (silver) dataset with player statistics for players.

    Args:
        df (pyspark.sql.DataFrame): The input dataframe containing player statistics.

    Returns:
        pyspark.sql.DataFrame: The filtered dataframe with player statistics for players who have played at least 5 years in the NBA.
        dict: A dictionary where the key is the player's slug and the value is a list of the player's statistics.
    """
    logger.debug("Filtering dataset for players who have played for more than 5 years...")

    df_filtered = df

    # Apply cleaning functions
    df_filtered = cleaning.clean_rows(df_filtered)
    df_filtered = cleaning.clean_columns(df_filtered)
    df_filtered = cleaning.clean_nontensor_values(df_filtered)

    # Create a dictionary where the key is the slug and the value is the rows of the filtered dataset
    dict_df = df_filtered.groupBy('slug').agg(F.collect_list(F.struct(*df_filtered.columns)).alias("records")).collect()
    dict_df = {row['slug']: [record.asDict() for record in row['records']] for row in dict_df}

    return df_filtered, dict_df

def save_df_and_dict(df_filtered, dict_df):
    """
    Saves the given DataFrame to a CSV file.

    Args:
        df_filtered (pyspark.sql.DataFrame): The DataFrame to save.
        dict_df (dict): The dictionary to save.

    Returns:
        None
    """
    silver_dir = filename_grabber.get_specific_dir("silver")
    logger.debug(f"Saving dataset to '{silver_dir}'...")

    # Save the filtered dataset and dictionary to a csv and json file
    df_filtered_file_path = filename_grabber.get_data_file("silver", silver.DATA_FILE)
    df_filtered.write.csv(df_filtered_file_path, header=True, mode='overwrite')

    logger.debug(f"Cleaned Silver dataset saved to: '{df_filtered_file_path}'.")

    # Save dictionary to json
    dict_df_file_path = filename_grabber.get_data_file("silver", silver.DATA_FILE_JSON)
    with open(dict_df_file_path, 'w') as json_file:
        json.dump(dict_df, json_file, indent=4)

    logger.debug(f"Cleaned Silver dictionary saved to: '{dict_df_file_path}'.")

def log_summary(df, df_cleaned):
    """
    Prints a summary of the given DataFrame and its filtered version.

    Parameters:
    - df (pyspark.sql.DataFrame): The original DataFrame.
    - df_cleaned (pyspark.sql.DataFrame): The filtered DataFrame.

    Returns:
    None
    """
    logger.debug("Printing summary...")

    # Print the head of the filtered DataFrame
    df_cleaned.show()

    # Print the number of entries and the number of unique players in the original dataframe
    logger.info(f"Original DataFrame: Entries={df.count()}, Unique Players={df.select('slug').distinct().count()}")

    # Print the number of entries and the number of unique players in the cleaned dataframe
    logger.info(f"Filtered DataFrame: Entries={df_cleaned.count()}, Unique Players={df_cleaned.select('slug').distinct().count()}")

def run_processing(df=None):
    """
    Main processing function to create and save datasets.

    Args:
        df (pyspark.sql.DataFrame): The input dataframe containing player statistics.

    Returns:
        pyspark.sql.DataFrame: The cleaned dataframe and dictionary.
    """
    if df is None:
        df = read_data_from_bronze()

    # Create a cleaned dataframe and dictionary
    df_cleaned, dict_df = create_silver_dataset(df)

    # Save the filtered dataframe and dictionary
    save_df_and_dict(df_cleaned, dict_df)

    # Print a summary of the data
    log_summary(df, df_cleaned)

    return df_cleaned, dict_df

if __name__ == '__main__':
    run_processing()
