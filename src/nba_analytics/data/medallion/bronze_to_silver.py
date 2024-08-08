# Filtered, Cleaned, Augmented
# Define structure, enforce schema, and evolve schema as needed

import numpy as np
import pandas as pd
import json

from ..transformation import cleaning, filtering, processing

from ...utils import filename_grabber
from ...utils.config import settings
from ...utils.logger import get_logger


silver = settings.dataset.silver

logger = get_logger(__name__)

FILTER_AMT = settings.dataset.FILTER_AMT


def create_silver_dataset(df):
    """
    Creates a clean (silver) dataset with player statistics for players.

    Args:
        df (pandas.DataFrame): The input dataframe containing player statistics.

    Returns:
        pandas.DataFrame: The filtered dataframe with player statistics for players who have played at least 5 years in the NBA.
        numpy.array: The numpy array with player statistics for each 5 consecutive years a player plays.
        dict: A dictionary where the key is the player's slug and the value is a list of the player's statistics.
    """
    logger.debug("Filtering dataset for players who have played for more than 5 years...")

    df_filtered = df.copy()

    # df_filtered = extract_positions(df_filtered)
    # df_filtered = extract_team(df_filtered)
    df_filtered = cleaning.clean_rows(df_filtered)
    df_filtered = cleaning.clean_columns(df_filtered)
    df_filtered = cleaning.extract_columns(df_filtered)

    # Create a dictionary where the key is the slug and the value is the rows of the filtered dataset
    dict_df = processing.df_to_dict(df_filtered)

    return df_filtered, dict_df

    
def save_df_and_dict(df_filtered, dict_df):
    """
    Saves the given DataFrame to a CSV file.

    Args:
        df (pandas.DataFrame): The DataFrame to save.
        filename (str): The name of the file to save the DataFrame to.

    Returns:
        None
    """
    silver_dir = filename_grabber.get_specific_dir("silver")
    logger.debug(f"Saving dataset to '{silver_dir}'...")

    # Save the filtered dataset and dictionary to a csv and json file
    df_filtered_file_path = filename_grabber.get_data_file("silver", silver.DATA_FILE)
    df_filtered.to_csv(df_filtered_file_path, index=False)

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
    - df (pandas.DataFrame): The original DataFrame.
    - df_filtered (pandas.DataFrame): The filtered DataFrame.

    Returns:
    None
    """
    logger.debug("Printing summary...")

    # Print the head of the filtered DataFrame
    logger.info(f"Filtered DataFrame Head: {df_cleaned.head()}")

    # Print the number of entries and the number of unique players in the original dataframe
    logger.info(f"Original DataFrame: Entries={len(df)}, Unique Players={len(df['slug'].unique())}")

    # Print the number of entries and the number of unique players in the cleaned dataframe
    logger.info(f"Filtered DataFrame: Entries={len(df_cleaned)}, Unique Players={len(df_cleaned['slug'].unique())}")
    

def run_processing(df=None):
    # Load the data
    if df is None:
        df = pd.read_csv(filename_grabber.get_data_file("bronze", settings.dataset.bronze.DATA_FILE))

    # Create a cleaned dataframe and dictionary
    df_cleaned, dict_df = create_silver_dataset(df)

    # Save the filtered dataframe and dictionary
    save_df_and_dict(df_cleaned, dict_df)

    # Print a summary of the data
    log_summary(df, df_cleaned)

    return df_cleaned, dict_df


if __name__ == '__main__':
    run_processing()    