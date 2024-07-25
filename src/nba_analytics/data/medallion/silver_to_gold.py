# Bussiness-Level Aggregates
# Deliver continuously updated, clean data TODOwnstream users and apps

import numpy as np
import pandas as pd

from .transformation import cleaning, processing, filtering

from ..utils import filename_grabber
from ..utils.config import settings
from ..utils.logger import get_logger


logger = get_logger(__name__)

FILTER_AMT = settings.FILTER_AMT


def create_gold_datasets(df):
    """
    Creates a dataset with player statistics for players who have played at least 5 years in the NBA.

    Args:
        df (pandas.DataFrame): The input dataframe containing player statistics.

    Returns:
        pandas.DataFrame: The filtered dataframe with player statistics for players who have played at least 5 years in the NBA.
        numpy.array: The numpy array with player statistics for each 5 consecutive years a player plays.
        dict: A dictionary where the key is the player's slug and the value is a list of the player's statistics.
    """
    logger.debug("Filtering dataset for players who have played for more than 5 years...")

    df_filtered = df.copy()
    
    df_overlap, df_first_five = filtering.filter_players_over_5_years(df_filtered)
    np_overlap = processing.create_overlap_data(df_overlap)
    df_tensor_ready = cleaning.clean_nontensor_values(df_first_five)


    # TODO: Convert the DataFrames to a dictionary




    # Create a dictionary where the key is the slug and the value is the rows of the filtered dataset
    dict_df = df_filtered.groupby('slug').apply(lambda x: x.to_dict(orient='records'))

    return df_first_five, np_overlap, dict_df


def save_df_and_dict(df_first_five, np_overlap, dict_df):
    """
    Saves the given DataFrame to a CSV file.

    Args:
        df_first_five (pandas.DataFrame): The DataFrame to save.
        np_overlap (numpy.array): The numpy array to save.
        dict_df (dict): The dictionary to save.

    Returns:
        None
    """
    first_five_path = filename_grabber.get_data_file('gold', settings.DATA_FILE_5YEAR_NAME)
    tensor_ready_path = filename_grabber.get_data_file('gold', settings.DATA_FILE_5YEAR_TENSOR_NAME)
    overlap_path = filename_grabber.get_data_file('gold', settings.DATA_FILE_5YEAR_OVERLAP)
    dict_path = filename_grabber.get_data_file('gold', settings.DATA_FILE_5YEAR_JSON_NAME)

    logger.debug(f"Saving df_first_five dataset to '{first_five_path}'...")
    logger.debug(f"Saving overlap dataset to '{overlap_path}'...")

    # Save the filtered dataset and dictionary to a csv and json file
    df_first_five.to_csv(first_five_path, index=False)
    np.savez(overlap_path, np_overlap)
    dict_df.to_json(dict_path, indent=4)

    logger.debug(f"First-five dataset saved to: '{first_five_path}'.")
    logger.debug(f"Overlap dataset saved to: '{overlap_path}'.")
    logger.debug(f"Filtered dictionary saved to: '{dict_path}'.")


def log_summary(df_first_five, np_overlap):
    """
    Prints a summary of the given DataFrame and its filtered version.

    Parameters:
    - df_first_five (pandas.DataFrame): The filtered DataFrame.
    - np_overlap (numpy.array): The numpy array with player statistics for each 5 consecutive years a player plays.

    Returns:
    None
    """
    logger.debug("Printing summary...")

    # Print the head of the filtered DataFrame
    logger.info(df_first_five.head())

    # Reshape the 2D numpy array to its original shape
    reshaped_overlap = np_overlap.reshape(np_overlap.shape[0], FILTER_AMT, -1)

    # Print the number of entries and the number of unique players in the filtered dataframe
    logger.info(f"First Five DataFrame: Entries={len(df_first_five)}, Unique Players={len(df_first_five['slug'].unique())}")

    # Print the number of entries and the number of unique players in the overlap dataframe
    logger.info(f"Overlap DataFrame: Entries={reshaped_overlap.shape[0] * reshaped_overlap.shape[1]}, Unique Players={len(np_overlap)}")


def run_processing(df=None):
    # Load the data
    if df is None:
        df = pd.read_csv(filename_grabber.get_data_file("silver", settings.DATA_FILE_NAME))

    # Create a dataframe of players who have played for more than 5 years
    df_first_five, np_overlap, dict_df = create_gold_datasets(df)

    # Save the filtered dataframe and dictionary
    save_df_and_dict(df_first_five, np_overlap, dict_df)

    # Print a summary of the data
    log_summary(df_first_five, np_overlap)

    return df_first_five, np_overlap, dict_df


if __name__ == '__main__':
    run_processing()    