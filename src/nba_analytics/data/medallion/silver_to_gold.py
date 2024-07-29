# Bussiness-Level Aggregates
# Deliver continuously updated, clean data TODOwnstream users and apps
from collections import namedtuple

import numpy as np
import pandas as pd

from ..transformation import processing, filtering

from ...utils import filename_grabber
from ...utils.config import settings
from ...utils.logger import get_logger


logger = get_logger(__name__)

FILTER_AMT = settings.FILTER_AMT


# Define a named tuple for the return value of create_gold_datasets
GoldDataset = namedtuple('GoldDataset', ['df_continuous', 'df_first_continuous', 'np_overlap', 'dict_continuous', 'dict_first_continuous'])


def create_gold_datasets(df):
    """
    Creates a dataset with player statistics for players who have played at least FILTER_AMT years in the NBA.

    Args:
        df (pandas.DataFrame): The input dataframe containing player statistics.

    Returns:
        GoldDataset: A named tuple containing:
            - df_continuous (pandas.DataFrame): Filtered dataframe with continuous stretches of at least FILTER_AMT years.
            - df_first_continuous (pandas.DataFrame): Filtered dataframe with the first continuous stretch of at least FILTER_AMT years.
            - np_overlap (numpy.array): Numpy array with player statistics for each continuous period of FILTER_AMT years.
            - dict_continuous (dict): Dictionary with player statistics for continuous stretches.
            - dict_first_continuous (dict): Dictionary with player statistics for first continuous stretches.
    """
    logger.debug(f"Filtering dataset for players who have played for more than {FILTER_AMT} years...")

    df_filtered = filtering.filter_columns(df.copy())

    # Filter the dataset to include only players who have continuous stretches of at least FILTER_AMT years
    df_continuous = filtering.filter_atleast_continuous_years(df_filtered, FILTER_AMT)
    df_first_continuous = filtering.filter_first_continuous_years(df_filtered, FILTER_AMT)

    # Create overlap dataset
    np_overlap = processing.create_overlap_data(df_continuous)

    # Convert the DataFrames to dictionaries
    dict_continuous = processing.df_to_dict(df_continuous)
    dict_first_continuous = processing.df_to_first_years_dict(df_first_continuous)

    return GoldDataset(df_continuous, df_first_continuous, np_overlap, dict_continuous, dict_first_continuous)


def save_df_and_dict(df_first_continuous, np_overlap, dict_continuous):
    """
    Saves the given DataFrame, numpy array, and dictionary to respective files.

    Args:
        df_first_continuous (pandas.DataFrame): The DataFrame to save.
        np_overlap (numpy.array): The numpy array to save.
        dict_continuous (dict): The dictionary to save.

    Returns:
        None
    """
    first_continuous_path = filename_grabber.get_data_file('gold', settings.dataset.gold.DATA_FILE_CONTINUOUS_FIRST)
    overlap_path = filename_grabber.get_data_file('gold', settings.dataset.gold.DATA_FILE_CONTINUOUS_OVERLAP)
    dict_path = filename_grabber.get_data_file('gold', settings.dataset.gold.DATA_FILE_CONTINUOUS_FIRST_JSON)

    logger.debug(f"Saving first_continuous dataset to '{first_continuous_path}'...")
    logger.debug(f"Saving overlap dataset to '{overlap_path}'...")

    df_first_continuous.to_csv(first_continuous_path, index=False)
    np.save(overlap_path, np_overlap)
    with open(dict_path, 'w') as json_file:
        json.dump(dict_continuous, json_file, indent=4)

    logger.debug(f"First-continuous dataset saved to: '{first_continuous_path}'.")
    logger.debug(f"Overlap dataset saved to: '{overlap_path}'.")
    logger.debug(f"Filtered dictionary saved to: '{dict_path}'.")


def log_summary(df_first_continuous, np_overlap):
    """
    Prints a summary of the given DataFrame and its filtered version.

    Args:
        df_first_continuous (pandas.DataFrame): The filtered DataFrame.
        np_overlap (numpy.array): The numpy array with player statistics for each continuous period of FILTER_AMT years.

    Returns:
        None
    """
    logger.debug("Printing summary...")

    logger.info(df_first_continuous.head())

    reshaped_overlap = np_overlap.reshape(np_overlap.shape[0], FILTER_AMT, -1)

    logger.info(f"First Continuous DataFrame: Entries={len(df_first_continuous)}, Unique Players={len(df_first_continuous['slug'].unique())}")
    logger.info(f"Overlap DataFrame: Entries={reshaped_overlap.shape[0] * reshaped_overlap.shape[1]}, Unique Players={len(np_overlap)}")

def run_processing(df=None):
    """
    Main processing function to create and save datasets.

    Args:
        df (pandas.DataFrame): The input dataframe containing player statistics.

    Returns:
        GoldDataset: The named tuple containing the filtered DataFrame, numpy array, and dictionaries.
    """
    if df is None:
        df = pd.read_csv(filename_grabber.get_data_file("silver", settings.DATA_FILE))

    gold_dataset = create_gold_datasets(df)

    save_df_and_dict(gold_dataset.df_first_continuous, gold_dataset.np_overlap, gold_dataset.dict_first_continuous)
    log_summary(gold_dataset.df_first_continuous, gold_dataset.np_overlap)

    return gold_dataset


if __name__ == '__main__':
    run_processing()