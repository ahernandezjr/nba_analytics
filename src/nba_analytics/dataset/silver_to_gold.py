# Bussiness-Level Aggregates
# Deliver continuously updated, clean data to downstream users and apps

import numpy as np
import pandas as pd

from . import cleaning
from . import filtering

from ..utils import filename_grabber
from ..utils.config import settings
from ..utils.logger import get_logger


logger = get_logger(__name__)

FILTER_AMT = settings.FILTER_AMT


def create_overlap_data(df):
    """
    Creates a DataFrame of players who have played for more than 5 years.

    Args:
        df (pandas.DataFrame): The input DataFrame containing player statistics.

    Returns:
        numpy.array: The numpy array with player statistics for each 5 consecutive years a player plays.
    """
    logger.debug("Creating overlap data...")

    df_overlap = df.copy()

    # Convert DataFrame to numpy array
    np_overlap = df_overlap.to_numpy()

    # Get all unique slugs
    np_uniques = np.unique(np_overlap[:, 0])

    # Create a list of numpy arrays for each player
    np_overlap = [np_overlap[np_overlap[:, 0] == unique] for unique in np_uniques]

    # Create new numpy array
    np_out = []

    for player_stats in np_overlap:
        for i in range(len(player_stats) - 5):
            if player_stats[i][1] + 5 == player_stats[i + 5][1]:
                np_out.append(player_stats[i:i+5])

    # Convert list of numpy arrays to numpy array
    np_out = np.array(np_out)

    # Remove the slug column
    np_out = np_out[:, :, 1:]

    # Convert 3D numpy array to 2D numpy array
    np_out = np_out.reshape(np_out.shape[0], -1)

    return np_out


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
    np_overlap = create_overlap_data(df_overlap)

    # Create a dictionary where the key is the slug and the value is the rows of the filtered dataset
    dict_df = df_filtered.groupby('slug').apply(lambda x: x.to_dict(orient='records'))

    return df_first_five, np_overlap, dict_df



def save_df_and_dict(df_tensor_ready, np_overlap, df_dict):
    """
    Saves the given DataFrame to a CSV file.

    Args:
        df (pandas.DataFrame): The DataFrame to save.
        filename (str): The name of the file to save the DataFrame to.

    Returns:
        None
    """
    logger.debug(f"Saving dataset to '{filename_grabber.get_data_file()}'...")

    # Create df filename
    # filename_df = os.path.join(os.getcwd(), DATASET_DIR, f"{filename}.csv")
    # Create df_dict filename
    # filename_dict = filename_df = os.path.join(os.getcwd(), DATASET_DIR, f"{filename}.json")


    # Save the filtered dataset and dictionary to a csv and json file
    # df1.to_csv(DATA_FILE_5YEAR_NAME, index=False)
    df_tensor_ready.to_csv(filename_grabber.get_data_file_5year_tensor(), index=False)

    # Save 3D numpy array to csv
    # np.savetxt(filename_grabber.get_data_file_5year_overlap(), np_overlap, delimiter=',', fmt='%s')
    # np_overlap.to_csv(filename_grabber.get_data_file_5year_overlap(), index=False)
    np.savez(filename_grabber.get_data_file_5year_overlap() + '.npz', np_overlap)
    # Save dictionary to json
    df_dict.to_json(filename_grabber.get_data_file_5year_json(), indent=4)

    # logger.debug(f"Filtered dataset saved to: '{DATA_FILE_5YEAR_NAME}'.")
    logger.debug(f"Tensor-ready dataset saved to: '{filename_grabber.get_data_file_5year_tensor()}'.")
    logger.debug(f"Overlap dataset saved to: '{filename_grabber.get_data_file_5year_overlap()}'.")
    logger.debug(f"Filtered dictionary saved to: '{filename_grabber.get_data_file_5year_json()}'.")


def run_processing():
    # Load the data
    df = pd.read_csv(filename_grabber.get_data_file())

    # Create a dataframe of players who have played for more than 5 years
    df_tensor_ready, np_overlap, dict_df = create_silver_dataset(df)

    # Save the filtered dataframe and dictionary
    save_df_and_dict(df_tensor_ready, np_overlap, dict_df)

    return df_tensor_ready, np_overlap, dict_df


def print_summary(df_tensor_ready, np_overlap):
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
    print(df_tensor_ready.head())

    # Reshape the 2D numpy array to its original shape
    reshaped_overlap = np_overlap.reshape(np_overlap.shape[0], FILTER_AMT, -1)

    # Print the number of entries and the number of unique players in the original dataframe
    # print(f"Original DataFrame: Entries={len(df)}, Unique Players={len(df['slug'].unique())}")

    # Print the number of entries and the number of unique players in the tensor dataframe
    print(f"Filtered DataFrame: Entries={len(df_tensor_ready)}, Unique Players={len(df_tensor_ready['slug'].unique())}")

    # Print the number of entries and the number of unique players in the overlap dataframe
    print(f"Filtered DataFrame: Entries={reshaped_overlap.shape[0] * reshaped_overlap.shape[1]}, Unique Players={len(np_overlap)}")


if __name__ == '__main__':
    run_processing()    