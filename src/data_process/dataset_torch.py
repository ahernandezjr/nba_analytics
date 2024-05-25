import os, sys
import pandas as pd
import torch

from ..utils.config import settings
from ..utils.logger import getLogger


# Set configs from settings
DATA_DIR = settings.DATA_DIR
DATA_FILE_NAME = settings.DATA_FILE_NAME
DATA_FILE_5YEAR_NAME = settings.DATA_FILE_5YEAR_NAME
DATA_FILE_5YEAR_JSON_NAME = settings.DATA_FILE_5YEAR_JSON_NAME


# Create logger
logger = getLogger(__name__)


class NBAPlayerDataset(torch.utils.data.Dataset):
    """
    A custom dataset class for the NBA player statistics dataset.
    """
    def __init__(self, dict):
        self.dict = dict

    def __len__(self):
        return len(self.dict)

    def __getitem__(self, idx):
        # Get first item in the dict
        player_id = list(self.dict.keys())[idx]

        # Get player data list for each year
        player_data = self.dict[player_id]

        # print all types of values in player_data
        # print(player_data)
        # print([list(map(lambda x: type(x), player_data[i])) for i in range(len(player_data))])

        # Convert values to proper types (i.e. str should be converted to int or float)
        player_data = [list(map(lambda x: int(x) if isinstance(x, str) and x.isdigit() else float(x) if isinstance(x, str) and x.replace('.', '', 1).isdigit() else x, player_data[i].values())) for i in range(len(player_data))]

        # Remove all non-tensor compatible values from player_data
        player_data = [list(filter(lambda x: type(x) in [int, float], player_data[i])) for i in range(len(player_data))]

        

        # Convert player_data to tensor
        player_data = torch.tensor(player_data)

        return player_id, player_data


def create_dataset(df_filename=DATA_FILE_5YEAR_NAME,
                   dict_filename=DATA_FILE_5YEAR_JSON_NAME):
    """
    Creates a custom dataset for the NBA player statistics.

    Args:
        df (pandas.DataFrame): The DataFrame containing the player statistics.

    Returns:
        NBAPlayerDataset: The custom dataset for the NBA player statistics.
    """
    logger.debug("Creating dataset...")

    # Create df filename
    # filename_df = os.path.join(DATA_DIR, f"{filename}.csv")
    # Create df_dict filename
    # filename_dict = filename_df = os.path.join(DATA_DIR, f"{filename}.json")

    # Load the dataset with proper numeric types
    df = pd.read_csv(df_filename).apply(pd.to_numeric, errors='coerce')

    # Load the dictionary with proper numeric types
    df_dict = pd.read_json(dict_filename, typ='series').to_dict()

    # Create the dataset
    dataset = NBAPlayerDataset(df_dict)

    return dataset


def test_dataset():
    """
    Tests the NBAPlayerDataset class.
    """
    # Create the dataset
    dataset = create_dataset()

    # Check first 5 items in the dataset
    for i in range(5):
        player_id, player_stats = dataset[i]
        print(f"Player ID: {player_id}")
        print(f"Player Stats: {player_stats}")