import os

import numpy as np
import pandas as pd
import torch

from ..utils import filename_grabber
from ..utils.config import settings
from ..utils.logger import get_logger
from sklearn.preprocessing import StandardScaler


FILTER_AMT = settings.FILTER_AMT


# Create logger
logger = get_logger(__name__)


class NBAPlayerDataset(torch.utils.data.Dataset):
    """
    A custom dataset class for the NBA player statistics dataset.
    """
    def __init__(self, arr):
        # Load data and create a list of unique player ids from slug column
        self.arr = arr
        # self.id_list = self.get_id_list()

        # Create a StandardScaler object and fit it to the data
        self.scaler = StandardScaler()
        self.fit_scaler(self.arr)

    def __len__(self):
        # return len(self.dict)
        # Return length of unique slug values in the slug column
        return self.arr.shape[0]

    def __getitem__(self, idx):
        player_data = self.arr[idx]

        # Sort by Year
        player_data[player_data[:, 0].argsort()]

        # Apply StandardScaler to player_data
        player_data = self.transform(player_data)
        player_data = torch.from_numpy(player_data)

        # Create target by taking the last row of player_data
        targets = player_data[-1]

        return player_data, targets

    # def get_id_list(self):
    #     '''
    #     Get all unique values from the first column of the 3D numpy array.
    #     '''
    #     return np.unique(self.arr[:, :, 0])

    # def get_dict(self):
    #     return self.dict

    def fit_scaler(self, data):
        """
        Fit the StandardScaler to the data.

        Args:
            data (torch.Tensor): The data to fit the scaler on.
        """
        reshaped_data = data.reshape(-1, data.shape[-1])
        self.scaler.fit(reshaped_data)

    def transform(self, data):
        """
        Transform the data using the fitted scaler.

        Args:
            data (torch.Tensor): The data to transform.

        Returns:
            torch.Tensor: The transformed data.
        """
        return self.scaler.transform(data)

    def inverse_fit_scaler(self, data):
        """
        Unfit the data using the inverse transformation of the fitted scaler.

        Args:
            data (torch.Tensor): The data to untransform.

        Returns:
            torch.Tensor: The untransformed data.
        """
        return self.scaler.inverse_transform(data)


def create_dataset(df_filename=settings.DATA_FILE_5YEAR_NAME,
                   np_overlap_filename=settings.DATA_FILE_5YEAR_OVERLAP,
                   dict_filename=settings.DATA_FILE_5YEAR_JSON_NAME):
    """
    Creates a custom dataset for the NBA player statistics.

    Args:
        df (pandas.DataFrame): The DataFrame containing the player statistics.

    Returns:
        NBAPlayerDataset: The custom dataset for the NBA player statistics.
    """
    logger.info("Creating dataset...")

    logger.info(f"Loading data from {df_filename}, {np_overlap_filename}, and {dict_filename}...")
    
    # Load the dataset with proper numeric types
    # df = pd.read_csv(df_filename).apply(pd.to_numeric, errors='coerce')

    # Load the numpy array with proper numeric types
    np_overlap = np.loadtxt(filename_grabber.get_data_file("gold", np_overlap_filename),
                            delimiter=",")

    # Reshape the 2D numpy array to its original shape
    np_overlap = np_overlap.reshape(np_overlap.shape[0], FILTER_AMT, -1)

    # Load the dictionary with proper numeric types
    df_dict = pd.read_json(filename_grabber.get_data_file("gold", dict_filename),
                           typ='series').to_dict()

    # Create the dataset
    dataset = NBAPlayerDataset(np_overlap)

    logger.info("Dataset created successfully.")

    return dataset


def print_dataset_example():
    """
    Tests the NBAPlayerDataset class.
    """
    logger.info("Testing creation of NBAPlayerDataset class...")
    # Create the dataset
    dataset = create_dataset()

    # Check first 5 items in the dataset
    for i in range(6):
        player_data, targets = dataset[i]
        # print(f"Player ID: {player_id}")
        logger.info(f"Player Data:\n{targets}\nPlayers Targets:\n{player_data}")


def get_dataset_example(index=None):
    """
    """
    logger.info("Getting example from dataset...")

    # Create the dataset
    dataset = create_dataset()

    # If no index is provided, get a random index of length of the dataset
    if index is None:
        index = np.random.randint(len(dataset))
    
    # Return the player data and targets
    player_data, targets = dataset[index]
    return player_data, targets



# EXTRA: This is the original code that was in the NBAPlayerDataset class
# def __init__(self, dict):
#     self.dict = dict
#     self.scaler = StandardScaler()

# def __len__(self):
#     return len(self.dict)

# def __getitem__(self, idx):
#     # Get first item in the dict
#     player_id = list(self.dict.keys())[idx]

#     # Get player data list for each year
#     player_data = self.dict[player_id]

#     # print all types of values in player_data
#     # print(player_data)
#     # print([list(map(lambda x: type(x), player_data[i])) for i in range(len(player_data))])

#     # Convert values to proper types (i.e. str should be converted to int or float)
#     player_data = [list(map(lambda x: int(x) if isinstance(x, str) and x.isdigit() else float(x) if isinstance(x, str) and x.replace('.', '', 1).isdigit() else x, player_data[i].values())) for i in range(len(player_data))]

#     # Remove all non-tensor compatible values from player_data
#     player_data = [list(filter(lambda x: type(x) in [int, float], player_data[i])) for i in range(len(player_data))]

#     # Convert player_data to tensor
#     player_data = torch.tensor(player_data)

#     # Create target by taking the last row of player_data
#     targets = player_data[-1]

#     # Apply StandardScaler to player_data
#     player_data = self.scaler.transform(player_data)

#     return player_data, targets