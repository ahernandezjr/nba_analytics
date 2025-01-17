import numpy as np
import pandas as pd
import torch

from ...utils import filename_grabber
from ...utils.config import settings
from ...utils.logger import get_logger
from sklearn.preprocessing import StandardScaler

gold = settings.dataset.gold

# Create logger
logger = get_logger(__name__)


class NBAPlayerDataset(torch.utils.data.Dataset):
    """
    A custom dataset class for the NBA player statistics dataset.
    """
    def __init__(self, df):
        # Load data and create a list of unique player ids from slug column
        self.df = df
        self.id_list = self.get_id_list()

        # Create a StandardScaler object and fit it to the data
        self.scaler = StandardScaler()
        self.fit_scaler(self.df.drop(columns=['slug']))

    def __len__(self):
        # return len(self.dict)
        # Return length of unique slug values in the slug column
        return len(self.id_list)

    def __getitem__(self, idx):
        # Get player id from index of unique list
        player_id = self.id_list[idx]

        player_data = self.df.copy()

        # Get columns where slug is equal to the idx
        player_data = self.df[self.df['slug'] == player_id]

        # Drop the slug column
        player_data = player_data.drop(columns=['slug'])

        # Sort by Year
        player_data = player_data.sort_values('Year')

        # Convert player_data to tensor
        player_data = torch.from_numpy(player_data.values)

        # Apply StandardScaler to player_data
        player_data = self.transform(player_data)
        player_data = torch.from_numpy(player_data)

        # Create target by taking the last row of player_data
        targets = player_data[-1]

        return player_data, targets

    def get_id_list(self):
        return self.df['slug'].unique()

    # def get_dict(self):
    #     return self.dict

    def fit_scaler(self, data):
        """
        Fit the StandardScaler to the data.

        Args:
            data (torch.Tensor): The data to fit the scaler on.
        """
        self.scaler.fit(data.values)

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


def create_dataset(df_filename=gold.DATA_FILE_CONTINUOUS_FIRST,
                   dict_filename=gold.DATA_FILE_CONTINUOUS_FIRST_JSON):
    """
    Creates a custom dataset for the NBA player statistics.

    Args:
        df (pandas.DataFrame): The DataFrame containing the player statistics.

    Returns:
        NBAPlayerDataset: The custom dataset for the NBA player statistics.
    """
    logger.info("Creating dataset...")

    # Get the data file and dictionary file
    df_filename = filename_grabber.get_data_file('gold', gold.DATA_FILE_CONTINUOUS_FIRST)
    dict_filename = filename_grabber.get_data_file('gold', gold.DATA_FILE_CONTINUOUS_FIRST_JSON)

    logger.info(f"Loading data from {df_filename} and {dict_filename}...")
    # Load the dataset with proper numeric types
    # df = pd.read_csv(df_filename).apply(pd.to_numeric, errors='coerce')
    df = pd.read_csv(df_filename)

    # Load the dictionary with proper numeric types
    df_dict = pd.read_json(dict_filename, typ='series').to_dict()

    # Create the dataset
    dataset = NBAPlayerDataset(df)

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