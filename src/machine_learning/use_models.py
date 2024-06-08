import os, sys

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader

from sklearn.preprocessing import StandardScaler, MinMaxScaler 

import matplotlib.pyplot as plt

from ..data_process.dataset_torch import NBAPlayerDataset, get_dataset_example

from .train_models import get_model
from .models.lstm import get_custom_lstm, get_nn_LSTM
from .models.neuralnet import CustomNN

from ..utils.config import settings
from ..utils.logger import get_logger


# Create logger
logger = get_logger(__name__)


# Set configs from settings
DATA_DIR = settings.DATA_DIR
DATA_FILE_NAME = settings.DATA_FILE_NAME
DATA_FILE_5YEAR_NAME = settings.DATA_FILE_5YEAR_NAME
DATA_FILE_5YEAR_JSON_NAME = settings.DATA_FILE_5YEAR_JSON_NAME


# set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Load the dictionary with proper numeric types
df_dict = pd.read_json(DATA_FILE_5YEAR_JSON_NAME, typ='series').to_dict()

# Instantiate the dataset
nba_dataset = NBAPlayerDataset(df_dict)

# Create a training DataLoader and test DataLoader
train_loader = DataLoader(nba_dataset, batch_size=32, shuffle=True)
test_loader  = DataLoader(nba_dataset, batch_size=32, shuffle=False)

# Define hyperparameters
learning_rate = 0.001

# input_size = 39 # number of features
input_size  = nba_dataset[0][0].shape[1] # number of features
hidden_size = 39 # number of features in hidden state
# output_size = 39 # number of output classes 
output_size = nba_dataset[0][0].shape[1] # number of features
num_layers = 3 # number of stacked lstm layers


def prompt_user(pth_files):
    """
    Prompt the user to select a .pth file
    """
    # Prompt user to select a .pth file
    logger.info('Select a .pth file:')
    for i, f in enumerate(pth_files):
        logger.info(f'{i}: {f}')

    # Get user input
    user_input = input('Enter the number of the .pth file: ')
    try:
        user_input = int(user_input)
    except ValueError:
        logger.error('Invalid input. Exiting...')
        sys.exit(1)

    # If user input is not within the range of the number of .pth files, exit
    if user_input < 0 or user_input >= len(pth_files):
        logger.error('Invalid input. Exiting...')
        sys.exit(1)

    logger.info(f'Selected user input: {user_input}')

    return user_input


def load_model(pth_files, load_index):
    """
    Load a model from the data directory
    """
    # Load the selected .pth file
    pth_file = pth_files[load_index]
    logger.info(f'Loading {pth_file} and creating model...')

    # Load the model
    model_name = pth_file.split('.')[0]
    model = get_model(model_name=model_name,
                        input_size=input_size,
                        hidden_size=hidden_size,
                        output_size=output_size,
                        num_layers=num_layers)
    model.load_state_dict(torch.load(os.path.join(DATA_DIR, pth_file)))
    model.eval()

    logger.info(f'Loaded model: {model}')
    return model


def use_model(file_index=None):
    """
    Use a model from the data directory
    """
    # If data directory does not exist, exit
    if not os.path.exists(DATA_DIR):
        logger.error(f'Data directory {DATA_DIR} does not exist. Exiting...')
        sys.exit(1)

    # If no files within the data directory end with .pth , exit
    pth_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.pth')]
    logger.info(f'Found .pth files: {pth_files}')
    if not pth_files:
        logger.error(f'No .pth files in {DATA_DIR}. Exiting...')
        sys.exit(1)

    # Prompt user to select a .pth file if none is given
    if file_index is None:
        load_index = prompt_user(pth_files)
        logger.info(f'Using prompted file index {load_index}.')
    else:
        load_index = file_index
        logger.info(f'Using argument file index {load_index}.')

    model = load_model(pth_files, load_index)

    dataset_index = 0
    player_data, targets = get_dataset_example(dataset_index)

    # Test the model on player_data
    outputs = model(player_data)

    logger.info("Model Results:")
    logger.info(f"Inputs: {player_data}")
    logger.info(f"Outputs: {outputs[0]}")
    logger.info(f"Targets: {targets}")
