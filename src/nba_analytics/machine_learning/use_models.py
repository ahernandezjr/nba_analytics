import os, sys

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader

from ..data.dataset.torch_overlap import create_dataset

from .train_models import get_model

from ..utils import filename_grabber
from ..utils.config import settings
from ..utils.logger import get_logger


gold = settings.dataset.gold

# Create logger
logger = get_logger(__name__)


MODELS_DIR = filename_grabber.get_models_dir()

FILTER_AMT = settings.dataset.FILTER_AMT

# set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Load the dataset from the tensor file
# df = pd.read_csv(filename_grabber.get_data_file(dir_name='gold',
#                                                 data_file_name=gold.DATA_FILE_CONTINUOUS_FIRST))

# # Load the numpy array with proper numeric types
# np_overlap = np.loadtxt(filename_grabber.get_data_file(dir_name='gold',
#                                                        data_file_name=gold.DATA_FILE_CONTINUOUS_OVERLAP),
#                         delimiter=",")

# # Reshape the 2D numpy array to its original shape
# np_overlap = np_overlap.reshape(np_overlap.shape[0], FILTER_AMT, -1)

# # Load the dictionary with proper numeric types
# df_dict = pd.read_json(filename_grabber.get_data_file(dir_name='gold',
#                                                       data_file_name=gold.DATA_FILE_CONTINUOUS_FIRST_JSON),
#                        typ='series').to_dict()

# Instantiate the dataset
nba_dataset = create_dataset()

# Create a training DataLoader and test DataLoader
train_loader = DataLoader(nba_dataset, batch_size=32, shuffle=True)
test_loader  = DataLoader(nba_dataset, batch_size=32, shuffle=False)

# Define hyperparameters
learning_rate = 0.01


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

    model_name = pth_file.split('.')[0]

    # input_size = 39 # number of features
    input_size  = nba_dataset[0][0].shape[1] # number of features
    hidden_size = nba_dataset[0][0].shape[1] # number of features in hidden state
    output_size = nba_dataset[0][0].shape[1] # number of features
    num_layers = 5 # number of stacked lstm layers

    if model_name == 'nn_many_to_one':
        input_size = nba_dataset[0][0][:-1].flatten().shape[0]
     
    # Load the model
    model = get_model(model_name=model_name,
                        input_size=input_size,
                        hidden_size=hidden_size,
                        output_size=output_size,
                        num_layers=num_layers)
    
    model.load_state_dict(torch.load(os.path.join(os.getcwd(), MODELS_DIR, pth_file)))
    model.eval()

    logger.info(f'Loaded model: {model}')
    return model


def use_model(file_index=None):
    """
    Use a model from the data directory
    """
    all_models = []
    all_inputs = []
    all_outputs = []

    # If data directory does not exist, exit
    if not os.path.exists(MODELS_DIR):
        logger.error(f'Data directory {MODELS_DIR} does not exist. Exiting...')
        sys.exit(1)

    # If no files within the data directory end with .pth , exit
    pth_files = [f for f in os.listdir(MODELS_DIR) if f.endswith('.pth')]
    logger.info(f'Found .pth files: {pth_files}')
    if not pth_files:
        logger.error(f'No .pth files in {MODELS_DIR}. Exiting...')
        sys.exit(1)

    if file_index <= -1:
        load_index = 0
        logger.info(f'Using all indices ({file_index} given).')
    # Prompt user to select a .pth file if none is given
    elif file_index is None:
        load_index = prompt_user(pth_files)
        logger.info(f'Using prompted file index {load_index}.')
    else:
        load_index = file_index
        logger.info(f'Using argument file index {load_index}.')

    for f in os.listdir(MODELS_DIR):
        model = load_model(pth_files, load_index)
        model = model.to(device)

        all_models.append(model.name)

        model_inputs = []
        model_outputs = []

        # Iterate through entire dataset
        for inputs, _ in nba_dataset:
            # dataset_index = 0
            # inputs, targets = get_dataset_example(dataset_index)

            inputs = inputs.float()
            # targets = targets.float()
            
            inputs = inputs.to(device)
            # targets = targets.to(device)

            input_0 = inputs[0]
            outputs = None
            if model.name == 'lstm_model':
                outputs = model.forward(inputs)[0]
                outputs = outputs[0]
            elif model.name == 'nn_one_to_one_model':
                inputs = inputs[0]
                outputs = model.forward(inputs)
            elif model.name == 'nn_many_to_one_model':
                inputs = inputs[:-1].view(-1)
                outputs = model.forward(inputs)
            else:
                outputs = model.forward(inputs)
            
            input_0 = input_0.detach().cpu().numpy()
            inputs = inputs.detach().cpu().numpy()
            outputs = outputs.detach().cpu().numpy()

            # Test the model on player_data
            model_inputs.append(input_0)
            model_outputs.append(outputs)

        all_inputs.append(model_inputs)
        all_outputs.append(model_outputs)

        logger.info(f"Model {model.name} Results Example:")
        logger.info(f"Inputs: {all_inputs[0]}")
        logger.info(f"Outputs: {all_outputs[0]}")
        # logger.info(f"Targets: {targets}")

        # logger.info(f"De-scaled Inputs: {nba_dataset.inverse_fit_scaler(inputs.detach().cpu().numpy())}")
        # logger.info(f"De-scaled Outputs: {nba_dataset.inverse_fit_scaler(outputs[0].detach().cpu().numpy())}")
            
        load_index += 1
        if file_index != -1:
            return all_models, model_inputs, model_outputs
        
    return all_models, all_inputs, all_outputs
        
    
def iterate_models():
    pass