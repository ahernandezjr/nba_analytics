import os

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# from ..dataset.torch import NBAPlayerDataset
from ..dataset.torch_overlap import NBAPlayerDataset

from .models.lstm import get_custom_lstm, get_nn_LSTM
from .models.neuralnet import CustomNN

from ..utils import filename_grabber
from ..utils.config import settings
from ..utils.logger import get_logger


# Create logger
logger = get_logger(__name__)


FILTER_AMT = settings.FILTER_AMT

# set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the dataset from the tensor file
df = pd.read_csv(filename_grabber.get_data_file(dir_name='gold',
                                                data_file_name=settings.DATA_FILE_5YEAR_TENSOR_NAME))

# Load the numpy array with proper numeric types
np_overlap = np.loadtxt(filename_grabber.get_data_file(dir_name='gold',
                                                       data_file_name=settings.DATA_FILE_5YEAR_OVERLAP),
                        delimiter=",")

# Reshape the 2D numpy array to its original shape
np_overlap = np_overlap.reshape(np_overlap.shape[0], FILTER_AMT, -1)

# Load the dictionary with proper numeric types
df_dict = pd.read_json(filename_grabber.get_data_file(dir_name='gold',
                                                      data_file_name=settings.DATA_FILE_5YEAR_JSON_NAME),
                       typ='series').to_dict()

# Instantiate the dataset
nba_dataset = NBAPlayerDataset(np_overlap)

# Create a training DataLoader and test DataLoader
train_loader = DataLoader(nba_dataset, batch_size=32, shuffle=True)
test_loader  = DataLoader(nba_dataset, batch_size=32, shuffle=False)

# Define hyperparameters
learning_rate = 0.01

# input_size = 39 # number of features


def get_model(model_name, input_size, hidden_size, output_size, num_layers):
    '''
    Get the model based on the model name.
    '''
    if model_name == 'nn_lstm':
        model = get_nn_LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers)
        model.name = model_name
        return model
    
    elif model_name == 'custom_lstm':
        model = get_custom_lstm(input_size=input_size, 
                                hidden_size=hidden_size, 
                                output_size=output_size,
                                num_layers=num_layers)
        model.name = model_name
        return model
    
    elif model_name == 'nn_one_to_one' or model_name == 'nn_many_to_one':
        model = CustomNN(input_size=input_size, 
                         hidden_size=hidden_size, 
                         output_size=output_size)
        model.name = model_name
        return model
    
    else:
        raise ValueError("Model name not recognized.")


def train_test_loop(epochs, model, optimizer, loss_fn, dataloader, train=True):
    '''
    Training loop for the model.
    '''
    pbar = tqdm(range(epochs)) if train else tqdm(range(dataloader.__len__()))
    for epoch in pbar:
        for i, (inputs, targets) in enumerate(dataloader):
            # Change inputs and targets to float and move to device
            inputs, targets = inputs.float().to(device), targets.float().to(device)

            loss = None
            outputs = None
            if model.name == 'nn_lstm':
                outputs = model.forward(inputs)[0]
                loss = loss_fn(outputs, inputs)
            elif model.name == 'nn_one_to_one':
                inputs = inputs[:, 0]
                outputs = model.forward(inputs)
                loss = loss_fn(outputs, targets)
            elif model.name == 'nn_many_to_one':
                inputs = inputs[:, :-1].view(inputs.shape[0], -1)
                outputs = model.forward(inputs)
                outputs = outputs
                loss = loss_fn(outputs, targets)
            else:
                outputs = model.forward(inputs)
                loss = loss_fn(outputs, inputs)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
                pbar.set_description(f"Epoch: {epoch+1}/{epochs}, Batch: {i+1}/{len(dataloader)}")
                pbar.set_postfix({"Loss": loss.item()})
            else:
                pbar.set_description(f"Batch: {i+1}/{len(dataloader)}")
                pbar.set_postfix({"Loss": loss.item()})


def run_model(model_name, epochs=1000):
    '''
    Run the LSTM model.
    '''
    input_size  = nba_dataset[0][0].shape[1] # number of features
    hidden_size = nba_dataset[0][0].shape[1] # number of features in hidden state
    output_size = nba_dataset[0][0].shape[1] # number of features
    num_layers = 5 # number of stacked lstm layers

    if model_name == 'nn_many_to_one':
        input_size = nba_dataset[0][0][:-1].flatten().shape[0]
    
    logger.info(f"Running {model_name} model on {device}...")
    logger.info(f"Hyperparameters: input_size={input_size}, hidden_size={hidden_size}, output_size={output_size}, num_layers={num_layers}.\n\
                Using learning rate of: {learning_rate}.\n\
                Using device: {device}.\n\
                Training DataLoader length: {len(train_loader)}\n\
                Test DataLoader length: {len(test_loader)}")

    # Get the model
    model = get_model(model_name=model_name,
                        input_size=input_size,
                        hidden_size=hidden_size,
                        output_size=output_size,
                        num_layers=num_layers)
    
    model = model.to(device)

    # Define the loss function and the optimizer
    criterion = nn.MSELoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    logger.info(f"Training model (starting loop)...")
    model.train()
    train_test_loop(epochs=epochs,
                    model=model,
                    optimizer=optimizer,
                    loss_fn=criterion,
                    dataloader=train_loader,
                    train=True)
    
    # Test model
    logger.info(f"Evaluating model...")
    model.eval()
    with torch.no_grad():
        train_test_loop(epochs=None,
                        model=model,
                        optimizer=None,
                        loss_fn=criterion,
                        dataloader=test_loader,
                        train=False)

    # Save the model
    filename = f"{model_name}.pth"
    model_path = filename_grabber.get_model_file(filename)
    
    logger.info(f"Saving model at {model_path}.")
    torch.save(model.state_dict(), model_path)
    logger.info(f"Confirmation: Model saved at {model_path}.")
    
    return model


# Old code:
# def test_model(model, dataloader, loss_fn):
#     pbar = tqdm(range(dataloader.__len__()))
#     for epoch in pbar:
#         for i, (inputs, targets) in enumerate(dataloader):
#             inputs, targets = inputs.to(device), targets.to(device)

#             loss = None
#             outputs = None
#             if model.name == 'nn_lstm':
#                 outputs = model.forward(inputs)[0]
#                 loss = loss_fn(outputs[:, -1], targets)
#             else:
#                 outputs = model.forward(inputs) # forward pass, takes 0 index to ignore weights
#                 loss = loss_fn(outputs, inputs)

