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

from ..dataset.dataset_torch import NBAPlayerDataset

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
DATA_FILE_5YEAR_TENSOR_NAME = settings.DATA_FILE_5YEAR_TENSOR_NAME
DATA_FILE_5YEAR_JSON_NAME = settings.DATA_FILE_5YEAR_JSON_NAME


# set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the dataset from the tensor file
df = pd.read_csv(DATA_FILE_5YEAR_TENSOR_NAME)

# Load the dictionary with proper numeric types
df_dict = pd.read_json(DATA_FILE_5YEAR_JSON_NAME, typ='series').to_dict()

# Instantiate the dataset
nba_dataset = NBAPlayerDataset(df)

# Create a training DataLoader and test DataLoader
train_loader = DataLoader(nba_dataset, batch_size=32, shuffle=True)
test_loader  = DataLoader(nba_dataset, batch_size=32, shuffle=False)

# Define hyperparameters
learning_rate = 0.001

# input_size = 39 # number of features
input_size  = nba_dataset[0][0].shape[1] # number of features
hidden_size = 39 # number of features in hidden state
output_size = nba_dataset[0][0].shape[1] # number of features
num_layers = 3 # number of stacked lstm layers


def get_model(model_name, input_size, hidden_size, output_size, num_layers):
    '''
    Get the model based on the model name.
    '''
    if model_name == 'nn_lstm':
        model = get_nn_LSTM(input_size=input_size,
                                hidden_size=hidden_size,
                                num_layers=num_layers)
        model.name = 'nn_lstm'
        return model
    
    elif model_name == 'custom_lstm':
        model = get_custom_lstm(input_size=input_size, 
                                hidden_size=hidden_size, 
                                output_size=output_size,
                                num_layers=num_layers)
        model.name = 'custom_lstm'
        return model
    
    elif model_name == 'nn':
        model = CustomNN(input_size=input_size, 
                         hidden_size=hidden_size, 
                         output_size=output_size)
        model.name = 'nn'
        return model
    
    else:
        raise ValueError("Model name not recognized.")


def training_loop(epochs, model, optimizer, loss_fn, dataloader, train=True):
    '''
    Training loop for the model.
    '''
    pbar = tqdm(range(epochs)) if train else tqdm(range(dataloader.__len__()))
    for epoch in pbar:
        for i, (inputs, targets) in enumerate(dataloader):
            # Change inputs and targets to float
            inputs = inputs.float()
            targets = targets.float()

            inputs, targets = inputs.to(device), targets.to(device)

            loss = None
            outputs = None
            if model.name == 'nn_lstm':
                outputs = model.forward(inputs)[0]
                loss = loss_fn(outputs[:, -1], targets)
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
    logger.info(f"Running {model_name} model on {device}...")

    model = get_model(model_name=model_name,
                        input_size=input_size,
                        hidden_size=hidden_size,
                        output_size=output_size,
                        num_layers=num_layers)
    
    model = model.to(device)

    # Define the loss function and the optimizer
    criterion = nn.MSELoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    # Training loop
    logger.info(f"Training model (starting loop)...")
    model.train()
    training_loop(epochs=epochs,
                  model=model,
                  optimizer=optimizer,
                  loss_fn=criterion,
                  dataloader=train_loader,
                  train=True)
    
    # Test model
    logger.info(f"Evaluating model...")
    model.eval()
    with torch.no_grad():
        training_loop(epochs=None,
                      model=model,
                      optimizer=None,
                      loss_fn=criterion,
                      dataloader=test_loader,
                      train=False)

    # Save the model
    model_name = model.name
    filename = f"{model_name}.pth"
    model_path = os.path.join(DATA_DIR, filename)
    
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

