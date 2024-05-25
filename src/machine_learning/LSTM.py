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

from ..data_process.dataset_torch import NBAPlayerDataset

from ..utils.config import settings
from ..utils.logger import getLogger


# Create logger
logger = getLogger(__name__)


# Set configs from settings
DATA_DIR = settings.DATA_DIR
DATA_FILE_NAME = settings.DATA_FILE_NAME
DATA_FILE_5YEAR_NAME = settings.DATA_FILE_5YEAR_NAME
DATA_FILE_5YEAR_JSON_NAME = settings.DATA_FILE_5YEAR_JSON_NAME


class LSTM(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers):
        super().__init__()
        self.num_classes = num_classes # output size
        self.num_layers = num_layers # number of recurrent layers in the lstm
        self.input_size = input_size # input size
        self.hidden_size = hidden_size # neurons in each lstm layer
        # LSTM model
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True, dropout=0.2) # lstm
        self.fc_1 =  nn.Linear(hidden_size, 128) # fully connected 
        self.fc_2 = nn.Linear(128, num_classes) # fully connected last layer
        self.relu = nn.ReLU()
        
    def forward(self,x):
        # hidden state
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))
        # cell state
        c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))
        # propagate input through LSTM
        output, (hn, cn) = self.lstm(x, (h_0, c_0)) # (input, hidden, and internal state)
        hn = hn.view(-1, self.hidden_size) # reshaping the data for Dense layer next
        out = self.relu(hn)
        out = self.fc_1(out) # first dense
        out = self.relu(out) # relu
        out = self.fc_2(out) # final output
        return out


def training_loop(epochs, lstm, optimizer, loss_fn, dataloader):
    for epoch in tqdm(range(epochs)):
        for i, (inputs, targets) in enumerate(dataloader):
            lstm.train()
            outputs = lstm.forward(inputs) # forward pass
            optimizer.zero_grad() # calculate the gradient, manually setting to 0
            # obtain the loss function
            loss = loss_fn(outputs, targets)
            loss.backward() # calculates the loss of the loss function
            optimizer.step() # improve from loss, i.e backprop
            # test loss
            lstm.eval()
    

# def test_loop(lstm, optimizer, loss_fn, test_loader):

def run_lstm(epochs=1000):
    '''
    Run the LSTM model.
    '''
    # Load the dictionary with proper numeric types
    df_dict = pd.read_json(DATA_FILE_5YEAR_JSON_NAME, typ='series').to_dict()

    # Instantiate the dataset
    nba_dataset = NBAPlayerDataset(df_dict)

    # Create a training DataLoader and test DataLoader
    train_loader = DataLoader(nba_dataset, batch_size=32, shuffle=True)
    test_loader  = DataLoader(nba_dataset, batch_size=32, shuffle=False)

    

    # # Check the first item in the DataLoader
    # for i, (inputs, targets) in enumerate(dataloader):
    #     print(inputs)
    #     print(targets)
    #     break

    # Define hyperparameters
    learning_rate = 0.001 # 0.001 lr

    input_size = 39 # number of features
    hidden_size = 39 # number of features in hidden state
    num_layers = 5 # number of stacked lstm layers

    num_classes = 39 # number of output classes 

    # Create the LSTM model
    model = LSTM(num_classes=num_classes, 
                 input_size=input_size, 
                 hidden_size=hidden_size, 
                 num_layers=num_layers)

    # Define the loss function and the optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    training_loop(n_epochs=epochs,
                lstm=model,
                optimizer=optimizer,
                loss_fn=criterion,
                dataloader=train_loader)


# TO DO: Implement to predict every year for first 5 years
def arima():
    '''
    Run the ARIMA model.
    '''
    pass
