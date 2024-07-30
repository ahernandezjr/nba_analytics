"""
This module provides implementations for LSTM (Long Short-Term Memory) models tailored for NBA player statistics data.
It includes a custom LSTM class and functions to create custom and standard LSTM models.
"""

import torch
import torch.nn as nn
from torch.autograd import Variable

from ...utils.config import settings
from ...utils.logger import get_logger

# Create logger
logger = get_logger(__name__)

class CustomLSTM(nn.Module):
    """
    Custom LSTM (Long Short-Term Memory) class for the NBA player statistics dataset.
    
    This class defines an LSTM model with a specified number of input features, hidden units, 
    output features, and layers. It includes fully connected layers and ReLU activation functions.
    
    Attributes:
        num_classes (int): Number of output features.
        num_layers (int): Number of LSTM layers.
        input_size (int): Number of input features.
        hidden_size (int): Number of hidden units in the LSTM layers.
        lstm (nn.LSTM): LSTM layer.
        fc_1 (nn.Linear): First fully connected layer.
        fc_2 (nn.Linear): Second fully connected layer.
        relu (nn.ReLU): ReLU activation function.
    """
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        """
        Initializes the CustomLSTM class with the given parameters.
        
        Args:
            input_size (int): Number of input features.
            hidden_size (int): Number of hidden units in the LSTM layers.
            output_size (int): Number of output features.
            num_layers (int): Number of LSTM layers.
        """
        super().__init__()
        self.num_classes = output_size
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        # LSTM model
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True, dropout=0.2)
        self.fc_1 = nn.Linear(hidden_size, 128)
        self.fc_2 = nn.Linear(128, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        Defines the forward pass of the model.
        
        Args:
            x (torch.Tensor): Input tensor.
        
        Returns:
            torch.Tensor: Output tensor after passing through the LSTM and fully connected layers.
        """
        # hidden state
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))
        # cell state
        c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))
        # propagate input through LSTM
        output, (hn, cn) = self.lstm(x, (h_0, c_0))
        hn = hn.view(-1, self.hidden_size)
        out = self.relu(hn)
        out = self.fc_1(out)
        out = self.relu(out)
        out = self.fc_2(out)
        return out

def get_custom_lstm(input_size, hidden_size, output_size, num_layers):
    """
    Creates and returns an instance of the CustomLSTM model.
    
    Args:
        input_size (int): Number of input features.
        hidden_size (int): Number of hidden units in the LSTM layers.
        output_size (int): Number of output features.
        num_layers (int): Number of LSTM layers.
    
    Returns:
        CustomLSTM: An instance of the CustomLSTM model.
    """
    return CustomLSTM(input_size, hidden_size, output_size, num_layers)

def get_nn_LSTM(input_size, hidden_size, num_layers):
    """
    Creates and returns an instance of the standard PyTorch LSTM model.
    
    Args:
        input_size (int): Number of input features.
        hidden_size (int): Number of hidden units in the LSTM layers.
        num_layers (int): Number of LSTM layers.
    
    Returns:
        nn.LSTM: An instance of the standard PyTorch LSTM model.
    """
    return nn.LSTM(
        input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
        batch_first=True
    )
