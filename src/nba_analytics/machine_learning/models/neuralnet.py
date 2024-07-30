"""
This module provides a custom neural network implementation for the NBA player statistics dataset.
It includes a custom fully connected neural network class with ReLU activation functions.
"""

import torch.nn as nn
from ...utils.logger import get_logger

# Create logger
logger = get_logger(__name__)

class CustomNN(nn.Module):
    """
    Custom Neural Network class for the NBA player statistics dataset.
    
    This class defines a fully connected neural network with a specified number of input features,
    hidden units, and output features. It includes ReLU activation functions.

    Attributes:
        input_size (int): Number of input features.
        hidden_size (int): Number of hidden units in the hidden layer.
        output_size (int): Number of output features.
        fc1 (nn.Linear): First fully connected layer.
        relu (nn.ReLU): ReLU activation function.
        fc2 (nn.Linear): Second fully connected layer.
    """
    def __init__(self, input_size, hidden_size, output_size):
        """
        Initializes the CustomNN class with the given parameters.
        
        Args:
            input_size (int): Number of input features.
            hidden_size (int): Number of hidden units in the hidden layer.
            output_size (int): Number of output features.
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """
        Defines the forward pass of the model.
        
        Args:
            x (torch.Tensor): Input tensor.
        
        Returns:
            torch.Tensor: Output tensor after passing through the fully connected layers and ReLU activations.
        """
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out
