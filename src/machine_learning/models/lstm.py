import torch
import torch.nn as nn
from torch.autograd import Variable

from ...utils.config import settings
from ...utils.logger import get_logger


# Create logger
logger = get_logger(__name__)


class CustomLSTM(nn.Module):
    '''
    Custom LSTM class for the NBA player statistics dataset.
    '''
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super().__init__()
        self.num_classes = output_size
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        # LSTM model
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True, dropout=0.2) # lstm
        self.fc_1 =  nn.Linear(hidden_size, 128)
        self.fc_2 = nn.Linear(128, output_size)
        self.relu = nn.ReLU()

    def forward(self,x):
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
    '''
    Get a custom LSTM model.
    '''
    return CustomLSTM(input_size, hidden_size, output_size, num_layers)


def get_nn_LSTM(input_size, hidden_size, num_layers):
    '''
    Use LSTM model from PyTorch.
    '''
    return nn.LSTM( 
        input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
        batch_first=True
    )
