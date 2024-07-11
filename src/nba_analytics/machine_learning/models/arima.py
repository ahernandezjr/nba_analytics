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

from ...dataset.torch import NBAPlayerDataset

from ...utils.config import settings
from ...utils.logger import get_logger


# Create logger
logger = get_logger(__name__)


# TO DO: Implement to predict every year for first 5 years
def arima():
    '''
    Run the ARIMA model.
    '''
    pass
