import os

import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

from .torch import NBAPlayerDataset
from ..machine_learning.use_models import use_model

from .analytics import get_num_samples, get_num_features, get_mean_std, get_min_max

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

# Create a DataLoader
data_loader = DataLoader(nba_dataset, batch_size=32, shuffle=False)




# Create a bar graph showing feature importance
def create_feature_importance_graph():
    # https://machinelearningmastery.com/calculate-feature-importance-with-python/
    # https://machinelearningmastery.com/feature-selection-machine-learning-python/
    pass





# Create 2-D PCA plot
def create_pca_plot():
    from sklearn.decomposition import PCA

    # Get the number of samples
    num_samples = get_num_samples()

    # Get the number of features
    num_features = get_num_features()

    # Get the mean and standard deviation
    X_mean, X_std, y_mean, y_std = get_mean_std()

    # Get the min and max values
    X_min, X_max, y_min, y_max = get_min_max()

    # Create a PCA object
    pca = PCA(n_components=2)

    # Get the X and y values
    X = np.array([nba_dataset[i][0] for i in range(len(nba_dataset))])

    X = X[:, 0]

    # Get size of a sample of X
    size = X[0].size

    # Reshape the X values to 2D array
    X = np.reshape(X, (len(nba_dataset), size) )

    # Fit the PCA object
    X_pca = pca.fit_transform(X)

    # Create a figure
    fig, ax = plt.subplots(figsize=(10, 10))

    # Plot the PCA values
    ax.scatter(X_pca[:, 0], X_pca[:, 1])
    ax.set_title('PCA Plot')

    # Show the plot
    plt.show()

    # Save the plot to GRAPHS_DIR
    fig.savefig(os.path.join(settings.GRAPHS_DIR, 'pca.png'))


def generate_matplotlib_stackbars(df, filename):
    
    # Create subplot and bar
    fig, ax = plt.subplots()
    
    # Get the minimum and maximum age of players
    min_age = np.min(df['Age'])
    max_age = np.max(df['Age'])
    
    # Filter the dataset based on age
    filtered_df = df[(df['Age'] >= min_age) & (df['Age'] <= max_age)]
    
    # Calculate the average stats
    avg_stats = filtered_df.mean()
    
    # Plot the average stats
    ax.bar(avg_stats.index, avg_stats.values, color="#E63946")
    
    # Set Title
    ax.set_title('Average Stats by Age', fontweight="bold")

    # Set xticklabels
    ax.set_xticklabels(avg_stats.index, rotation=90)
    plt.xticks(range(len(avg_stats.index)))

    # Set ylabel
    ax.set_ylabel('Average Stats') 

    # Save the plot as a PNG
    plt.savefig(filename, dpi=300, bbox_inches='tight', pad_inches=0)
    
    plt.show()
