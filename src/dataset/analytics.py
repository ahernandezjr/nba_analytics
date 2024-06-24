import os

import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

from .torch import NBAPlayerDataset
from ..machine_learning.use_models import use_model

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


# Perform analytics on the dataset

# Get shape of the dataset
def get_input_shape(index=0):
    X, y = nba_dataset[index]
    print(f"X: {X.shape}, y: {y.shape}")

    return X, y


# Get the mean and standard deviation of the dataset
def get_mean_std():
    X = np.array([nba_dataset[i][0] for i in range(len(nba_dataset))])
    y = np.array([nba_dataset[i][1] for i in range(len(nba_dataset))])

    X_mean, X_std = np.mean(X), np.std(X)
    y_mean, y_std = np.mean(y), np.std(y)

    return X_mean, X_std, y_mean, y_std


# Get the min and max values of the dataset
def get_min_max():
    X = np.array([nba_dataset[i][0] for i in range(len(nba_dataset))])
    y = np.array([nba_dataset[i][1] for i in range(len(nba_dataset))])

    X_min, X_max = np.min(X), np.max(X)
    y_min, y_max = np.min(y), np.max(y)

    return X_min, X_max, y_min, y_max


# Get the number of features in the dataset
def get_num_features():
    X, y = nba_dataset[0]
    num_features = X.shape[1]

    return num_features


# Get the number of samples in the dataset
def get_num_samples():
    num_samples = len(nba_dataset)

    return num_samples


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


# Create graphs
def create_data_graphs():
    # Get the number of samples
    num_samples = get_num_samples()

    # Get the number of features
    num_features = get_num_features()

    # Get the mean and standard deviation
    X_mean, X_std, y_mean, y_std = get_mean_std()

    # Get the min and max values
    X_min, X_max, y_min, y_max = get_min_max()

    # Create a figure
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))

    # Plot the number of samples
    axs[0, 0].bar(['Number of Samples'], [num_samples])
    axs[0, 0].set_title('Number of Samples')
    axs[0, 0].text(0, num_samples, f'{num_samples:.3g}', ha='center', va='bottom')
    

    # Plot the number of features
    axs[0, 1].bar(['Number of Features'], [num_features])
    axs[0, 1].set_title('Number of Features')
    axs[0, 1].text(0, num_features, f'{num_features:.3g}', ha='center', va='bottom')

    # Plot the mean and standard deviation
    axs[1, 0].bar(['X Mean', 'X Std', 'y Mean', 'y Std'], [X_mean, X_std, y_mean, y_std])
    axs[1, 0].set_title('Mean and Standard Deviation')
    axs[1, 0].text(0, X_mean, f'{X_mean:.3g}', ha='center', va='bottom')
    axs[1, 0].text(1, X_std, f'{X_std:.3g}', ha='center', va='bottom')
    axs[1, 0].text(2, y_mean, f'{y_mean:.3g}', ha='center', va='bottom')
    axs[1, 0].text(3, y_std, f'{y_std:.3g}', ha='center', va='bottom')

    # Plot the min and max values
    axs[1, 1].bar(['X Min', 'X Max', 'y Min', 'y Max'], [X_min, X_max, y_min, y_max])
    axs[1, 1].set_title('Min and Max Values')
    axs[1, 1].text(0, X_min, f'{X_min:.3g}', ha='center', va='bottom')
    axs[1, 1].text(1, X_max, f'{X_max:.3g}', ha='center', va='bottom')
    axs[1, 1].text(2, y_min, f'{y_min:.3g}', ha='center', va='bottom')
    axs[1, 1].text(3, y_max, f'{y_max:.3g}', ha='center', va='bottom')

    # Show the plots
    plt.show()

    # Save plots to GRAPHS_DIR
    fig.savefig(os.path.join(settings.GRAPHS_DIR, 'analytics.png'))


# Create graphs based on outputs/predictions of all models
def create_prediction_graphs():
    models, inputs, outputs = use_model(-1)


    
    # Create a figure to hold all the graphs
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))

    # For every input and output, create a graph
    for i in range(len(inputs)):
        # Calculate the difference between inputs and outputs for the feature
        diff = [inputs[i][j] - outputs[i][j] for j in range(len(inputs[i]))]

        # Plot the difference
        axs[i//2, i%2].bar(range(len(diff)), diff)
        axs[i//2, i%2].set_xlabel('Feature')
        axs[i//2, i%2].set_ylabel('Difference')
        axs[i//2, i%2].set_title(f'{models[i]}')
    
    # Adjust spacing between subplots
    plt.tight_layout()

    # Show the plot
    plt.show()

    # Save the plot to GRAPHS_DIR
    fig.savefig(os.path.join(settings.GRAPHS_DIR, f'model_predictions.png'))
