import os

import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

from ..dataset.torch_overlap import NBAPlayerDataset
from ..machine_learning.use_models import use_model

from .analytics import get_num_samples, get_num_features, get_mean_std, get_min_max

from ..utils.config import settings
from ..utils.logger import get_logger


# Create logger
logger = get_logger(__name__)


# Set configs from settings
DATASET_DIR = settings.DATASET_DIR
DATA_FILE_NAME = settings.DATA_FILE_NAME
DATA_FILE_5YEAR_NAME = settings.DATA_FILE_5YEAR_NAME
DATA_FILE_5YEAR_TENSOR_NAME = settings.DATA_FILE_5YEAR_TENSOR_NAME
DATA_FILE_5YEAR_OVERLAP = settings.DATA_FILE_5YEAR_OVERLAP
DATA_FILE_5YEAR_JSON_NAME = settings.DATA_FILE_5YEAR_JSON_NAME


FILTER_AMT = settings.FILTER_AMT

# set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the dataset from the tensor file
df = pd.read_csv(os.path.join(DATASET_DIR, DATA_FILE_5YEAR_TENSOR_NAME))

# Load the numpy array with proper numeric types
np_overlap = np.loadtxt(os.path.join(DATASET_DIR, DATA_FILE_5YEAR_OVERLAP), delimiter=",")

# Reshape the 2D numpy array to its original shape
np_overlap = np_overlap.reshape(np_overlap.shape[0], FILTER_AMT, -1)

# Load the dictionary with proper numeric types
df_dict = pd.read_json(os.path.join(DATASET_DIR, DATA_FILE_5YEAR_JSON_NAME), typ='series').to_dict()

# Instantiate the dataset
nba_dataset = NBAPlayerDataset(np_overlap)

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


# Create graphs
def create_data_graphs():
    '''
    Create a grouped bar graph showing the number of samples, number of features, mean and standard deviation, and min and max values of the dataset.
    '''
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
    '''
    Create a grouped bar graph showing the difference between the inputs and outputs of each model.
    
    The x-axis represents the features of the input and output data.
    The y-axis represents the difference between the input and output data.
    '''
    models, inputs, outputs = use_model(-1)

    # Create a figure to hold the graph
    fig, ax = plt.subplots(figsize=(10, 10))

    # Set the width of each bar
    bar_width = 0.2

    # Set the positions of the bars on the x-axis
    positions = np.arange(len(inputs[0][0]))

    # For every input and output, create a grouped bar graph
    for i in range(len(inputs)):
        # Convert inputs and outputs to numpy arrays
        input_array = np.array(inputs[i])
        output_array = np.array(outputs[i])

        # Calculate the average difference between inputs and outputs for the feature of each sample
        diff = np.mean(np.abs(input_array - output_array), axis=0)

        # Set the position of the bars for each model
        bar_positions = positions + (i * bar_width)

        # Plot the difference as a grouped bar graph
        ax.bar(bar_positions, diff, bar_width, label=models[i])

    # Set the x-axis labels
    ax.set_xticks(positions + (bar_width * (len(inputs) - 1)) / 2)
    ax.set_xticklabels(range(len(inputs[0][0])))

    # Set the y-axis label
    ax.set_ylabel('Difference')

    # Set the title
    ax.set_title('Model Predictions')

    # Add a legend
    ax.legend()

    # Show the plot
    plt.show()

    # Save the plot to GRAPHS_DIR
    fig.savefig(os.path.join(settings.GRAPHS_DIR, 'model_predictions.png'))
