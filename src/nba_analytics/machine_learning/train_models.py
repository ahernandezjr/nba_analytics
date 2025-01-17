"""
This module trains and tests various neural network models on NBA player statistics data.
It includes model definition, training, evaluation, and saving functionalities.
"""

import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import mlflow
import mlflow.pytorch
from contextlib import nullcontext

# Import dataset class and model functions
from ..data.dataset.torch_overlap import NBAPlayerDataset
from .models.lstm import get_custom_lstm, get_nn_LSTM
from .models.neuralnet import CustomNN
from ..utils import filename_grabber
from ..utils.config import settings
from ..utils.logger import get_logger
from ..utils.mlflow_utils import setup_mlflow


gold = settings.dataset.gold

# Create logger
logger = get_logger(__name__)

# Configuration
FILTER_AMT = settings.dataset.FILTER_AMT
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set up MLflow
experiment = setup_mlflow()

# Load the dataset from the tensor file
df = pd.read_csv(filename_grabber.get_data_file(dir_name='gold',
                                               data_file_name=gold.DATA_FILE_CONTINUOUS_FIRST),
                 encoding='utf-8')

# Change from loadtxt to load since we're working with a .npy file
np_overlap = np.load(filename_grabber.get_data_file(dir_name='gold',
                                                   data_file_name=gold.DATA_FILE_CONTINUOUS_OVERLAP),
                    allow_pickle=True)

# Reshape the 2D numpy array to its original shape
np_overlap = np_overlap.reshape(np_overlap.shape[0], FILTER_AMT, -1)

# Load the dictionary with proper numeric types
df_dict = pd.read_json(filename_grabber.get_data_file(dir_name='gold',
                                                      data_file_name=gold.DATA_FILE_CONTINUOUS_FIRST_JSON),
                       typ='series').to_dict()

# Instantiate dataset and DataLoader
nba_dataset = NBAPlayerDataset(np_overlap)
train_loader = DataLoader(nba_dataset, batch_size=32, shuffle=True)
test_loader  = DataLoader(nba_dataset, batch_size=32, shuffle=False)

# Define hyperparameters
learning_rate = 0.01

def get_model(model_name, input_size, hidden_size, output_size, num_layers):
    """
    Retrieves the model based on the provided model name.

    Args:
        model_name (str): Name of the model to retrieve.
        input_size (int): Number of input features.
        hidden_size (int): Number of hidden units in the hidden layer.
        output_size (int): Number of output features.
        num_layers (int): Number of layers in the model.

    Returns:
        nn.Module: The requested model.
    """
    if model_name == 'nn_lstm':
        model = get_nn_LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)
        model.name = model_name
        return model
    
    elif model_name == 'custom_lstm':
        model = get_custom_lstm(input_size=input_size, hidden_size=hidden_size, output_size=output_size, num_layers=num_layers)
        model.name = model_name
        return model
    
    elif model_name in ['nn_one_to_one', 'nn_many_to_one']:
        model = CustomNN(input_size=input_size, hidden_size=hidden_size, output_size=output_size)
        model.name = model_name
        return model
    
    else:
        raise ValueError("Model name not recognized.")

def train_test_loop(epochs, model, optimizer, loss_fn, dataloader, train=True):
    """
    Runs the training or testing loop for the model.

    Args:
        epochs (int): Number of epochs for training.
        model (nn.Module): The model to train or test.
        optimizer (torch.optim.Optimizer): Optimizer for training.
        loss_fn (nn.Module): Loss function to use.
        dataloader (DataLoader): DataLoader for training or testing data.
        train (bool): If True, runs the training loop; if False, runs the testing loop.
    """
    total_loss = 0
    num_batches = 0
    
    pbar = tqdm(range(epochs))
    for epoch in pbar:
        batch_losses = []
        for i, (inputs, targets) in enumerate(dataloader):
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
                loss = loss_fn(outputs, targets)
            else:
                outputs = model.forward(inputs)
                loss = loss_fn(outputs, inputs)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Log metrics for training
                batch_losses.append(loss.item())
                if (i + 1) % 10 == 0:  # Log every 10 batches
                    mlflow.log_metric("train_batch_loss", np.mean(batch_losses), step=epoch * len(dataloader) + i)
                    batch_losses = []
            
            total_loss += loss.item()
            num_batches += 1
            
            desc = f"Epoch: {epoch+1}/{epochs}, Batch: {i+1}/{len(dataloader)}" if train else f"Batch: {i+1}/{len(dataloader)}"
            pbar.set_description(desc)
            pbar.set_postfix({"Loss": loss.item()})
    
    avg_loss = total_loss / num_batches
    return avg_loss

def run_multiple_models(model_names, epochs=100):
    """
    Runs multiple models under a single MLflow parent run.
    
    Args:
        model_names (list): List of model names to run
        epochs (int): Number of epochs for training
    
    Returns:
        dict: Dictionary of trained models
    """
    models = {}
    
    # End any existing runs to ensure clean state
    if mlflow.active_run():
        mlflow.end_run()
    
    # Start a parent run under the experiment
    with mlflow.start_run(run_name="NBA Models Comparison", experiment_id=experiment.experiment_id) as parent_run:
        # Log parent run parameters
        mlflow.log_params({
            "models": model_names,
            "epochs": epochs,
            "device": str(device),
            "learning_rate": learning_rate
        })
        
        # Run each model as a child run
        for model_name in model_names:
            logger.info(f"Starting training for {model_name}...")
            with mlflow.start_run(run_name=model_name, nested=True) as child_run:
                models[model_name] = run_model(model_name, epochs, log_to_mlflow=True)
                logger.info(f"Finished training {model_name}. Run ID: {child_run.info.run_id}")
        
        logger.info(f"All models trained. Parent run ID: {parent_run.info.run_id}")
        return models

def run_model(model_name, epochs=1000, log_to_mlflow=False):
    """
    Runs the specified model on the data, including training, testing, and saving.

    Args:
        model_name (str): Name of the model to run.
        epochs (int): Number of epochs for training.
        log_to_mlflow (bool): Whether this is part of a nested run

    Returns:
        nn.Module: The trained model.
    """
    input_size = nba_dataset[0][0].shape[1]
    hidden_size = nba_dataset[0][0].shape[1]
    output_size = nba_dataset[0][0].shape[1]
    num_layers = 5

    if model_name == 'nn_many_to_one':
        input_size = nba_dataset[0][0][:-1].flatten().shape[0]
    
    # If we're already in a nested run (log_to_mlflow=True), don't create a new run
    run_context = nullcontext() if log_to_mlflow else mlflow.start_run(experiment_id=experiment.experiment_id)
    
    with run_context:
        # Log parameters
        params = {
            "model_name": model_name,
            "epochs": epochs,
            "input_size": input_size,
            "hidden_size": hidden_size,
            "output_size": output_size,
            "num_layers": num_layers,
            "learning_rate": learning_rate,
            "device": str(device)
        }
        mlflow.log_params(params)
        
        logger.info(f"Running {model_name} model on {device}...")
        logger.info(f"Hyperparameters: {params}")

        # Get the model
        model = get_model(model_name=model_name, input_size=input_size, hidden_size=hidden_size, 
                         output_size=output_size, num_layers=num_layers)
        model = model.to(device)

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        # Training loop
        logger.info(f"Training model (starting loop)...")
        model.train()
        train_loss = train_test_loop(epochs=epochs, model=model, optimizer=optimizer, 
                                   loss_fn=criterion, dataloader=train_loader, train=True)
        mlflow.log_metric("final_train_loss", train_loss)
        
        # Test model
        logger.info(f"Evaluating model...")
        model.eval()
        with torch.no_grad():
            test_loss = train_test_loop(epochs=epochs, model=model, optimizer=None, 
                                      loss_fn=criterion, dataloader=test_loader, train=False)
            mlflow.log_metric("test_loss", test_loss)

        # Save the model
        filename = f"{model_name}.pth"
        model_path = filename_grabber.get_model_file(filename)
        logger.info(f"Saving model at {model_path}.")
        torch.save(model.state_dict(), model_path)
        
        # Log the model and artifacts to MLflow
        mlflow.pytorch.log_model(
            model, 
            "model",
            registered_model_name=model_name
        )
        mlflow.log_artifact(model_path, "saved_model")
        
        # Log additional metadata
        mlflow.log_param("model_save_path", model_path)
        mlflow.log_param("run_id", mlflow.active_run().info.run_id)
        
        if not log_to_mlflow:
            logger.info(f"Model saved and logged to MLflow. Run ID: {mlflow.active_run().info.run_id}")
    
    return model
