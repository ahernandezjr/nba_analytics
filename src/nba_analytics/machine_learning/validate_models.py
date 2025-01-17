"""
This module validates trained models using MLflow's validation utilities.
It ensures models are properly saved and can be served correctly.
"""

import os
import numpy as np
import torch
from torch.utils.data import DataLoader
import mlflow
from mlflow.models import validate_serving_input, convert_input_example_to_serving_input

from ..data.dataset.torch_overlap import NBAPlayerDataset, create_dataset
from ..utils import filename_grabber
from ..utils.config import settings
from ..utils.logger import get_logger
from ..utils.mlflow_utils import setup_mlflow
from .train_models import get_model


# Create logger
logger = get_logger(__name__)

# Configuration
FILTER_AMT = settings.dataset.FILTER_AMT
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set up MLflow
experiment = setup_mlflow()
EXPERIMENT_NAME = settings.mlflow.EXPERIMENT_NAME

def get_input_example(model_name, dataset):
    """
    Creates an input example for model validation based on the model type.
    
    Args:
        model_name (str): Name of the model to create input for
        dataset (NBAPlayerDataset): The dataset to get example from
    
    Returns:
        dict: Input example formatted for MLflow serving
    """
    inputs, _ = dataset[0]
    inputs = inputs.float()

    if model_name == 'nn_lstm':
        # LSTM expects sequence data
        return {"inputs": inputs.numpy().tolist()}
    elif model_name == 'nn_one_to_one':
        # One-to-one takes single timestep
        return {"inputs": inputs[0].numpy().tolist()}
    elif model_name == 'nn_many_to_one':
        # Many-to-one takes flattened sequence except last timestep
        flattened = inputs[:-1].view(-1).numpy()
        return {"inputs": flattened.tolist()}
    else:
        return {"inputs": inputs.numpy().tolist()}

def validate_model(run_id, model_name):
    """
    Validates a trained model using MLflow's validation utilities.
    
    Args:
        run_id (str): MLflow run ID of the model to validate
        model_name (str): Name of the model to validate
    
    Returns:
        bool: True if validation successful, False otherwise
    """
    logger.info(f"Validating model {model_name} from run {run_id}...")
    
    try:
        # Construct model URI
        model_uri = f'runs:/{run_id}/model'
        
        # Load dataset
        dataset = create_dataset()
        
        # Get input example for this model type
        input_example = get_input_example(model_name, dataset)
        
        # Convert to serving input format
        serving_payload = convert_input_example_to_serving_input(input_example)
        
        # Validate the serving payload works on the model
        validate_serving_input(model_uri, serving_payload)
        
        logger.info(f"Model {model_name} (run_id: {run_id}) validated successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Error validating model {model_name} (run_id: {run_id}): {str(e)}")
        return False

def validate_all_models_in_experiment(experiment_name=EXPERIMENT_NAME):
    """
    Validates all models in the specified MLflow experiment.
    
    Args:
        experiment_name (str): Name of the MLflow experiment to validate models from
    
    Returns:
        dict: Dictionary of validation results by model name
    """
    # Get experiment
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        logger.error(f"Experiment {experiment_name} not found!")
        return {}
    
    # Get all runs in the experiment
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
    
    validation_results = {}
    
    for _, run in runs.iterrows():
        # Skip parent runs (runs that have nested runs)
        if 'models' in run.params:
            continue
            
        run_id = run.run_id
        model_name = run.params.get('model_name', 'unknown')
        
        # Validate the model
        is_valid = validate_model(run_id, model_name)
        validation_results[model_name] = {
            'run_id': run_id,
            'is_valid': is_valid
        }
    
    # Log summary
    logger.info("\nValidation Summary:")
    for model_name, result in validation_results.items():
        status = "✓ VALID" if result['is_valid'] else "✗ INVALID"
        logger.info(f"{model_name}: {status}")
    
    return validation_results

def validate_latest_models():
    """
    Validates only the most recent version of each model type.
    
    Returns:
        dict: Dictionary of validation results for latest models
    """
    # Get experiment
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        logger.error(f"Experiment {EXPERIMENT_NAME} not found!")
        return {}
    
    # Get all runs
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
    
    # Filter for latest version of each model
    latest_runs = {}
    for _, run in runs.iterrows():
        # Skip parent runs
        if 'models' in run.params:
            continue
            
        model_name = run.params.get('model_name', 'unknown')
        start_time = run.start_time
        
        if model_name not in latest_runs or start_time > latest_runs[model_name]['start_time']:
            latest_runs[model_name] = {
                'run_id': run.run_id,
                'start_time': start_time
            }
    
    # Validate latest versions
    validation_results = {}
    for model_name, run_info in latest_runs.items():
        is_valid = validate_model(run_info['run_id'], model_name)
        validation_results[model_name] = {
            'run_id': run_info['run_id'],
            'is_valid': is_valid
        }
    
    # Log summary
    logger.info("\nLatest Models Validation Summary:")
    for model_name, result in validation_results.items():
        status = "✓ VALID" if result['is_valid'] else "✗ INVALID"
        logger.info(f"{model_name}: {status}")
    
    return validation_results

if __name__ == '__main__':
    # Validate all models in the experiment
    validate_all_models_in_experiment()
    
    # Or validate only latest versions
    # validate_latest_models()
