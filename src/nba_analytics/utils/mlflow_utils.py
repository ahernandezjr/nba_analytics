"""
Utility functions for MLflow setup and configuration.
"""

import os
import platform
import mlflow
from pathlib import Path
from .config import settings
from .logger import get_logger

logger = get_logger(__name__)

def _normalize_windows_path(path):
    """
    Normalize Windows path for MLflow.
    
    Args:
        path (str): Windows path to normalize
        
    Returns:
        str: Normalized path for MLflow
    """
    # Convert to Path object and resolve
    path = Path(path).resolve()
    # Convert to string and replace backslashes
    path_str = str(path).replace('\\', '/')
    # Handle Windows drive letter
    if ':' in path_str:
        path_str = path_str.replace(':', '')
    return path_str

def get_project_root():
    """
    Get the project root directory (nba_project).
    Assumes the current file is in nba_analytics/src/nba_analytics/utils/
    
    Returns:
        Path: Path to the project root directory
    """
    current_file = Path(__file__).resolve()
    # Go up 5 levels: utils -> nba_analytics -> src -> nba_analytics -> nba_project
    project_root = current_file.parents[4]
    return project_root

def setup_mlflow():
    """
    Sets up MLflow tracking URI and creates/gets the experiment.
    Uses the project root directory for storing MLflow data.
    
    Returns:
        mlflow.entities.Experiment: The MLflow experiment
    """
    try:
        # Get project root directory
        project_root = get_project_root()
        mlruns_dir = project_root / "mlruns"
        
        # Ensure directory exists
        mlruns_dir.mkdir(exist_ok=True)
        
        # Set up tracking URI based on OS
        if platform.system() == 'Windows':
            tracking_uri = f"file:///{_normalize_windows_path(mlruns_dir)}"
        else:
            tracking_uri = f"file://{str(mlruns_dir)}"
        
        # Set up MLflow
        mlflow.set_tracking_uri(tracking_uri)
        logger.info(f"MLflow tracking URI: {tracking_uri}")
        
        # Create or get the experiment
        experiment = mlflow.get_experiment_by_name(settings.mlflow.EXPERIMENT_NAME)
        if experiment is None:
            # Create new experiment without specifying artifact location
            experiment_id = mlflow.create_experiment(
                settings.mlflow.EXPERIMENT_NAME,
                tags={"project": "nba_analytics"}
            )
            experiment = mlflow.get_experiment(experiment_id)
            logger.info(f"Created new experiment: {settings.mlflow.EXPERIMENT_NAME}")
        else:
            logger.info(f"Using existing experiment: {settings.mlflow.EXPERIMENT_NAME}")
        
        return experiment
        
    except Exception as e:
        logger.error(f"Error setting up MLflow: {str(e)}")
        raise 