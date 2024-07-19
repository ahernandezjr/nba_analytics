import os
from dynaconf import Dynaconf
import pkg_resources

# Set the SETTINGS_FILE_FOR_DYNACONF environment variable
# os.environ['SETTINGS_FILE_FOR_DYNACONF'] = 'config/settings.toml'

# Create a settings object
settings_path = pkg_resources.resource_filename('nba_analytics', 'config/settings.toml')
settings = Dynaconf(load_dotenv=True,
                    settings_files=[settings_path])

# Functions to set the settings config
# Set the data directories
def set_data_dir(data_dir: str):
    settings.DATA_DIR = data_dir

def set_dataset_dir(dataset_dir: str):
    settings.DATASET_DIR = dataset_dir

def set_bronze_dir(dataset_dir: str):
    settings.BRONZE_DIR = dataset_dir

def set_silver_dir(dataset_dir: str):
    settings.SILVER_DIR = dataset_dir

def set_gold_dir(dataset_dir: str):
    settings.GOLD_DIR = dataset_dir

# Set the logs, models, graphs, and reports directories
def set_logs_dir(logs_dir: str):
    settings.LOGS_DIR = logs_dir

def set_models_dir(models_dir: str):
    settings.MODELS_DIR = models_dir

def set_graphs_dir(graphs_dir: str):
    settings.GRAPHS_DIR = graphs_dir

def set_reports_dir(reports_dir: str):
    settings.REPORTS_DIR = reports_dir


# Functions to set the azure config
def set_sql_server_domain(sql_server_domain: str):
    settings.azure.SQL_SERVER_DOMAIN = sql_server_domain

def set_sql_server_name(sql_server_name: str):
    settings.azure.SQL_SERVER_NAME = sql_server_name

def set_sql_database_name(sql_database_name: str):
    settings.azure.SQL_DATABASE_NAME = sql_database_name

def set_sql_username(sql_username: str):
    settings.azure.SQL_USERNAME = sql_username

def set_sql_password(sql_password: str):
    settings.azure.SQL_PASSWORD = sql_password


# Functions to reset the settings and azure config
def reset_settings(settings):
    ...
    
def reset_azure(azure):
    ...