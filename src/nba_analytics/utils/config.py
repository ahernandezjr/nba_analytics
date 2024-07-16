import os
from dynaconf import Dynaconf
import pkg_resources

# Set the SETTINGS_FILE_FOR_DYNACONF environment variable
# os.environ['SETTINGS_FILE_FOR_DYNACONF'] = 'config/settings.toml'

# Create a settings object
settings_path = pkg_resources.resource_filename('nba_analytics', 'config/settings.toml')
settings = Dynaconf(load_dotenv=True,
                    settings_files=[settings_path])

# Create an azure object
azure_path = pkg_resources.resource_filename('nba_analytics', 'config/azure.toml')
azure = Dynaconf(load_dotenv=True,
                 settings_files=[azure_path])


# Functions to set the settings config
def set_data_dir(data_dir: str):
    settings.DATA_DIR = data_dir

def set_dataset_dir(dataset_dir: str):
    settings.DATASET_DIR = dataset_dir

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
    azure.SQL_SERVER_DOMAIN = sql_server_domain

def set_sql_server_name(sql_server_name: str):
    azure.SQL_SERVER_NAME = sql_server_name

def set_sql_database_name(sql_database_name: str):
    azure.SQL_DATABASE_NAME = sql_database_name

def set_sql_username(sql_username: str):
    azure.SQL_USERNAME = sql_username

def set_sql_password(sql_password: str):
    azure.SQL_PASSWORD = sql_password
