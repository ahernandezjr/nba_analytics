import os
from .config import settings


# Functions for different data directories
def get_data_dir():
    return os.path.join(os.getcwd(), settings.DATA_DIR)

def get_logs_dir():
    return os.path.join(get_data_dir(), settings.LOGS_DIR)

def get_graphs_dir():
    return os.path.join(get_data_dir(), settings.GRAPHS_DIR)

def get_reports_dir():
    return os.path.join(get_data_dir(), settings.REPORTS_DIR)


# Functions for dataset medallion directories
def get_dataset_dir():
    return os.path.join(get_data_dir(), settings.DATASET_DIR)

def get_bronze_dir():
    return os.path.join(get_dataset_dir(), settings.BRONZE_DIR)

def get_silver_dir():
    return os.path.join(get_dataset_dir(), settings.SILVER_DIR)

def get_gold_dir():
    return os.path.join(get_dataset_dir(), settings.GOLD_DIR)


# Function to get specific directory
def get_specific_dir(dir_name):
    '''Returns the directory path for the given directory name.'''
    if dir_name is None:
        return get_data_dir()
    elif dir_name == 'data':
        return get_data_dir()
    elif dir_name == 'dataset':
        return get_dataset_dir()
    elif dir_name == 'bronze':
        return get_bronze_dir()
    elif dir_name == 'silver':
        return get_silver_dir()
    elif dir_name == 'gold':
        return get_gold_dir()
    else:
        raise ValueError(f"Invalid directory name: {dir_name}")


# Function to get data file
def get_data_file(dir_name=None, data_file_name=None):
    '''Returns the data file path for the given directory name and data file name.'''
    dataset_dir = get_specific_dir(dir_name)
    return os.path.join(os.getcwd(), dataset_dir, data_file_name)


# Function to get model file
def get_models_dir():
    return os.path.join(get_data_dir(), settings.MODELS_DIR)

def get_model_file(model_file_name):
    '''Returns the model file path for the given model file name.'''
    return os.path.join(get_models_dir(), model_file_name)


# def get_any_file(data_file):
#     '''Returns the file path for the given data file name.'''
#     dataset_dir = get_dataset_dir()
#     return os.path.join(os.getcwd(), dataset_dir, data_file)

# # Function to get all files
# def get_all_files():
#     '''Returns a list of all data files'''
#     files = [get_data_file(),
#              get_data_file_5year(),
#              get_data_file_5year_tensor(),
#              get_data_file_5year_overlap(),
#              get_data_file_5year_json()]
#     return files
