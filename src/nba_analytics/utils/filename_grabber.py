import os
from .config import settings


def get_dataset_dir():
    return settings.DATASET_DIR

def get_data_file():
    dataset_dir = get_dataset_dir()
    return os.path.join(os.getcwd(), dataset_dir, settings.DATA_FILE_NAME)

def get_data_file_5year():
    dataset_dir = get_dataset_dir()
    return os.path.join(os.getcwd(), dataset_dir, settings.DATA_FILE_5YEAR_NAME)

def get_data_file_5year_tensor():
    dataset_dir = get_dataset_dir()
    return os.path.join(os.getcwd(), dataset_dir, settings.DATA_FILE_5YEAR_TENSOR_NAME)

def get_data_file_5year_overlap():
    dataset_dir = get_dataset_dir()
    return os.path.join(os.getcwd(), dataset_dir, settings.DATA_FILE_5YEAR_OVERLAP)

def get_data_file_5year_json():
    dataset_dir = get_dataset_dir()
    return os.path.join(os.getcwd(), dataset_dir, settings.DATA_FILE_5YEAR_JSON_NAME)


def get_any_file(data_file):
    dataset_dir = get_dataset_dir()
    return os.path.join(os.getcwd(), dataset_dir, data_file)

def get_all_files():
    files = [get_data_file(),
             get_data_file_5year(),
             get_data_file_5year_tensor(),
             get_data_file_5year_overlap(),
             get_data_file_5year_json()]
    return files