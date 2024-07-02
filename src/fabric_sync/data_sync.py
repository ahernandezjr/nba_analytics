'''
Might not need this file
'''

import os
from azure.storage.filedatalake import (
    DataLakeServiceClient,
    DataLakeDirectoryClient,
    FileSystemClient
)

from azure.identity import DefaultAzureCredential

from ..utils.config import settings, azure
from ..utils.logger import get_logger


# Create logger
logger = get_logger(__name__)

# Replace with your own values
DATASET_DIR = settings.DATASET_DIR
DATALAKE_ACCOUNT_NAME = azure.DATALAKE_ACCOUNT_NAME
DATALAKE_ACCOUNT_KEY = azure.DATALAKE_ACCOUNT_KEY
DATALAKE_ACCOUNT_URI = azure.DATALAKE_ACCOUNT_URI
DATALAKE_FILESYSTEM_NAME = azure.DATALAKE_FILESYSTEM_NAME
DATALAKE_DIRECTORY_PATH = azure.DATALAKE_DIRECTORY_PATH


def get_service_client_token_credential(account_url) -> DataLakeServiceClient:
    logger.info(f"Getting service client token credential with Account URL: {account_url}")

    token_credential = DefaultAzureCredential()
    logger.info(f"Token credential: {token_credential}")

    service_client = DataLakeServiceClient(account_url, credential=token_credential)
    logger.info(f"Service client: {service_client}")

    return service_client


def create_file_system(service_client: DataLakeServiceClient, file_system_name: str) -> FileSystemClient:
    logger.info(f"Creating file system from name: {file_system_name}")
    
    file_system_client = service_client.create_file_system(file_system=file_system_name)
    logger.info(f"File system client: {file_system_client}")

    return file_system_client


def create_directory(file_system_client: FileSystemClient, directory_name: str) -> DataLakeDirectoryClient:
    logger.info(f"Creating directory with name: {directory_name}")
    
    directory_client = file_system_client.create_directory(directory_name)
    logger.info(f"Directory client: {directory_client}")

    return directory_client


def get_directory_client(file_system_client: FileSystemClient, directory_name: str) -> DataLakeDirectoryClient:
    logger.info(f"Getting directory client from name: {directory_name}")
    
    directory_client = file_system_client.get_directory_client(directory_name)
    logger.info(f"Directory client: {directory_client}")

    return directory_client


# def rename_directory(directory_client: DataLakeDirectoryClient, new_dir_name: str):
#     directory_client.rename_directory(
#         new_name=f"{directory_client.file_system_name}/{new_dir_name}")
    

def upload_file_to_directory(directory_client: DataLakeDirectoryClient, local_path: str, file_name: str):
    logger.info(f"Uploading file to directory: {file_name}")
    
    file_client = directory_client.get_file_client(file_name)
    logger.info(f"File client: {file_client}")

    with open(file=os.path.join(local_path, file_name), mode="rb") as data:
        file_client.upload_data(data, overwrite=True)




# def init_datalake_conn():
#     """
#     Initialize the connection to the Azure Data Lake
#     """
#     # Create a DataLakeDirectoryClient object for the datalake directory
#     datalake_directory_client = DataLakeDirectoryClient(account_url=DATALAKE_ACCOUNT_URI,
#                                                     file_system_name=DATALAKE_FILESYSTEM_NAME,
#                                                     directory_name=DATALAKE_DIRECTORY_PATH)
#     return datalake_directory_client


def sync_data():
    """
    Sync the data from the dataset directory to the datalake directory
    """
    # Create a DataLakeServiceClient object
    service_client = get_service_client_token_credential(DATALAKE_ACCOUNT_URI)

    # Create a FileSystemClient object
    file_system_client = create_file_system(service_client, DATALAKE_FILESYSTEM_NAME)

    # Create a DataLakeDirectoryClient object
    directory_client = get_directory_client(file_system_client, DATALAKE_DIRECTORY_PATH)

    # Upload all files from the dataset directory to the datalake directory
    for file in os.listdir(DATASET_DIR):
        upload_file_to_directory(directory_client, DATASET_DIR, file)
        logger.info(f'Uploaded {file} to datalake directory.')


if __name__ == '__main__':   
    sync_data()
