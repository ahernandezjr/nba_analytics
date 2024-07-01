from azure.storage.filedatalake import DataLakeStoreAccount, DataLakeDirectoryClient

from ..utils.config import settings, azure
from ..utils.logger import get_logger


# Create logger
logger = get_logger(__name__)

# Replace with your own values
DATASET_DIR = settings.DATASET_DIR
DATALAKE_ACCOUNT_NAME = azure.DATALAKE_ACCOUNT_NAME
DATALAKE_ACCOUNT_KEY = azure.DATALAKE_ACCOUNT_KEY
DATALAKE_FILESYSTEM_NAME = azure.DATALAKE_FILESYSTEM_NAME
DATALAKE_DIRECTORY_PATH = azure.DATALAKE_DIRECTORY_PATH

# Create a DataLakeStoreAccount object
account = DataLakeStoreAccount(account_url=f"https://{DATALAKE_ACCOUNT_NAME}.dfs.core.windows.net",
                               credential=DATALAKE_ACCOUNT_KEY)

# Create a DataLakeDirectoryClient object for the datalake directory
datalake_directory_client = DataLakeDirectoryClient(account=account,
                                                   file_system_name=DATALAKE_FILESYSTEM_NAME,
                                                   directory_path=DATALAKE_DIRECTORY_PATH)

# Sync all the data from the dataset directory to the datalake directory
for root, dirs, files in os.walk(DATASET_DIR):
    for file in files:
        local_file_path = os.path.join(root, file)
        remote_file_path = os.path.join(DATALAKE_DIRECTORY_PATH, file)
        with open(local_file_path, 'rb') as local_file:
            datalake_directory_client.upload_file(file=remote_file_path, data=local_file)