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