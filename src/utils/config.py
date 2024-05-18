import os
from dynaconf import Dynaconf

# Set the SETTINGS_FILE_FOR_DYNACONF environment variable
os.environ['SETTINGS_FILE_FOR_DYNACONF'] = 'config/settings.toml'

# Create a settings object
settings = Dynaconf(load_dotenv=True)