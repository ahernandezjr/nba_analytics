import os, sys
import logging
import sys
from colorlog import ColoredFormatter

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
LOGS_DIR = 'logs'
LOG_FILE = os.path.join(LOGS_DIR, 'nba_player_stats.log')

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Create a colored formatter
formatter = ColoredFormatter(
    "%(log_color)s%(asctime)s - %(levelname)s - %(message)s",
    datefmt=None,
    reset=True,
    log_colors={
        'DEBUG':    'cyan',
        'INFO':     'blue',
        'WARNING':  'yellow',
        'ERROR':    'red',
        'CRITICAL': 'red,bg_white',
    },
    secondary_log_colors={},
    style='%'
)

# Console handler
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# File handler
file_handler = logging.FileHandler(LOG_FILE, mode='w')
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)
