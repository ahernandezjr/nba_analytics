import os, sys
import time
import logging
from colorlog import ColoredFormatter

from ..utils import filename_grabber
from ..utils.config import settings


LOGS_DIR = filename_grabber.get_logs_dir()


# get current time
def get_current_time():
    return time.strftime("[%Y%m%d-%H%M%S]")


# delete old logs up to x
def delete_old_logs(keep=5):
    logs = [f for f in os.listdir(LOGS_DIR) if f.endswith('.log')]
    logs.sort(key=lambda x: os.path.getmtime(os.path.join(LOGS_DIR, x)))
    for log in logs[:-keep]:
        os.remove(os.path.join(LOGS_DIR, log))


# create logger
def get_logger(name):
    # Create logs directory if it does not exist
    if not os.path.exists(LOGS_DIR):
        os.makedirs(LOGS_DIR)

    # Delete old logs
    delete_old_logs()

    # Create log file path
    log_file = os.path.join(LOGS_DIR, f'{1}.log')
    logging.basicConfig(filename=log_file, level=logging.DEBUG)

    # Configure logging
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    format_string = '| %(asctime)s - %(levelname)s - %(module)s.%(funcName)s:\n---| %(message)s'

    # Create a colored formatter
    formatter = ColoredFormatter(
        '%(log_color)s' + format_string,
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

    if not logger.handlers:
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # File handler
        file_handler = logging.FileHandler(log_file, mode='w')
        file_handler.setFormatter(logging.Formatter(format_string))
        logger.addHandler(file_handler)

    return logger
