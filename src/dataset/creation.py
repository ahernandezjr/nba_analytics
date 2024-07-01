import os
import time
import pandas as pd
from basketball_reference_web_scraper import client

from ..utils.logger import get_logger

DATASET_DIR = 'data'
LOGS_DIR = 'logs'
PLAYERS_CSV_PATH = os.path.join(DATASET_DIR, 'nba_players.csv')
PLAYERSTATS_CSV_PATH = os.path.join(DATASET_DIR, 'nba_player_stats.csv')
PLAYERS_CSV_PATH_BASIC = os.path.join(DATASET_DIR, 'nba_players_basic.csv')
PLAYERS_CSV_PATH_ADVANCED = os.path.join(DATASET_DIR, 'nba_players_advanced.csv')
LOG_FILE = os.path.join(LOGS_DIR, 'nba_player_stats.log')

BATCH_SIZE = 100
RETRY_DELAY = 60
YEARS_BACK = 12

logger = get_logger(__name__)

def create_directories():
    for directory in [DATASET_DIR, LOGS_DIR]:
        os.makedirs(directory, exist_ok=True)

def update_players_csv(file_path_basic, file_path_advanced, year):
    players_data_basic = client.players_season_totals(season_end_year=year)
    players_data_advanced = client.players_advanced_season_totals(season_end_year=year)
    
    players_data_basic = pd.DataFrame(players_data_basic)
    players_data_advanced = pd.DataFrame(players_data_advanced)

    players_data_basic['Year'] = year
    players_data_advanced['Year'] = year
    
    if not os.path.exists(file_path_basic) or os.stat(file_path_basic).st_size == 0:
        logger.info(f"Generating new data for {file_path_basic} for year {year}")
        players_data_basic.to_csv(file_path_basic, index=False)
    else:
        logger.info(f"Appending new data to {file_path_basic} for year {year}")
        players_data_basic.to_csv(file_path_basic, mode='a', index=False, header=False)
    
    if not os.path.exists(file_path_advanced) or os.stat(file_path_advanced).st_size == 0:
        logger.info(f"Generating new data for {file_path_advanced} for year {year}")
        players_data_advanced.to_csv(file_path_advanced, index=False)
    else:
        logger.info(f"Appending new data to {file_path_advanced} for year {year}")
        players_data_advanced.to_csv(file_path_advanced, mode='a', index=False, header=False)

    logger.info(f"Data saved to {file_path_basic} and {file_path_advanced} for year {year}")

def merge_player_data(file_path_basic, file_path_advanced, output_file_path):
    players_basic_df = pd.read_csv(file_path_basic)
    players_advanced_df = pd.read_csv(file_path_advanced)

    merged_players_df = pd.concat([players_basic_df, players_advanced_df], axis=1)

    merged_players_df = merged_players_df.loc[:, ~merged_players_df.columns.duplicated()]

    merged_players_df.to_csv(output_file_path, index=False)

def collect_data():
    create_directories()

    for year in range(2001, 2025):
        update_players_csv(PLAYERS_CSV_PATH_BASIC, PLAYERS_CSV_PATH_ADVANCED, year)
        merge_player_data(PLAYERS_CSV_PATH_BASIC, PLAYERS_CSV_PATH_ADVANCED, PLAYERSTATS_CSV_PATH)
        time.sleep(3.5)

    logger.info("Data collection completed.")

if __name__ == '__main__':
    collect_data()
    