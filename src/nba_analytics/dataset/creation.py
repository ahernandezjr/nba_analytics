import os
import time
import pandas as pd
from basketball_reference_web_scraper import client

from ..utils.config import settings, azure
from ..utils.logger import get_logger


DATASET_DIR = settings.DATASET_DIR
LOGS_DIR = settings.LOGS_DIR
PLAYERS_CSV_PATH = os.path.join(os.getcwd(), DATASET_DIR, 'nba_players.csv')
PLAYERSTATS_CSV_PATH = os.path.join(os.getcwd(), DATASET_DIR, 'nba_player_stats.csv')
PLAYERS_CSV_PATH_BASIC = os.path.join(os.getcwd(), DATASET_DIR, 'nba_players_basic.csv')
PLAYERS_CSV_PATH_ADVANCED = os.path.join(os.getcwd(), DATASET_DIR, 'nba_players_advanced.csv')
LOG_FILE = os.path.join(os.getcwd(), LOGS_DIR, 'nba_player_stats.log')

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
        # Compare file_path_basic to players_data_basic and append data to file_df if the slug/year does not exist
        file_df = pd.read_csv(file_path_basic)
        for index, row in players_data_basic.iterrows():
            if not ((file_df['slug'] == row['slug']) & (file_df['Year'] == row['Year'])).any():
                file_df = file_df.append(row)
        file_df.to_csv(file_path_basic, index=False)
    
    if not os.path.exists(file_path_advanced) or os.stat(file_path_advanced).st_size == 0:
        logger.info(f"Generating new data for {file_path_advanced} for year {year}")
        players_data_advanced.to_csv(file_path_advanced, index=False)
    else:
        logger.info(f"Appending new data to {file_path_advanced} for year {year}")
        # Compare file_path_advanced to players_data_advanced and append data to file_df if the slug/year does not exist
        file_df = pd.read_csv(file_path_advanced)
        for index, row in players_data_advanced.iterrows():
            if not ((file_df['slug'] == row['slug']) & (file_df['Year'] == row['Year'])).any():
                file_df = file_df.append(row)
        file_df.to_csv(file_path_advanced, index=False)

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
        time.sleep(3.5) # Sleep for 3.5 seconds to avoid rate limiting

    logger.info("Data collection completed.")

if __name__ == '__main__':
    collect_data()
    