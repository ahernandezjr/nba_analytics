import os
import sys
import time
import csv
import math
import pandas as pd
from basketball_reference_web_scraper import client
from basketball_reference_web_scraper.data import OutputType

# Add parent directory to the sys.path to enable importing from sibling modules
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.path.dirname(__file__), '..', '..')))
from src.utils.logger import logger  # Importing logger from sibling module

# Constants
DATASET_DIR = 'data'
LOGS_DIR = 'logs'
PLAYERS_CSV_PATH = os.path.join(os.getcwd(), DATASET_DIR, 'nba_players.csv')
PLAYERSTATS_CSV_PATH = os.path.join(os.getcwd(), DATASET_DIR, 'nba_player_stats.csv')
# Define file paths for basic and advanced player data
PLAYERS_CSV_PATH_BASIC = os.path.join(os.getcwd(), DATASET_DIR, 'nba_players_basic.csv')
PLAYERS_CSV_PATH_ADVANCED = os.path.join(os.getcwd(), DATASET_DIR, 'nba_players_advanced.csv')
LOG_FILE = os.path.join(os.getcwd(), LOGS_DIR, 'nba_player_stats.log')

BATCH_SIZE = 100
RETRY_DELAY = 60  # Delay in seconds before retrying after rate limit exceeded
YEARS_BACK = 12  # Number of years to go back


# Create directories if they don't exist
for directory in [DATASET_DIR, LOGS_DIR]:
    os.makedirs(directory, exist_ok=True)


# # Function to check if a file is empty
# def is_file_empty(file_path):
#     """Check if a file is empty."""
#     return os.path.exists(file_path) and os.stat(file_path).st_size == 0

# # Function to retrieve career statistics for a given player ID with rate limiting handling
# def get_player_season_stats(player_id, season, retry_delay=RETRY_DELAY):
#     """Retrieve career statistics for a given player ID and season with rate limiting handling."""
#     consecutive_years_no_stats = 0  # Counter for consecutive years without stats
#     while True:
#         try:
#             start_year, end_year = map(int, season.split('-'))
#             player_stats = client.players_season_totals(season_end_year=end_year, output_type=OutputType.CSV)
#             if player_stats is not None:
#                 return player_stats
#             else:
#                 logger.warning(f"No career statistics found for player ID {player_id} in season {season}")
#                 consecutive_years_no_stats += 1
#                 if consecutive_years_no_stats >= 2:
#                     logger.info(f"Skipping remaining searches for player ID {player_id} since no stats found for two consecutive years before {season}")
#                     break
#                 else:
#                     # Try the previous year
#                     start_year -= 1
#                     end_year -= 1
#                     season = f"{start_year}-{end_year}"
#         except Exception as e:
#             logger.error(f"Error retrieving career statistics for player ID {player_id} in season {season}: {str(e)}")
#             if "Read timed out" in str(e):
#                 # NBA API throttle, add delay and retry
#                 logger.warning("Rate limit exceeded. Adding delay before retrying...")
#                 time.sleep(retry_delay)
#             else:
#                 # Other errors, return None without logging
#                 return None
def update_players_csv(file_path_basic, file_path_advanced, year):
    """Update player collection in CSV file for a specific year."""
    
    # Obtain data
    players_data_basic    = client.players_season_totals(season_end_year=year)
    players_data_advanced = client.players_advanced_season_totals(season_end_year=year)
    
    # Convert to dataframes
    players_data_basic    = pd.DataFrame(players_data_basic)
    players_data_advanced = pd.DataFrame(players_data_advanced)

    # Add year column to the dataframes
    players_data_basic['Year']    = year
    players_data_advanced['Year'] = year
    
    # Add data to files
    # Check if basic data file exists and is empty
    if not os.path.exists(file_path_basic) or os.stat(file_path_basic).st_size == 0:
        logger.info(f"Generating new data for {file_path_basic} for year {year}")
        players_data_basic.to_csv(file_path_basic, index=False)  # Save basic data to file
    else:
        # logger.info(f"Loading existing basic data from {file_path_basic}")
        # Append new basic data to existing basic file
        logger.info(f"Appending new data to {file_path_basic} for year {year}")
        players_data_basic.to_csv(file_path_basic, mode='a', index=False, header=False)
    
    # Check if advanced data file exists and is empty
    if not os.path.exists(file_path_advanced) or os.stat(file_path_advanced).st_size == 0:
        logger.info(f"Generating new data for {file_path_advanced} for year {year}")
        players_data_advanced.to_csv(file_path_advanced, index=False)  # Save advanced data to file
    else:
        # logger.info(f"Loading existing advanced data from {file_path_advanced}")
        # Append new advanced data to existing advanced file
        logger.info(f"Appending new data to {file_path_advanced} for year {year}")
        players_data_advanced.to_csv(file_path_advanced, mode='a', index=False, header=False)

    logger.info(f"Data saved to {file_path_basic} and {file_path_advanced} for year {year}")


def merge_player_data(file_path_basic, file_path_advanced, output_file_path):
    """Merge player data from basic and advanced CSV files."""
    # Read basic and advanced player data from CSV files
    players_basic_df = pd.read_csv(file_path_basic)
    players_advanced_df = pd.read_csv(file_path_advanced)

    # Merge basic and advanced DataFrames along the columns axis
    merged_players_df = pd.concat([players_basic_df, players_advanced_df], axis=1)

    # Drop duplicate columns, keeping only the first occurrence
    merged_players_df = merged_players_df.loc[:, ~merged_players_df.columns.duplicated()]

    # Save the merged data to a new CSV file
    merged_players_df.to_csv(output_file_path, index=False)

# # Define file paths for basic and advanced player data
# DATASET_DIR = "data"
# PLAYERS_CSV_PATH_BASIC = os.path.join(os.getcwd(), DATASET_DIR, 'nba_players_basic.csv')
# PLAYERS_CSV_PATH_ADVANCED = os.path.join(os.getcwd(), DATASET_DIR, 'nba_players_advanced.csv')
# PLAYERSTATS_CSV_PATH = os.path.join(os.getcwd(), DATASET_DIR, 'merged_player_stats.csv')

# Create directories if they don't exist
for directory in [DATASET_DIR, LOGS_DIR]:
    os.makedirs(directory, exist_ok=True)

# Iterate through the years 2001-2024
for year in range(2001, 2025):
# for year in range(2001, 2025):
    # Update player collection in CSV files for basic and advanced data
    update_players_csv(PLAYERS_CSV_PATH_BASIC, PLAYERS_CSV_PATH_ADVANCED, year)
    
    # Merge player data for the current year
    merge_player_data(PLAYERS_CSV_PATH_BASIC, PLAYERS_CSV_PATH_ADVANCED, PLAYERSTATS_CSV_PATH)
    
    # Add a small delay for anti-scraping measures
    time.sleep(3.5)  # Sleep for 1 second


# # Create directories if they don't exist
# for directory in [DATASET_DIR, LOGS_DIR]:
#     os.makedirs(directory, exist_ok=True)

# # Update player collection in CSV file
# update_players_csv(PLAYERS_CSV_PATH)

# # Load the player data from the CSV file
# players_df = pd.read_csv(PLAYERS_CSV_PATH)

# Process players and collect statistics
# with open(DATA_CSV_PATH, 'w', newline='', encoding='utf-8') as csvfile:
#     writer = csv.DictWriter(csvfile, fieldnames=['PlayerID', 'Name', 'ActiveStatus', 'Season', 'Points', 'Assists', 'Rebounds', 'Steals', 'Blocks', 'Turnovers'])
#     writer.writeheader()

#     num_players = len(players_df)
#     num_batches = math.ceil(num_players / BATCH_SIZE)

#     for i in range(num_batches):
#         start_idx = i * BATCH_SIZE
#         end_idx = min((i + 1) * BATCH_SIZE, num_players)
#         batch_player_ids = players_df.iloc[start_idx:end_idx]['slug'].tolist()

#         logger.info(f"Batch {i+1}/{num_batches}: Processing {len(batch_player_ids)} players.")

#         for player_id in batch_player_ids:
#             for year in range(YEARS_BACK):
#                 season = f"{2023 - year}-{2024 - year}"
#                 player_stats = get_player_season_stats(player_id, season, retry_delay=RETRY_DELAY)

#                 if player_stats is not None:
#                     for _, row in player_stats.iterrows():
#                         player_id = row['player_id']
#                         player_name = row['name']
#                         active_status = 'Active' if player_name else 'Inactive'

#                         writer.writerow({
#                             'PlayerID': player_id,
#                             'Name': player_name,
#                             'ActiveStatus': active_status,
#                             'Season': season,
#                             'Points': row['points'],
#                             'Assists': row['assists'],
#                             'Rebounds': row['total_rebounds'],
#                             'Steals': row['steals'],
#                             'Blocks': row['blocks'],
#                             'Turnovers': row['turnovers']
#                         })
#                 else:
#                     logger.warning(f"No career statistics found for player ID {player_id} in season {season}")

#         progress_percent = ((i + 1) / num_batches) * 100
#         logger.info(f"Progress: {progress_percent:.2f}%")

logger.info("Data collection completed.")
