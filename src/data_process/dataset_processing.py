import os, sys
import pandas as pd

# add working directory to the path
from ..utils.config import settings
from ..utils.logger import getLogger

# Set configs from settings
DATA_DIR = settings.DATA_DIR

# Create logger
logger = getLogger(__name__)


def extract_positions(df):
    """
    Clean the positions column in the given DataFrame.

    Args:
        df (pandas.DataFrame): The DataFrame containing the positions column.

    Returns:
        pandas.DataFrame: The DataFrame with the positions column cleaned.
    """
    df['positions'] = df['positions'].str.extract("<Position\.(.*): '.*'>", expand=False)
    logger.info(f"Positions extracted: {df['positions'].unique()}")
    return df


def extract_team(df):
    """
    Cleans the 'team' column in the given DataFrame.

    Args:
        df (pandas.DataFrame): The DataFrame containing the 'team' column.

    Returns:
        df (pandas.DataFrame): The DataFrame with the 'team' column cleaned.
    """
    df['team'] = df['team'].str.extract("Team\.(.*)", expand=False)
    logger.info(f"Teams extracted: {df['team'].unique()}")
    return df


def clean_columns(df):
    """
    Removes the 'is_combined_totals' column from the given DataFrame.

    Args:
        df (pandas.DataFrame): The DataFrame to clean.

    Returns:
        pandas.DataFrame: The cleaned DataFrame without the 'is_combined_totals' column.
    """
    columns_to_drop = ['is_combined_totals']
    df = df.drop(columns=columns_to_drop)
    logger.info(f"Columns cleaned: {columns_to_drop}")
    return df


def clean_rows(df):
    """
    Clean the rows of a DataFrame by removing rows with missing values and duplicate values.
    
    Args:
        df (pandas.DataFrame): The DataFrame to be cleaned.
        
    Returns:
        pandas.DataFrame: The cleaned DataFrame.
    """
    # Remove rows with missing values
    df = df.dropna()
    
    # Remove rows with duplicate values
    df = df.drop_duplicates()

    logger.info("Rows cleaned: Missing values and duplicates removed.")
    
    return df


def filter_players_over_5_years(df):
    """
    Filters the given DataFrame to include only players who have played for more than 5 years.

    Args:
        df (pandas.DataFrame): The input DataFrame containing player data.

    Returns:
        pandas.DataFrame: The filtered DataFrame containing players who have played for more than 5 years.
    """
    # Group by player id and count the number of unique years they played
    # player_years = df.groupby('slug')['Year'].nunique()

    # Create a dictionary of players with a unique key of player id and a value of a list of their years played
    player_years_dict = df.groupby('slug')['Year'].apply(list).to_dict()

    # Remove a player from the dictionary if the player has played:
        # for less than 5 years;
        # during 2001, remove that player from the dictionary;
        # has a gap in their years played.
    player_years_dict = {player : years for player, years in player_years_dict.items() if \
                          len(years) >= 5 and \
                          2001 not in years and \
                          list(range(years[0], years[4]+1)) == years[0:5]}

    # Filter the dataframe for players in the dictionary
    df_filtered = df[df['slug'].isin(player_years_dict.keys())]
    
    # Filter for players who have played for more than 5 years
    # df_filtered = df[df['slug'].isin(players_over_5_years.index)]

    # Sort the DataFrame by player year and id
    df_filtered = df_filtered.sort_values(by=['slug', 'Year'])
    # df_filtered = df_filtered.sort_values(by=['Year', 'slug'])

    # Filter dataframe so that each player only has their first 5 years of data
    
    logger.info(f"Players filtered: Players who have played for more than 5 years.")

    return df_filtered


def print_summary(df, df_filtered):
    """
    Prints a summary of the given DataFrame and its filtered version.

    Parameters:
    - df (pandas.DataFrame): The original DataFrame.
    - df_filtered (pandas.DataFrame): The filtered DataFrame.

    Returns:
    None
    """
    # Print the head of the filtered DataFrame
    print(df_filtered.head())

    # Print the number of entries and the number of unique players in the original dataframe
    print(f"Original DataFrame: Entries={len(df)}, Unique Players={len(df['slug'].unique())}")

    # Print the number of entries and the number of unique players in the filtered dataframe
    print(f"Filtered DataFrame: Entries={len(df_filtered)}, Unique Players={len(df_filtered['slug'].unique())}")

    logger.info("Summary printed.")

def filter_5Year_dataset(df):
    """
    Creates a dataset with player statistics for players who have played at least 5 years in the NBA.

    Args:
        df (pandas.DataFrame): The input dataframe containing player statistics.

    Returns:
        pandas.DataFrame: The filtered dataframe with player statistics for players who have played at least 5 years in the NBA.
    """
    df_filtered = df.copy()

    df_filtered = clean_rows(df_filtered)
    df_filtered = clean_columns(df_filtered)
    df_filtered = extract_positions(df_filtered)
    df_filtered = extract_team(df_filtered)
    df_filtered = filter_players_over_5_years(df_filtered)

    return df_filtered


if __name__ == '__main__':
    # Load the data
    df = pd.read_csv('data/nba_player_stats.csv')

    # Create a dataset of players who have played for more than 5 years
    df_filtered = filter_5Year_dataset(df)

    # Save the filtered dataset, overwriting the original file
    df_filtered.to_csv(DATA_DIR + 'nba_player_stats_5years.csv', index=False)

    logger.info(f"Filtered dataset saved to: '{DATA_DIR}nba_player_stats_5years.csv'.")

    # Print the summary
    print_summary(df, df_filtered)
    