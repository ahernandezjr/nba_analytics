"""
This module provides functions for filtering and processing NBA player statistics data.
It includes functions to filter players based on continuous years of play and to filter specific columns.
"""

import pandas as pd
from sklearn.decomposition import PCA

from ...utils import filename_grabber
from ...utils.config import settings
from ...utils.logger import get_logger

FILTER_AMT = settings.dataset.FILTER_AMT

# Create logger
logger = get_logger(__name__)


def filter_columns(df):
    """
    Filters the DataFrame to keep only specific columns relevant for analysis.

    Args:
        df (pandas.DataFrame): The input DataFrame containing player data.

    Returns:
        pandas.DataFrame: The filtered DataFrame with only the specified columns.
    """
    logger.debug(f"Filtering columns...")

    columns_to_keep = ['slug', 'Year', 'age',
                       'minutes_played', 'made_field_goals', 'attempted_field_goals',
                       'attempted_three_point_field_goals', 'attempted_free_throws',
                       'defensive_rebounds', 'turnovers', 'player_efficiency_rating',
                       'total_rebound_percentage', 'value_over_replacement_player']
    
    filtered_df = df[columns_to_keep]

    return filtered_df

# TO BE IMPLEMENTED AT DEPTH IN FUTURE
def filter_nontensor_values(df):
    """
    Filters the given DataFrame to remove non-tensor values.

    Args:
        df (pandas.DataFrame): The input DataFrame.

    Returns:
        pandas.DataFrame: The filtered DataFrame.
    """
    logger.debug("Filtering non-tensor values...")

    df_filtered = df.copy()

    # Perform numerical conversion on all columns except the slug column
    for column in df_filtered.columns:
        if column != 'slug':
            # Convert values to proper types (i.e. str should be converted to int or float)
            df_filtered[column] = df_filtered[column].apply(pd.to_numeric, errors='coerce')

    # Drop columns that have NaN values
    df_filtered = df_filtered.dropna(axis=1)

    # TODO: Implement further data cleaning steps here
        # One hot encoding for categorical values etc...

    # # Apply PCA to the filtered DataFrame
    # df_filtered = pca_analysis(df_filtered)

    return df_filtered

def get_player_years_dict(df):
    """
    Creates a dictionary of players with a unique key of player id and a value of a list of their years played.

    Args:
        df (pandas.DataFrame): The input DataFrame containing player data.

    Returns:
        dict: A dictionary with player ids as keys and lists of years played as values.
    """
    return df.groupby('slug')['Year'].apply(list).to_dict()

def get_continuous_years(years, min_years=FILTER_AMT):
    """
    Gets all continuous periods of given length in the list of years.

    Args:
        years (list): List of years the player has played.
        min_years (int): Minimum number of continuous years required.

    Returns:
        list: List of all years within continuous periods of given length.
    """
    continuous_years = set()
    years = sorted(years)
    for i in range(len(years) - min_years + 1):
        if all(years[j] - years[i] == j - i for j in range(i, i + min_years)):
            continuous_years.update(years[i:i + min_years])
            j = i + min_years
            while j < len(years) and years[j] - years[j - 1] == 1:
                continuous_years.add(years[j])
                j += 1
    return list(continuous_years)

def has_continuous_stretch(years, min_years):
    """
    Checks if there is any continuous stretch of given length in the list of years.

    Args:
        years (list): List of years the player has played.
        min_years (int): Minimum number of continuous years required.

    Returns:
        bool: True if there is a continuous stretch of given length, False otherwise.
    """
    return any(
        years[i] + min_years - 1 == years[i + min_years - 1]
        for i in range(len(years) - min_years + 1)
    )

def filter_atleast_continuous_years(df, min_years=FILTER_AMT):
    """
    Filters the given DataFrame to include only players who have continuous stretches of given length.

    Args:
        df (pandas.DataFrame): The input DataFrame containing player data.
        min_years (int, optional): Minimum number of continuous years required. Default is 5.

    Returns:
        pandas.DataFrame: The DataFrame containing players who have continuous stretches of given length.
    """
    logger.debug(f"Filtering players who have continuous stretches of at least {min_years} years...")

    player_years_dict = get_player_years_dict(df)

    dict_continuous = {
        player: years for player, years in player_years_dict.items()
        if len(years) >= min_years and 2001 not in years and get_continuous_years(years, min_years)
    }

    df_continuous = df[df['slug'].isin(dict_continuous.keys())]
    df_continuous = df_continuous.sort_values(by=['slug', 'Year'])

    df_continuous = df_continuous.groupby('slug').filter(
        lambda x: has_continuous_stretch(x['Year'].values, min_years)
    )

    return df_continuous

def filter_first_continuous_years(df, min_years=FILTER_AMT):
    """
    Filters the given DataFrame to include only players whose first 5 years have a continuous stretch of given length.

    Args:
        df (pandas.DataFrame): The input DataFrame containing player data.
        min_years (int, optional): Minimum number of continuous years required. Default is 5.

    Returns:
        pandas.DataFrame: The DataFrame containing players whose first 5 years have a continuous stretch of given length.
    """
    logger.debug(f"Filtering players for a continuous stretch of {min_years} years in their first 5 years...")

    player_years_dict = get_player_years_dict(df)

    dict_first_continuous = {
        player: years for player, years in player_years_dict.items()
        if len(years) >= min_years and 2001 not in years and 
        get_continuous_years(years[:5], min_years)
    }

    df_first_continuous = df[df['slug'].isin(dict_first_continuous.keys())]
    df_first_continuous = df_first_continuous.sort_values(by=['slug', 'Year'])
    df_first_continuous = df_first_continuous.groupby('slug').head(5)

    return df_first_continuous
