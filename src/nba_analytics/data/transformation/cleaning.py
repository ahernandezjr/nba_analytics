import numpy as np
import pandas as pd

from . import reduction

from ...utils import filename_grabber
from ...utils.config import settings
from ...utils.logger import get_logger


logger = get_logger(__name__)


def extract_positions(df):
    """
    Clean the positions column in the given DataFrame.

    Args:
        df (pandas.DataFrame): The DataFrame containing the positions column.

    Returns:
        pandas.DataFrame: The DataFrame with the positions column cleaned.
    """
    logger.debug(f"Extracting positions...")

    df['positions'] = df['positions'].str.extract("<Position\.(.*): '.*'>", expand=False)

    logger.debug(f"Positions extracted: {df['positions'].unique()}")

    return df


def extract_team(df):
    """
    Cleans the 'team' column in the given DataFrame.

    Args:
        df (pandas.DataFrame): The DataFrame containing the 'team' column.

    Returns:
        df (pandas.DataFrame): The DataFrame with the 'team' column cleaned.
    """
    logger.debug(f"Extracting teams...")

    filtered_df = df.copy()

    filtered_df['team'] = filtered_df['team'].str.extract("Team\.(.*)", expand=False)
    
    logger.debug(f"Teams extracted: {filtered_df['team'].unique()}.")

    return df


def clean_columns(df):
    """
    Removes the 'is_combined_totals' column from the given DataFrame.
    Current columns to keep:
        minutes_played
        made_field_goals
        attempted_field_goals
        attempted_three_point_field_goals
        attempted_free_throws
        defensive_rebounds
        turnovers
        player_efficiency_rating
        total_rebound_percentage
        value_over_replacement_player

    Args:
        df (pandas.DataFrame): The DataFrame to clean.

    Returns:
        pandas.DataFrame: The cleaned DataFrame without the 'is_combined_totals' column.
    """
    logger.debug(f"Cleaning columns...")

    columns_to_drop = ['is_combined_totals']
    clean_df = df.drop(columns=columns_to_drop)

    logger.debug(f"Cleaned columns: {columns_to_drop}.")

    return clean_df


def clean_rows(df):
    """
    Clean the rows of a DataFrame by removing rows with missing values and duplicate values.
    
    Args:
        df (pandas.DataFrame): The DataFrame to be cleaned.
        
    Returns:
        pandas.DataFrame: The cleaned DataFrame.
    """
    logger.debug("Cleaning rows...")

    clean_df = df.copy()

    # Remove rows with missing values
    clean_df = clean_df.dropna()
    
    # Remove rows with duplicate values
    clean_df = clean_df.drop_duplicates()
    
    # Remove both of duplicate years per player from the DataFrame
    clean_df = clean_df.drop_duplicates(subset=['slug', 'Year'], keep=False)

    return clean_df


# TO BE IMPLEMENTED AT DEPTH IN FUTURE
def clean_nontensor_values(df):
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

            # Filter the DataFrame to remove non-tensor values (NOT WORKING)
            # df_filtered = df_filtered.select_dtypes(include=['number'])
            # df_filtered = df_filtered.dropna()

    # Drop columns that have NaN values
    df_filtered = df_filtered.dropna(axis=1)

    # TODO: Implement further data cleaning steps here
        # One hot encoding for categorical values etc...

    # # Apply PCA to the filtered DataFrame
    # df_filtered = pca_analysis(df_filtered)

    return df_filtered


def standardize_data(df):
    """
    Standardizes the given DataFrame.

    Args:
        df (pandas.DataFrame): The DataFrame to standardize.

    Returns:
        pandas.DataFrame: The standardized DataFrame.
    """
    logger.debug("Standardizing data...")

    # Standardize the data
    df_standardized = df.copy()
    df_standardized = (df_standardized - df_standardized.mean()) / df_standardized.std()

    return df_standardized
