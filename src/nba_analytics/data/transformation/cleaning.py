"""
This module provides various data cleaning functions for NBA player statistics data.
It includes functions to clean specific columns, remove non-tensor values, and standardize the data.
"""

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

    df['positions'] = df['positions'].str.extract(r"<Position\.(.*?): '.*?'>", expand=False)
    
    logger.debug(f"Positions extracted: {df['positions'].unique()}")

    return df

def extract_team(df):
    """
    Cleans the 'team' column in the given DataFrame.

    Args:
        df (pandas.DataFrame): The DataFrame containing the 'team' column.

    Returns:
        pandas.DataFrame: The DataFrame with the 'team' column cleaned.
    """
    logger.debug(f"Extracting teams...")

    df['team'] = df['team'].str.extract(r"Team\.(.*)", expand=False)

    logger.debug(f"Teams extracted: {df['team'].unique()}.")

    return df

def extract_columns(df):
    """
    Extracts the columns from the given DataFrame that are relevant for analysis.

    Args:
        df (pandas.DataFrame): The DataFrame to extract columns from.

    Returns:
        pandas.DataFrame: The DataFrame with the relevant columns extracted.
    """
    logger.debug(f"Extracting columns...")

    df = extract_team(df)
    df = extract_positions(df)

    logger.debug(f"Columns extracted.")

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
