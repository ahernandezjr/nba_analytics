import pandas as pd
from sklearn.decomposition import PCA

from ..utils.config import settings
from ..utils.logger import get_logger


# Set configs from settings
DATA_DIR = settings.DATA_DIR
DATA_FILE_NAME = settings.DATA_FILE_NAME
DATA_FILE_5YEAR_NAME = settings.DATA_FILE_5YEAR_NAME
DATA_FILE_5YEAR_TENSOR_NAME = settings.DATA_FILE_5YEAR_TENSOR_NAME
DATA_FILE_5YEAR_JSON_NAME = settings.DATA_FILE_5YEAR_JSON_NAME


# Create logger
logger = get_logger(__name__)



def filter_players_over_5_years(df):
    """
    Filters the given DataFrame to include only players who have played for more than 5 years.

    Args:
        df (pandas.DataFrame): The input DataFrame containing player data.

    Returns:
        pandas.DataFrame: The overlapping DataFrame containing players who have played for 5 or more years.
        pandas.DataFrame: The non-overlapping DataFrame containing the first 5 years of players who have played for 5 or more years.
    """
    logger.debug("Filtering players who have played for more than 5 years...")

    # Remove both of duplicate years per player from the DataFrame
    df_filtered = df.drop_duplicates(subset=['slug', 'Year'], keep=False)

    # Create a dictionary of players with a unique key of player id and a value of a list of their years played
    player_years_dict = df_filtered.groupby('slug')['Year'].apply(list).to_dict()

    # Include a player from the dictionary if the player has played:
        # 1. for equal to or more than 5 years;
        # 2. during 2001, remove that player from the dictionary; and
        # 3. has continuous 5 years in years (checks for a gap or trades).
    dict_overlap = {player: years for player, years in player_years_dict.items() if
                            len(years) >= 5 and
                            2001 not in years and
                            any(years[i] + 5 == years[i + 5] for i in range(len(years)) if
                                i + 5 < len(years))}
    
    # Filter the dataframe for players in the dictionary
    df_overlap = df_filtered[df_filtered['slug'].isin(dict_overlap.keys())]
    
    # Sort the DataFrame by player year and id
    df_overlap = df_overlap.sort_values(by=['slug', 'Year'])

    # Remove rows if not part of a continuous 5 year period
    # iterate through the DataFrame to remove rows that are not part of a continuous 5 year period
    df_overlap = df_overlap.groupby('slug').filter(lambda x: any(x['Year'].values[i] + 5 == x['Year'].values[i + 5] \
                                                                    for i in range(len(x['Year'].values)) \
                                                                        if i + 5 < len(x['Year'].values)))

    # Exclude a player from the dictionary if the player has played:
        # 1. for less than 5 years;
        # 2. during 2001, remove that player from the dictionary; and
        # 3. has continuous years (checks for a gap or trades).
    # TO DO: ALLOW FOR CONTINUOUS PLAYERS OVER FIRST 5 YEARS AND SPLIT YEARS FOR LARGER TRAINING SET
    dict_first_five = {player : years for player, years in player_years_dict.items() if \
                          len(years) >= 5 and \
                          2001 not in years and \
                          list(range(years[0], years[len(years) - 1] + 1)) == years[0:len(years)]}

    # Filter the dataframe for players in the dictionary
    df_first_five = df[df['slug'].isin(dict_first_five.keys())]
    
    # Sort the DataFrame by player year and id
    df_first_five = df_first_five.sort_values(by=['slug', 'Year'])

    # For each player in the dataframe, keep only the first 5 years
    df_first_five = df_first_five.groupby('slug').head(5)

    return df_overlap, df_first_five


def pca_analysis(df):
    """
    Applies pca to the given DataFrame.

    Args:
        df (pandas.DataFrame): The DataFrame to PCA.

    Returns:
        pandas.DataFrame: The PCA output DataFrame.
    """
    logger.debug("Applying PCA to data...")

    df_principal = df.drop(columns=['slug'])

    # Implement PCA analysis to reduce the number of features
    pca = PCA(n_components=None)
    principalComponents = pca.fit_transform(df_principal)
    df_principal = pd.DataFrame(data = principalComponents)

    # Add the slug column back to the DataFrame
    df_principal['slug'] = df['slug']
    

    return df_principal


# Use R for techniques on column reduction



