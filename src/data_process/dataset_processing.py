import os, sys
import pandas as pd

# add working directory to the path
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from ..utils.config import settings

DATA_DIR = settings.DATA_DIR

def clean_positions(df):
    '''Clean the positions column in the DataFrame'''
    df['positions'] = df['positions'].str.extract("<Position\.(.*): '.*'>", expand=False)
    return df


def clean_team(df):
    '''Clean the team column in the DataFrame'''
    df['team'] = df['team'].str.extract("Team\.(.*)", expand=False)
    return df


def clean_columns(df):
    '''Clean the unnecessary columns ('is_combined_totals') from the DataFrame'''
    df = df.drop(columns=['is_combined_totals'])
    return df


def clean_rows(df):
    '''Clean the rows with missing values and duplicates'''
    # Remove rows with missing values
    df = df.dropna()
    
    # Remove rows with duplicate values
    df = df.drop_duplicates()
    
    return df


def filter_players_over_5_years(df):
    '''Filter players who have played for more than 5 years'''
    # Group by player id and count the number of unique years they played
    player_years = df.groupby('slug')['Year'].nunique()

    # Filter players who have played for more than 5 years
    players_over_5_years = player_years[player_years > 5]

    # Get a DataFrame of players who have played for more than 5 years
    df_filtered = df[df['slug'].isin(players_over_5_years.index)]

    # Sort the DataFrame by player year and id
    df_filtered = df_filtered.sort_values(by=['slug', 'Year'])
    # df_filtered = df_filtered.sort_values(by=['Year', 'slug'])
    
    return df_filtered


def print_summary(df, df_filtered):
    '''Print a summary of the original and filtered DataFrame'''
    # Print the head of the filtered DataFrame
    print(df_filtered.head())

    # Print the number of entries and the number of unique players in the original dataframe
    print(f"Original DataFrame: Entries={len(df)}, Unique Players={len(df['slug'].unique())}")

    # Print the number of entries and the number of unique players in the filtered dataframe
    print(f"Filtered DataFrame: Entries={len(df_filtered)}, Unique Players={len(df_filtered['slug'].unique())}")


def create_5Year_dataset(df):
    '''Create a csv file dataset of players who have played for more than 5 years'''
    df_filtered = df.copy()
    df_filtered = clean_positions(df_filtered)
    df_filtered = clean_team(df)
    df_filtered = clean_rows(df_filtered)
    df_filtered = clean_columns(df_filtered)
    df_filtered = filter_players_over_5_years(df_filtered)
    df_filtered.to_csv(DATA_DIR + '/nba_player_stats_5years.csv', index=False)

    return df_filtered


if __name__ == '__main__':
    # Load the data
    df = pd.read_csv('data/nba_player_stats.csv')

    # Create a dataset of players who have played for more than 5 years
    df_filtered = create_5Year_dataset(df)

    # Print the summary
    print_summary(df, df_filtered)
    