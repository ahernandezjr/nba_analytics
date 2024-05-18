import pandas as pd

# Load the data
df = pd.read_csv('data/nba_player_stats.csv')

# Group by player name and count the number of unique years they played
player_years = df.groupby('slug')['Year'].nunique()

# Filter players who have played for more than 5 years
players_over_5_years = player_years[player_years > 5]

# Get a DataFrame of players who have played for more than 5 years
df_filtered = df[df['slug'].isin(players_over_5_years.index)]

# print the head of the filtered DataFrame
print(df_filtered.head())

# print the number of unique players
print(len(df_filtered['slug'].unique()))