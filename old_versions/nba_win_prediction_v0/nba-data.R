# import games
## used to determine games, games dates, and games outcomes
games         = read.csv("D:\\repos\\nba-model\\games.csv")
games

# import games
## used to determine players
games_details = read.csv("D:\\repos\\nba-model\\games_details.csv")
games_details

## Export to csv
write.csv(new_games, "D:\\repos\\nba-model\\new_games.csv")


# Create dictionary for team_ids to team abbreviation
team_ids_dict = subset(games_details, select = c(TEAM_ID,
                                                 TEAM_ABBREVIATION,
                                                 TEAM_CITY))
team_ids_dict = team_ids_dict[!duplicated(team_ids_dict), ]
nrow(team_ids_dict)
team_ids_dict = head(team_ids_dict, 30)
team_ids_dict


# Create dictionary for player_ids to player name
player_ids_dict = subset(games_details, select = c(PLAYER_ID,
                                                   PLAYER_NAME))
player_ids_dict = player_ids_dict[!duplicated(player_ids_dict), ]
nrow(player_ids_dict)
player_ids_dict

## Export to csv
write.csv(player_ids_dict, "D:\\repos\\nba-model\\player_to_ids.csv")


# import players_data
players_data = read.csv("D:\\repos\\nba-model\\players_data.csv")
new_players = subset( players_data, select = -c(X, Rk, MVP))
new_players[is.na(new_players)] = 0
new_players

## Export to csv
write.csv(new_players, "D:\\repos\\nba-model\\new_players.csv")


# Minimization arrays
defunct = c("NJN", "NOH", "NOK", "SEA")
present = c("DNP", "NWT", "DND")


# Select certain games
new_games = subset(games, select = c(GAME_DATE_EST,
                                     GAME_ID,
                                     HOME_TEAM_ID,
                                     VISITOR_TEAM_ID,
                                     SEASON,
                                     PTS_home,
                                     PTS_away,
                                     HOME_TEAM_WINS))
new_games


# Minimize games_details
new_details = subset(games_details, select = c(GAME_ID,
                                                     TEAM_ID,
                                                     TEAM_ABBREVIATION,
                                                     PLAYER_ID,
                                                     PLAYER_NAME,
                                                     COMMENT))

new_details = new_details[new_details$TEAM_ABBREVIATION %in% defunct, ]
new_details = new_details[new_details$COMMENT =="",]

new_details = subset(games_details, select = c(GAME_ID,
                                               TEAM_ID,
                                               TEAM_ABBREVIATION,
                                               PLAYER_NAME,
                                               PLAYER_ID))

new_details

# total players in games
nrow(new_details)
# most players on one team in a game
max(table(new_details$GAME_ID, new_details$TEAM_ABBREVIATION))

## Export to csv
write.csv(new_details, "D:\\repos\\nba-model\\new_details.csv")

