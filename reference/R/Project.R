# Load required libraries
library(readr)
library(dplyr)
library(tidyr)

# Read the CSV file
nba_data <- read_csv("E:\repos\school\CS52540\R Project\data\nba_player_stats.csv")

# Select relevant columns and clean data
nba_data_cleaned <- nba_data %>%
  select(-c(slug, name, is_combined_totals)) %>%
  mutate(
    positions = gsub("\\[|\\]|<Position\\.|:.*", "", positions),
    team = gsub("Team\\.", "", team)
  )

# Convert positions to usable format
positions <- unique(unlist(strsplit(nba_data_cleaned$positions, ",")))
for (pos in positions) {
  nba_data_cleaned <- nba_data_cleaned %>%
    mutate(!!paste0("position_", pos) := grepl(pos, positions))
}

# Convert team to usable format
teams <- unique(nba_data_cleaned$team)
for (team in teams) {
  nba_data_cleaned <- nba_data_cleaned %>%
    mutate(!!paste0("team_", team) := team == nba_data_cleaned$team)
}

# Select predictor variables and target variable
predictor_vars <- select(nba_data_cleaned, -c(age, box_plus_minus))
target_var <- "age"  # Change to "box_plus_minus" if needed

# Split data into training and testing sets
set.seed(123)
train_indices <- sample(1:nrow(nba_data_cleaned), 0.8 * nrow(nba_data_cleaned))
train_data <- predictor_vars[train_indices, ]
train_target <- nba_data_cleaned[train_indices, ][[target_var]]
test_data <- predictor_vars[-train_indices, ]
test_target <- nba_data_cleaned[-train_indices, ][[target_var]]

# Train a model (example: Linear Regression)
model <- lm(train_target ~ ., data = train_data)

# Make predictions on test data
predictions <- predict(model, newdata = test_data)

# Evaluate model performance (example: Mean Absolute Error)
mae <- mean(abs(predictions - test_target))
print(paste("Mean Absolute Error:", mae))
