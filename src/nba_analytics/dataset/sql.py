# File to convert data file to SQL database
import os
import numpy as np
import pandas as pd
import csv

import urllib

from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, ForeignKey
from sqlalchemy.orm import relationship, sessionmaker, declarative_base

# from ...dataset import reduction

from ..utils import filename_grabber
from ..utils.config import settings
from ..utils.logger import get_logger


driver = "{ODBC Driver 17 for SQL Server}"
server = str.join("", [settings.cloud.SQL_SERVER_NAME, settings.cloud.SQL_SERVER_DOMAIN])
database = settings.cloud.SQL_DATABASE_NAME
user = settings.cloud.SQL_USERNAME
password = settings.cloud.SQL_PASSWORD


logger = get_logger(__name__)


def get_engine():
    """
    Create a SQL engine
    """
    logger.info(f"Creating SQL engine with the following parameters:\
                \nDriver: {driver}\
                \nServer: {server}\
                \nDatabase: {database}\
                \nUser: {user}\
                \nPassword: _password_")


    conn = f"""Driver={driver};Server=tcp:{server},1433;Database={database};
    Uid={user};Pwd={password};Encrypt=yes;TrustServerCertificate=no;Connection Timeout=30;"""
    
    params = urllib.parse.quote_plus(conn)
    conn_str = 'mssql+pyodbc:///?autocommit=true&odbc_connect={}'.format(params)

    engine = create_engine(conn_str, echo=True)

    return engine

def create_sql_table(data_file_name, table_name):
    """
    Create a SQL table from a data file
    """
    # Get engine
    engine = get_engine()

    # Get the data file
    data_file_path = filename_grabber.get_any_file(data_file_name)
    data = pd.read_csv(data_file_path)

    # Get the columns and data types
    columns = data.columns
    data_types = [Float if np.issubdtype(data[col].dtype, np.number) else String for col in columns]
    
    sql_table = pd.DataFrame(columns=columns)
    sql_table.to_sql(table_name, con=engine, if_exists='replace', index=False, dtype=dict(zip(columns, data_types)))

    logger.info(f"Created SQL table {table_name} from {data_file_name}")

    # Insert data into the table
    data.to_sql(table_name, con=engine, if_exists='append', index=False)

    logger.info(f"Inserted data into SQL table {table_name}")

    # Close the engine
    engine.dispose()
    
    return sql_table


def split_tables():
    ... # TODO: Seperate tables by relevant information


def view_table(table_name):
    """
    View a SQL table
    """
    # Get engine
    engine = get_engine()

    # Get the data
    data = pd.read_sql(f"SELECT * FROM {table_name}", con=engine)

    # Close the engine
    engine.dispose()

    return data



# TODO: Below
Base = declarative_base()

engine = get_engine()
Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)
session = Session()

# Defining the tables
class Player(Base):
    __tablename__ = 'players'
    player_id = Column(Integer, primary_key=True, autoincrement=True)
    slug = Column(String, unique=True, nullable=False)
    name = Column(String, nullable=False)
    positions = Column(String)
    age = Column(Integer)

    basic_stats = relationship('BasicStat', back_populates='player')
    advanced_stats = relationship('AdvancedStat', back_populates='player')

class Team(Base):
    __tablename__ = 'teams'
    team_id = Column(Integer, primary_key=True, autoincrement=True)
    team_name = Column(String, unique=True, nullable=False)

    basic_stats = relationship('BasicStat', back_populates='team')
    advanced_stats = relationship('AdvancedStat', back_populates='team')

class BasicStat(Base):
    __tablename__ = 'basic_stats'
    basic_stat_id = Column(Integer, primary_key=True, autoincrement=True)
    player_id = Column(Integer, ForeignKey('players.player_id'))
    team_id = Column(Integer, ForeignKey('teams.team_id'))
    games_played = Column(Integer)
    games_started = Column(Integer)
    minutes_played = Column(Integer)
    made_field_goals = Column(Integer)
    attempted_field_goals = Column(Integer)
    made_three_point_field_goals = Column(Integer)
    attempted_three_point_field_goals = Column(Integer)
    made_free_throws = Column(Integer)
    attempted_free_throws = Column(Integer)
    offensive_rebounds = Column(Integer)
    defensive_rebounds = Column(Integer)
    assists = Column(Integer)
    steals = Column(Integer)
    blocks = Column(Integer)
    turnovers = Column(Integer)
    personal_fouls = Column(Integer)
    points = Column(Integer)
    year = Column(Integer)

    player = relationship('Player', back_populates='basic_stats')
    team = relationship('Team', back_populates='basic_stats')

class AdvancedStat(Base):
    __tablename__ = 'advanced_stats'
    advanced_stat_id = Column(Integer, primary_key=True, autoincrement=True)
    player_id = Column(Integer, ForeignKey('players.player_id'))
    team_id = Column(Integer, ForeignKey('teams.team_id'))
    games_played = Column(Integer)
    minutes_played = Column(Integer)
    player_efficiency_rating = Column(Float)
    true_shooting_percentage = Column(Float)
    three_point_attempt_rate = Column(Float)
    free_throw_attempt_rate = Column(Float)
    offensive_rebound_percentage = Column(Float)
    defensive_rebound_percentage = Column(Float)
    total_rebound_percentage = Column(Float)
    assist_percentage = Column(Float)
    steal_percentage = Column(Float)
    block_percentage = Column(Float)
    turnover_percentage = Column(Float)
    usage_percentage = Column(Float)
    offensive_win_shares = Column(Float)
    defensive_win_shares = Column(Float)
    win_shares = Column(Float)
    win_shares_per_48_minutes = Column(Float)
    offensive_box_plus_minus = Column(Float)
    defensive_box_plus_minus = Column(Float)
    box_plus_minus = Column(Float)
    value_over_replacement_player = Column(Float)
    is_combined_totals = Column(Boolean)
    year = Column(Integer)

    player = relationship('Player', back_populates='advanced_stats')
    team = relationship('Team', back_populates='advanced_stats')


# CSV to SQL Functions
def insert_player(session, slug, name, positions, age):
    player = session.query(Player).filter_by(slug=slug).first()
    if not player:
        player = Player(slug=slug, name=name, positions=positions, age=age)
        session.add(player)
        session.commit()
    return player.player_id

def insert_team(session, team_name):
    team = session.query(Team).filter_by(team_name=team_name).first()
    if not team:
        team = Team(team_name=team_name)
        session.add(team)
        session.commit()
    return team.team_id

def insert_basic_stats(session, row):
    player_id = insert_player(session, row['slug'], row['name'], row['positions'], row['age'])
    team_id = insert_team(session, row['team'])
    basic_stat = BasicStat(
        player_id=player_id,
        team_id=team_id,
        games_played=row['games_played'],
        games_started=row['games_started'],
        minutes_played=row['minutes_played'],
        made_field_goals=row['made_field_goals'],
        attempted_field_goals=row['attempted_field_goals'],
        made_three_point_field_goals=row['made_three_point_field_goals'],
        attempted_three_point_field_goals=row['attempted_three_point_field_goals'],
        made_free_throws=row['made_free_throws'],
        attempted_free_throws=row['attempted_free_throws'],
        offensive_rebounds=row['offensive_rebounds'],
        defensive_rebounds=row['defensive_rebounds'],
        assists=row['assists'],
        steals=row['steals'],
        blocks=row['blocks'],
        turnovers=row['turnovers'],
        personal_fouls=row['personal_fouls'],
        points=row['points'],
        year=row['Year']
    )
    session.add(basic_stat)
    session.commit()

def insert_advanced_stats(session, row):
    player_id = insert_player(session, row['slug'], row['name'], row['positions'], row['age'])
    team_id = insert_team(session, row['team'])
    advanced_stat = AdvancedStat(
        player_id=player_id,
        team_id=team_id,
        games_played=row['games_played'],
        minutes_played=row['minutes_played'],
        player_efficiency_rating=row['player_efficiency_rating'],
        true_shooting_percentage=row['true_shooting_percentage'],
        three_point_attempt_rate=row['three_point_attempt_rate'],
        free_throw_attempt_rate=row['free_throw_attempt_rate'],
        offensive_rebound_percentage=row['offensive_rebound_percentage'],
        defensive_rebound_percentage=row['defensive_rebound_percentage'],
        total_rebound_percentage=row['total_rebound_percentage'],
        assist_percentage=row['assist_percentage'],
        steal_percentage=row['steal_percentage'],
        block_percentage=row['block_percentage'],
        turnover_percentage=row['turnover_percentage'],
        usage_percentage=row['usage_percentage'],
        offensive_win_shares=row['offensive_win_shares'],
        defensive_win_shares=row['defensive_win_shares'],
        win_shares=row['win_shares'],
        win_shares_per_48_minutes=row['win_shares_per_48_minutes'],
        offensive_box_plus_minus=row['offensive_box_plus_minus'],
        defensive_box_plus_minus=row['defensive_box_plus_minus'],
        box_plus_minus=row['box_plus_minus'],
        value_over_replacement_player=row['value_over_replacement_player'],
        is_combined_totals=row['is_combined_totals'],
        year=row['Year']
    )
    session.add(advanced_stat)
    session.commit()

# Insert data from CSV files
def csv_to_sql():
    with open('nba_players_basic.csv', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            insert_basic_stats(session, row)

    with open('nba_players_advanced.csv', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            insert_advanced_stats(session, row)
