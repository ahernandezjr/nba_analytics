import os
import numpy as np
import pandas as pd
import pyodbc

from . import filtering

from ...utils import filename_grabber
from ...utils.config import settings, azure
from ...utils.logger import get_logger


# Create logger
logger = get_logger(__name__)


# Connect to Azure SQL Database
conn_str = f"DRIVER={{ODBC Driver 17 for SQL Server}};\
                SERVER={settings.azure.SQL_SERVER_NAME}{settings.azure.SQL_SERVER_DOMAIN};\
                DATABASE={settings.azure.SQL_DATABASE_NAME};\
                UID={settings.azure.SQL_USERNAME};\
                PWD={settings.azure.SQL_PASSWORD}"
conn = pyodbc.connect(conn_str)

# Create a cursor object
cursor = conn.cursor()

# Create table based on nba_player_data.csv
df = pd.read_csv(filename_grabber.get_data_file)
table_name = 'nba_player_data'
create_table_query = f"CREATE TABLE {table_name} ("
for column in df.columns:
    column_name = column.replace(' ', '_')
    column_type = df[column].dtype
    if column_type == 'int64':
        column_type = 'INT'
    elif column_type == 'float64':
        column_type = 'FLOAT'
    elif column_type == 'object':
        column_type = 'VARCHAR(255)'
    elif column_type == 'bool':
        column_type = 'BIT'
    elif column_type == 'string':
        column_type = 'VARCHAR'
    create_table_query += f"{column_name} {column_type}, "
create_table_query = create_table_query.rstrip(', ') + ")"
cursor.execute(create_table_query)

# Insert data into the table
insert_query = f"INSERT INTO {table_name} VALUES ({', '.join(['?'] * len(df.columns))})"
cursor.executemany(insert_query, df.values.tolist())

# Commit the changes and close the connection
conn.commit()
conn.close()