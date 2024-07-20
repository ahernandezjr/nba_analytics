# File to convert data file to SQL database
import os
import numpy as np
import pandas as pd

import urllib
from sqlalchemy import create_engine, Float, String

from . import reduction

from ..utils import filename_grabber
from ..utils.config import settings
from ..utils.logger import get_logger


driver = "{ODBC Driver 17 for SQL Server}"
server = str.join("", [settings.cloud.SQL_SERVER_NAME, settings.cloud.SQL_SERVER_DOMAIN])
database = settings.cloud.SQL_DATABASE_NAME
user = settings.cloud.SQL_USERNAME
password = settings.cloud.SQL_PASSWORD


logger = get_logger(__name__)

# class AzureDbConnection:
#     """
#     Azure SQL database connection.
#     """
#     def __init__(self, conn_settings: ConnectionSettings, echo: bool = False) -> None:
#         conn_params = urllib.parse.quote_plus(
#             'Driver=%s;' % conn_settings.driver +
#             'Server=tcp:%s.database.windows.net,1433;' % conn_settings.server +
#             'Database=%s;' % conn_settings.database +
#             'Uid=%s;' % conn_settings.username +
#             'Pwd=%s;' % conn_settings.password +
#             'Encrypt=yes;' +
#             'TrustServerCertificate=no;' +
#             'Connection Timeout=%s;' % conn_settings.timeout
#         )
#         conn_string = f'mssql+pyodbc:///?odbc_connect={conn_params}'

#         self.db = create_engine(conn_string, echo=echo)

#     def connect(self) -> None:
#         """Estimate connection."""
#         self.conn = self.db.connect()

#     def get_tables(self) -> Iterable[str]:
#         """Get list of tables."""
#         inspector = inspect(self.db)
#         return [t for t in inspector.get_table_names()]

#     def dispose(self) -> None:
#         """Dispose opened connections."""
#         self.conn.close()
#         self.db.dispose()

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