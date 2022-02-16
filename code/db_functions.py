"""
Database setup and useful functions.
Based on: Imke Lansky (https://github.com/ImkeLansky/USA-BuildingHeightInference)

"""

from config import config
import sys
import psycopg2
import psycopg2.extras
import pandas as pd
import geopandas as gpd


def setup_connection():
    """
    Set up the connection to a given database provided in config file.

    Returns: PostgreSQL database connection 

    """
    
    try:
        # Read connection parameters 
        params = config()

        # Connect to PostgreSQL server
        print("\n>> Connecting to PostgreSQL database: {0}".format(params["database"]))
        return psycopg2.connect(**params)
        
    except (Exception, psycopg2.Error) as error:
        print("Error while connecting to PostgreSQL;", error)
        sys.exit()


def close_connection(connection, cursor):
    """
    Close connection to the database and cursor used to perform queries.

    Parameters: \n
    connection -- database connection \n
    cursor -- cursor for database connection \n

    """

    if cursor:
        cursor.close()
        print("\n>> Cursor is closed")

    if connection:
        connection.close()
        print("\n>> PostgreSQL connection is closed")


def read_data(connection, schema, table, where='', columns='*'):
    """
    Read data from database into pandas DataFrame.

    Parameters: \n
    connection -- database connection \n
    schema -- schema to read data from \n
    table -- table containing data inside schema \n
    where -- optional where clause \n
    columns -- columns to select (default = all columns) \n

    Returns: pandas DataFrame containing all data from the specified table

    """

    print("\n>> Reading data into pandas DataFrame")

    query = "SELECT " + columns + " FROM " + schema + "." + table + " " + where + ";"
    data = pd.read_sql_query(query, connection)
     
    return data


def read_spatial_data(connection, schema, table, geom_col='geom', where='', columns='*'): 
    """
    Read spatial data from database into pandas DataFrame.

    Parameters: \n
    connection -- database connection \n
    schema -- schema to read data from \n
    table -- table containing data inside schema \n
    geom_col -- geometry column (default = geom)
    where -- optional where clause (default = empty string) \n
    columns -- columns to select (default = all columns) \n

    Returns: geopandas DataFrame containing all data from the specified table

    """

    print("\n>> Reading data into geopandas DataFrame")

    query = "SELECT " + columns + " FROM " + schema + "." + table + " " + where + ";"
    data = gpd.GeoDataFrame.from_postgis(query, connection, geom_col)
     
    return data


def read_spatial_subset(connection, schema, table, wkt_geom, geom_col='geom', columns='*', extra_where='', srid='28992'): 
    """
    Read subset of data from database into geopandas DataFrame based on the intersection with input polygon geometry.

    Parameters: \n
    connection -- database connection \n
    schema -- schema to read data from \n
    table -- table containing data inside schema \n
    wkt_geom -- wkt geometry to perform intersection with (polygon or multipolygon) \n
    geom_col -- geometry column (default = geom)
    columns -- columns to select (default = all columns) \n
    extra_where -- any additional "WHERE" conditions, must start with "AND" (default = empty string) \n
    srid -- spatial reference system (default = EPSG:28992) \n

    Returns: geopandas DataFrame containing all data from the specified table that intersects with input polygon

    """

    print("\n>> Reading subset of data into geopandas DataFrame")

    query = "SELECT " + columns + " FROM " + schema + "." + table + " WHERE ST_intersects(ST_SetSRID(" + table + "." + geom_col + ", " + srid + "), ST_GeomFromText('" + wkt_geom + "', " + srid + ")) " + extra_where + ";"

    data = gpd.GeoDataFrame.from_postgis(query, connection, geom_col)
     
    return data


def unique_tables(cursor, schema):
    """
    Find all tables in a database schema. 

    Parameters: \n
    cursor -- cursor for database connection \n
    schema -- name of database schema \n

    Returns: set containing table names in the specified database schema

    """

    cursor.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = %s;", (schema,))

    all_tables = set(list(zip(*cursor.fetchall()))[0])

    return all_tables 


def create_temp_table(cursor, schema, table, pkey=None): 
    """
    Create a temporary table to extract the data into as copy of original table.

    Parameters: \n
    cursor -- cursor for database connection \n
    schema -- schema to store the data in the database \n
    table -- table to store the data in the database \n
    pkey -- column to create primary key on (optional) \n

    Returns: none

    """

    print('\n>> Dataset {0} -- creating temporary unlogged table'.format(table))

    cursor.execute("DROP TABLE IF EXISTS " + schema + "." + table + "_tmp;") # CASCADE? 
    cursor.execute("CREATE UNLOGGED TABLE " + schema + "." + table + "_tmp AS TABLE " + schema + "." + table + ";")

    if pkey is not None: 
        try: 
            cursor.execute("ALTER TABLE " + schema + "." + table + "_tmp ADD PRIMARY KEY (" + pkey + ");")
        except Exception as error:
            print('\nError: {0}'.format(str(error)))
    else: 
        pass


def replace_temp_table(cursor, schema, table, pkey=None, geom_index=None): 
    """
    Replace original table with temporary table containing extracted data, drop temporary table and create (optional) indexes on new table. 

    Parameters: \n
    cursor -- cursor for database connection \n
    schema -- schema to store the data in the database \n
    table -- table to store the data in the database \n
    pkey -- column to create primary key on (optional) \n
    geom_index -- column to create geometry index on (optional) \n

    Returns: none
    
    """

    print('\n>> Dataset {0} -- copying unlogged table to logged table'.format(table))
    cursor.execute("CREATE TABLE " + schema + "." + table + "_new AS TABLE " + schema + "." + table + "_tmp;")
    cursor.execute("DROP TABLE " + schema + "." + table + ";")
    cursor.execute("ALTER TABLE " + schema + "." + table + "_new RENAME TO " + table + ";")
    cursor.execute("DROP TABLE " + schema + "." + table + "_tmp;")
    
    if pkey is not None: 
        try: 
            cursor.execute("ALTER TABLE " + schema + "." + table + " ADD PRIMARY KEY (" + pkey + ");")
        except Exception as error:
            print('\nError: {0}'.format(str(error)))
    else: 
        pass

    if geom_index is not None: 
        try: 
            cursor.execute("CREATE INDEX IF NOT EXISTS " + table + "_" + geom_index + "_idx ON " + schema + "." + table + " USING GIST (" + geom_index + ");")
        except Exception as error:
            print('\nError: {0}'.format(str(error)))
    else: 
        pass
    
