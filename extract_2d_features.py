"""
Functions to extract and store features based on the 2D footprint of each building in the database. 
Based on: Imke Lansky (https://github.com/ImkeLansky/USA-BuildingHeightInference)

"""

import json, sys
from time import time
import geopandas as gpd
import db_functions
import geom_functions
import retrieve_data


def get_all_buildings(cursor, schema, table, wkt_geom): 
    """
    Get all building IDs that intersect with input WKT geometry and store results in database. 

    Parameters: \n
    cursor -- cursor for database connection \n
    schema -- schema to store the building IDs in the database \n
    table -- table to store the building IDs in the database \n
    wkt_geom -- WKT geometry representation of polygon to perform intersection with \n

    Returns: none
    
    """

    wkt_geom = gpd.array.to_wkt(wkt_geom)[0]

    print('\n>> Dataset {0} -- creating table containing all buildings inside buffer'.format(table))
    cursor.execute("DROP TABLE IF EXISTS " + schema + "." + table + ";")
    cursor.execute(
        "CREATE TABLE " + schema + "." + table + " AS " +  
            "(SELECT DISTINCT ON (identificatie) identificatie AS bag_id, geom AS footprint_geom " + 
            "FROM bag.pand " +
            "WHERE ST_intersects(pand.geom, ST_GeomFromText('" + wkt_geom + "', 28992)) " + 
            "AND einddatumtijdvakgeldigheid IS NULL " 
            "AND pandstatus LIKE '%Pand in gebruik%')" 
    )


def get_all_buildings_attribs(cursor, schema, table, wkt_geom):
    """
    Get all building footprints that intersect with the input WKT geometry,
    obtain the function of these building, compute the centroid of each footprint 
    and store the results in the database. 

    Parameters: \n
    cursor -- cursor for database connection \n
    schema -- schema to store the building footprints and attributes in the database \n
    table -- table to store the building footprints and attributes in the database \n
    wkt_geom -- WKT geometry representation of polygon to perform intersection with \n

    Returns: none

    """
    
    # Get all buildings within specified buffer of data 
    get_all_buildings(cursor, schema, table, wkt_geom)

    # Create temporary table to store extracted data in
    db_functions.create_temp_table(cursor, schema, table, pkey='bag_id')

    # Get function of all buildings 
    retrieve_data.get_function(cursor, schema, table, filter_input=wkt_geom)

    # Compute footprint centroids of all buildings 
    geom_functions.compute_centroids(cursor, schema, table)

    # Replace original table with unlogged temporary table
    db_functions.replace_temp_table(cursor, schema, table, pkey='bag_id', geom_index='footprint_geom')


def get_footprint_area(cursor, schema, table):
    """
    Compute footprint area and store results in database. 

    Parameters: \n
    cursor -- cursor for database connection \n
    schema -- schema to store results in database \n
    table -- table to store results in database \n

    Returns: none

    """

    print('\n>> Dataset {0} -- obtaining footprint area'.format(table))

    cursor.execute("ALTER TABLE " + schema + "." + table + "_tmp ADD COLUMN IF NOT EXISTS footprint_area REAL;")

    cursor.execute(
        "UPDATE " + schema + "." + table + "_tmp " + 
        "SET footprint_area = subquery.area " + 
        "FROM " + 
            "(SELECT bag_id, ST_Area(footprint_geom) AS area " + 
            "FROM " + schema + "." + table + "_tmp) AS subquery " + 
        "WHERE " + schema + "." + table + "_tmp.bag_id = subquery.bag_id;"
    )


def get_footprint_perim(cursor, schema, table):
    """
    Compute footprint perimeter and store results in database. 

    Parameters: \n
    cursor -- cursor for database connection \n
    schema -- schema to store results in database \n
    table -- table to store results in database \n

    Returns: none

    """

    print('\n>> Dataset {0} -- obtaining footprint perimeter'.format(table))

    cursor.execute("ALTER TABLE " + schema + "." + table + "_tmp ADD COLUMN IF NOT EXISTS footprint_perim REAL;")

    cursor.execute(
        "UPDATE " + schema + "." + table + "_tmp " + 
        "SET footprint_perim = subquery.perim " + 
        "FROM " + 
            "(SELECT bag_id, ST_PERIMETER(footprint_geom) AS perim " + 
            "FROM " + schema + "." + table + "_tmp) AS subquery " + 
        "WHERE " + schema + "." + table + "_tmp.bag_id = subquery.bag_id;"
    )


def get_footprint_vertices(cursor, schema, table):
    """
    Compute no. of footprint vertices after simplification using Douglas-Peucker and store results in database. 

    Parameters: \n
    cursor -- cursor for database connection \n
    schema -- schema to store results in database \n
    table -- table to store results in database \n

    Returns: none

    """

    print('\n>> Dataset {0} -- obtaining footprint vertices count'.format(table))

    cursor.execute("ALTER TABLE " + schema + "." + table + "_tmp ADD COLUMN IF NOT EXISTS footprint_no_vertices REAL;")

    cursor.execute(
        "UPDATE " + schema + "." + table + "_tmp " + 
        "SET footprint_no_vertices = subquery.no_vertices " + 
        "FROM " + 
            "(SELECT bag_id, ST_NPoints(ST_SimplifyPreserveTopology(footprint_geom, 0.1)) AS no_vertices " + 
            "FROM " + schema + "." + table + "_tmp) AS subquery " + 
        "WHERE " + schema + "." + table + "_tmp.bag_id = subquery.bag_id;"
    )


def get_adjacent_count(cursor, schema1, table1, schema2, table2, adjacent_distance): 
    """
    Get number of adjacent buildings of each building footprint and store results in the database. 

    Parameters: \n
    cursor -- cursor for database connection \n
    schema1 -- schema to store the features in the database \n
    table1 -- table to store the features in the database \n
    schema2 -- schema containing buildings to intersect with \n
    table2 -- table containing buildings to intersect with \n
    adjacent_distance -- distance to adjacent buildings \n

    Returns: none
    
    """
    
    print('\n>> Dataset {0} -- obtaining no. adjacent buildings from footprints'.format(table1))

    # Compute buffer around footprints
    geom_functions.compute_buffers(cursor, schema1, table1, adjacent_distance)

    cursor.execute("ALTER TABLE " + schema1 + "." + table1 + "_tmp ADD COLUMN IF NOT EXISTS footprint_no_adjacent INTEGER;")

    # Extract number of adjacent buildings based on buffer
    cursor.execute(
        "UPDATE " + schema1 + "." + table1 + "_tmp " +
        "SET footprint_no_adjacent = subquery.no_adjacent " + 
        "FROM " + 
            "(SELECT a.bag_id, COUNT(*) AS no_adjacent " + 
            "FROM " + schema1 + "." + table1 + "_tmp AS a " + 
            "JOIN " + schema2 + "." + table2 + " AS b ON ST_INTERSECTS(a.footprint_buffer, b.footprint_geom) " + 
            "WHERE a.bag_id != b.bag_id " + 
            "AND a.bag_function != 'Others' AND a.bag_function != 'Unknown' " + 
            "AND b.bag_function != 'Others' AND b.bag_function != 'Unknown' " + 
            "GROUP BY a.bag_id) AS subquery " + 
        "WHERE " + schema1 + "." + table1 + "_tmp.bag_id = subquery.bag_id;"
    )
    
    # Set number of adjacent buildings equal to zero when column is null
    # (except from when footprint geometry is equal to null)
    cursor.execute(
        "UPDATE " + schema1 + "." + table1 + "_tmp " + 
        "SET footprint_no_adjacent = 0 " + 
        "WHERE footprint_no_adjacent IS NULL "
        "AND footprint_geom IS NOT NULL;"
    )

    # Drop the buffer column
    cursor.execute("ALTER TABLE " + schema1 + "." + table1 + "_tmp DROP COLUMN footprint_buffer;")


def get_neighbour_count(cursor, schema1, table1, schema2, table2, neighbour_distances): 
    """
    Get number of neighbouring building centroids at different distances from 
    each building footprint centroid and store results in the database. 

    Parameters: \n
    cursor -- cursor for database connection \n
    schema1 -- schema to store the features in the database \n
    table1 -- table to store the features in the database \n
    schema2 -- schema containing buildings to intersect with \n
    table2 -- table containing buildings to intersect with \n
    neighbour_distances -- list of distances to neighbouring building centroids \n

    Returns: none

    """

    print('\n>> Dataset {0} -- obtaining no. neighbouring buildings from footprints'.format(table1))

    # Compute footprint centroid 
    geom_functions.compute_centroids(cursor, schema1, table1)

    # Extract number of neightbours based on centroid 
    for dist in neighbour_distances: 
        cursor.execute("ALTER TABLE " + schema1 + "." + table1 + "_tmp ADD COLUMN IF NOT EXISTS footprint_no_neighbours_" + str(dist) + "m INTEGER;")

        cursor.execute(
            "UPDATE " + schema1 + "." + table1 + "_tmp " + 
            "SET footprint_no_neighbours_" + str(dist) + "m = subquery.no_neighbours " + 
            "FROM " + 
                "(SELECT a.bag_id, COUNT(*) AS no_neighbours " + 
                "FROM " + schema1 + "." + table1 + "_tmp AS a " + 
                "JOIN " + schema2 + "." + table2 + " AS b " + 
                "ON ST_DWithin(a.footprint_centroid, b.footprint_centroid, " + str(dist) + ") " + 
                "WHERE a.bag_id != b.bag_id " + 
                "AND a.bag_function != 'Others' AND a.bag_function != 'Unknown' " + 
                "AND b.bag_function != 'Others' AND b.bag_function != 'Unknown' " + 
                "GROUP BY a.bag_id) AS subquery " +
            "WHERE " + schema1 + "." + table1 + "_tmp.bag_id = subquery.bag_id;"
        )

        # Set number of neighbouring buildings equal to zero when column is null
        # (except from when footprint geometry is equal to null)
        cursor.execute(
            "UPDATE " + schema1 + "." + table1 + "_tmp " + 
            "SET footprint_no_neighbours_" + str(dist) + "m = 0 " + 
            "WHERE footprint_no_neighbours_" + str(dist) + "m IS NULL "
            "AND footprint_geom IS NOT NULL;"
        )
    
    # Drop the centroid column
    cursor.execute("ALTER TABLE " + schema1 + "." + table1 + "_tmp DROP COLUMN footprint_centroid;")
    cursor.execute("ALTER TABLE " + schema2 + "." + table2 + " DROP COLUMN footprint_centroid;")


def main(params):

    # Load parameters 
    jparams = json.load(open(params))

    # Create connection to database
    conn = db_functions.setup_connection()
    conn.autocommit = True

    # Create a cursor (allows PostgreSQL commands to be executed)
    curs = conn.cursor()

    for table in jparams["training_tables"]:

        starttime = time()

        # Parameters
        gm_code = jparams["gemeente_codes"][table]
        dist_adjacent= jparams["distance_adjacent"]
        dist_neighbours = jparams["distance_neighbours"]

        train_schema = 'training_data'

        # Create temporary table to store extracted data in
        db_functions.create_temp_table(curs, train_schema, table, pkey='bag_id')

        # Get area, perimeter and vertices from footprint geometry 
        get_footprint_area(curs, train_schema, table)
        get_footprint_perim(curs, train_schema, table)
        get_footprint_vertices(curs, train_schema, table)

        # Extract and store required attributes of all buildings within 300m buffer of municipality 
        # (so that no. adjacent and neighbouring buildings can be obtained correctly)
        all_build_table = table + "_all"
        wkt_gm_buffer = geom_functions.get_gemeente_polygon(conn, gm_code, buffer=300)
        get_all_buildings_attribs(curs, train_schema, all_build_table, wkt_gm_buffer)

        # Get no. adjacent and neighbouring buildings 
        get_adjacent_count(curs, train_schema, table, train_schema, all_build_table, dist_adjacent)
        get_neighbour_count(curs, train_schema, table, train_schema, all_build_table, dist_neighbours)

        # Replace original table with unlogged temporary table
        db_functions.replace_temp_table(curs, train_schema, table, pkey='bag_id')

        # Drop table containing all buildings
        curs.execute("DROP TABLE " + train_schema + "." + all_build_table + ";")

        endtime = time()
        duration = endtime - starttime
        print('\n>> Computation time: ', round(duration, 2), 's \n\n' + 10*'-')

    # Close database connection
    db_functions.close_connection(conn, curs)


if __name__ == '__main__':
    main(sys.argv[1])