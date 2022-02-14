"""
Functions to calculate number of floors using height-based and area-based approaches. 

"""

import json, sys
import db_functions


def height_based(cursor, schema, table, ceiling_height):
    """
    Calculates number of floors using height-based approach and stores results in database. 

    Parameters: \n
    cursor -- cursor for database connection \n
    schema -- schema to store results in database \n
    table -- table to store the results in the database \n
    ceiling_height -- floor to ceiling height used to calculate # floors from building height

    Returns: none

    """

    cursor.execute("ALTER TABLE " + schema + "." + table + " DROP COLUMN IF EXISTS floors_height_based;")
    cursor.execute("ALTER TABLE " + schema + "." + table + " ADD COLUMN floors_height_based INTEGER;")
    cursor.execute("ALTER TABLE " + schema + "." + table + " DROP COLUMN IF EXISTS floors_height_based_unrounded;")
    cursor.execute("ALTER TABLE " + schema + "." + table + " ADD COLUMN floors_height_based_unrounded REAL;")

    cursor.execute(
        "UPDATE " + schema + "." + table + " " + 
        "SET floors_height_based = subquery.floors_height_based, " +
        "floors_height_based_unrounded = subquery.floors_height_based_unrounded " + 
        "FROM " + 
            "(SELECT bag_id, " + 
            "ROUND(h_70p / " + str(ceiling_height) + ") AS floors_height_based, " + 
            "(h_70p / " + str(ceiling_height) + ") AS floors_height_based_unrounded " + 
            "FROM " + schema + "." + table + ") AS subquery " +
        "WHERE " + schema + "." + table + ".bag_id = subquery.bag_id;"
    )


def area_based(cursor, schema, table):
    """
    Calculates number of floors using area-based approach and stores results in database. 

    Parameters: \n
    cursor -- cursor for database connection \n
    schema -- schema to store results in database \n
    table -- table to store the results in the database \n

    Returns: none

    """

    cursor.execute("ALTER TABLE " + schema + "." + table + " DROP COLUMN IF EXISTS floors_area_based;")
    cursor.execute("ALTER TABLE " + schema + "." + table + " ADD COLUMN floors_area_based INTEGER;")
    cursor.execute("ALTER TABLE " + schema + "." + table + " DROP COLUMN IF EXISTS floors_area_based_unrounded;")
    cursor.execute("ALTER TABLE " + schema + "." + table + " ADD COLUMN floors_area_based_unrounded REAL;")

    cursor.execute(
        "UPDATE " + schema + "." + table + " " + 
        "SET floors_area_based = subquery.floors_area_based, " +
        "floors_area_based_unrounded = subquery.floors_area_based_unrounded " +
        "FROM " + 
            "(SELECT bag_id, ROUND(bag_net_internal_area/footprint_area + 0.4) AS floors_area_based, " + 
            "bag_net_internal_area/footprint_area AS floors_area_based_unrounded " + 
            "FROM " + schema + "." + table + ") AS subquery " +
        "WHERE " + schema + "." + table + ".bag_id = subquery.bag_id;"
    )


def ref_model(cursor, schema, table):
    """
    Compute reference model from height- and area-based results. 
    Reference model equals the height-based estimate except if this is unavailable, 
    then reference model equals the area-based estimate.  

    Parameters: \n
    cursor -- cursor for database connection \n
    schema -- schema to store results in database \n
    table -- table to store the results in the database \n

    Returns: none

    """

    cursor.execute("ALTER TABLE " + schema + "." + table + " DROP COLUMN IF EXISTS ref_model;")
    cursor.execute("ALTER TABLE " + schema + "." + table + " ADD COLUMN ref_model INTEGER;")
    cursor.execute("ALTER TABLE " + schema + "." + table + " DROP COLUMN IF EXISTS ref_model_unrounded;")
    cursor.execute("ALTER TABLE " + schema + "." + table + " ADD COLUMN ref_model_unrounded REAL;")
 
    cursor.execute(
        "UPDATE " + schema + "." + table + " " + 
        "SET ref_model = subquery.ref_model, " +
        "ref_model_unrounded = subquery.ref_model_unrounded " + 
        "FROM " + 
            "(SELECT bag_id, " + 
            "CASE " + 
            "WHEN floors_height_based IS NULL THEN floors_area_based " + 
            "WHEN floors_height_based IS NOT NULL THEN floors_height_based " + 
            "END AS ref_model, " + 
            "CASE " + 
            "WHEN floors_height_based_unrounded IS NULL THEN floors_area_based_unrounded " + 
            "WHEN floors_height_based_unrounded IS NOT NULL THEN floors_height_based_unrounded " + 
            "END AS ref_model_unrounded " + 
            "FROM " + schema + "." + table + ") AS subquery " +
        "WHERE " + schema + "." + table + ".bag_id = subquery.bag_id;"
    )


def main(params): 

    # Load parameters 
    jparams = json.load(open(params))

    # Create connection to database
    conn = db_functions.setup_connection()
    conn.autocommit = True

    # Create a cursor (allows PostgreSQL commands to be executed)
    curs = conn.cursor()

    train_tables = jparams["training_tables"]
    train_schema = jparams["training_schema"]

    for table in train_tables:

        print('\n>> Dataset {0} -- calculating reference # floors'.format(table))

        ceiling_height = jparams["ceiling_height"]
        
        # Calculate # floors from height-based approach per LOD
        height_based(curs, train_schema, table, ceiling_height)

        # Calculate # floors from area-based approach
        area_based(curs, train_schema, table)

        # Create reference model (height-based when available otherwise area-based)
        ref_model(curs, train_schema, table)

    # Close database connection
    db_functions.close_connection(conn, curs)


if __name__ == '__main__':
    main(sys.argv[1])
