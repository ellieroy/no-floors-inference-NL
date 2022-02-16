"""
Standardises training data, performs preliminary cleaning and stores results in database. 

"""

import json, sys
import db_functions
from time import time


def get_floor_count(cursor, table):
    """
    Retrieve the number of floors from the raw data, standardise the training data structure and store results in database. 

    Preliminary data cleaning is also performed.

    Parameters: \n
    cursor -- cursor for database connection \n
    table -- table to store the standardised training data in the database \n

    Returns: none

    """

    print("\n>> Dataset {0} -- standardising training data".format(table))
    cursor.execute("DROP TABLE IF EXISTS training_data." + table + ";")

    if table == 'c1_ams': 

        cursor.execute(
            "CREATE TABLE training_data.c1_ams AS " + 
                "(SELECT LPAD(identificatie::varchar, 16, '0') AS bag_id, " + 
                "CASE " + 
                    "WHEN aantalbouwlagen - ABS(laagstebouwlaag) = hoogstebouwlaag + 1 AND aantalbouwlagen IS NOT NULL AND hoogstebouwlaag > laagstebouwlaag THEN aantalbouwlagen - ABS(laagstebouwlaag) " + 
                    "WHEN aantalbouwlagen - ABS(laagstebouwlaag) = hoogstebouwlaag + 1 AND aantalbouwlagen IS NOT NULL AND hoogstebouwlaag < laagstebouwlaag THEN aantalbouwlagen - ABS(hoogstebouwlaag) " + 
                    "WHEN hoogstebouwlaag IS NULL AND laagstebouwlaag IS NULL AND aantalbouwlagen IS NOT NULL THEN aantalbouwlagen " + 
                    "WHEN hoogstebouwlaag IS NULL AND laagstebouwlaag >= 0 AND aantalbouwlagen IS NOT NULL THEN aantalbouwlagen " + 
                    "WHEN hoogstebouwlaag IS NULL AND laagstebouwlaag < 0 AND aantalbouwlagen IS NOT NULL AND aantalbouwlagen >= ABS(laagstebouwlaag) THEN aantalbouwlagen - ABS(laagstebouwlaag) " + 
                    # -- WHEN hoogstebouwlaag IS NOT NULL AND laagstebouwlaag IS NULL AND aantalbouwlagen IS NOT NULL --> only one building and it is a shed
                "END AS no_floors, "
                "CONCAT(ligging, '; ', typewoonobject) AS notes " +
                "FROM c1_ams.training_data_buildings);"  
        )

    elif table == 'c2_rot': 
        
        cursor.execute(
            "CREATE TABLE training_data.c2_rot AS " + 
                "(SELECT LPAD(pand_id::varchar, 16, '0') AS bag_id, floors AS no_floors " + 
                "FROM c2_rot.training_data);" 
        )
        
        cursor.execute(
            "ALTER TABLE training_data.c2_rot " + 
            "ADD COLUMN notes text;"
        )

    elif table == 'c3_dhg': 
        
        # Obtain number of floors from flats
        cursor.execute(
            "CREATE TABLE training_data.c3_dhg AS " + 
                "(SELECT LPAD(pand_id::varchar, 16, '0') AS bag_id, hoogste_bouwlaag AS no_floors, " + 
                "CONCAT('galerijflat; ', 'etage or woonlaag: ', et_wl) AS notes " + 
                "FROM c3_dhg.training_data_flats);"   
        )

        # Obtain number of floors from houses
        cursor.execute(
            "INSERT INTO training_data.c3_dhg (bag_id, no_floors, notes) " + 
            "SELECT LPAD(pand_id::varchar, 16, '0') AS bag_id, aantalbouwlagen_pand AS no_floors, CONCAT('woonhuis; ', 'etage or woonlaag: ', et_wl, '; ', 'basement: ', kelder_j_n, '; ', 'loft: ', vliering_j_n) AS notes " + 
            "FROM c3_dhg.training_data_houses;" 
        )

    elif table == 'c4_rho': 

        cursor.execute(
            "CREATE TABLE training_data.c4_rho AS " + 
                "(SELECT LPAD(pnd_id_bag::varchar, 16, '0') AS bag_id, max(object_etageniveau::int) + 1 AS no_floors " + 
                "FROM c4_rho.bgo_verblijfsobject_geometrie_v2 " + 
                "WHERE object_etageniveau != '' " + 
                "GROUP BY pnd_id_bag);" 
        ) 

        cursor.execute(
            "ALTER TABLE training_data.c4_rho " + 
            "ADD COLUMN notes text;" 
        )

    else: 
        print('Data not found for input city! Choose from: <c1_ams>, <c2_rot>, <c3_dhg> or <c4_rho>')
    

def auto_clean(cursor, train_schema, table):
    """
    Perform automatic cleaning steps.

    Parameters: \n
    cursor -- cursor for database connection \n
    train_schema -- schema in which training data is stored \n
    table -- table to perform automatic cleaning steps on \n

    Returns: none

    """

    print('\n>> Dataset {0} -- performing automatic data cleaning'.format(table))
    
    # Remove null or zero values
    cursor.execute(
        "DELETE FROM " + train_schema + "." + table + "_tmp " + 
        "WHERE no_floors IS NULL OR no_floors = 0;"
    )

    # Remove values above 48 (highest no. floors in NL)
    cursor.execute(
        "DELETE FROM " + train_schema + "." + table + "_tmp " + 
        "WHERE no_floors > 48;"
    )

    # Remove negative values
    cursor.execute(
        "DELETE FROM " + train_schema + "." + table + "_tmp " + 
        "WHERE no_floors < 0;"
    )


def clean_dhg(cursor, train_schema):
    """
    Perfom manual cleaning steps for buildings in den haag. 

    Parameters: \n
    cursor -- cursor for database connection \n
    train_schema -- schema in which training data is stored \n

    Returns: none

    """

    print('\n>> Dataset c3_dhg -- performing manual data cleaning')

    # Fix no. floors of apartment blocks using ET / WL distinction
    cursor.execute(
        "UPDATE " + train_schema + ".c3_dhg_tmp " + 
        "SET no_floors = " + 
        "CASE " + 
            "WHEN notes LIKE '%galerijflat%' AND notes LIKE '%ET%' THEN no_floors + 1 " + 
            "WHEN notes LIKE '%galerijflat%' AND notes LIKE '%WL%' AND (no_floors = 2 OR no_floors >= 7) THEN no_floors + 1 " + 
        "END " + 
        "WHERE no_floors = " + 
        "CASE " + 
            "WHEN notes LIKE '%galerijflat%' AND notes LIKE '%ET%' THEN no_floors + 1 " + 
            "WHEN notes LIKE '%galerijflat%' AND notes LIKE '%WL%' AND (no_floors = 2 OR no_floors >= 7) THEN no_floors + 1 " + 
        "END;"
    )

    # Manually update cases where no. floors incorrect 
    cursor.execute(
        "UPDATE " + train_schema + ".c3_dhg_tmp " + 
        "SET no_floors = 14 " + 
        "WHERE bag_id = '0518100000277308';"
    )


def main(params):
    
    # Load parameters 
    jparams = json.load(open(params))

    # Create connection to database
    conn = db_functions.setup_connection()
    conn.autocommit = True

    # Create a cursor (allows PostgreSQL commands to be executed)
    curs = conn.cursor()

    # Create a schema to store training data
    curs.execute("CREATE SCHEMA IF NOT EXISTS training_data")
    
    for table in jparams['training_tables']:

        starttime = time()
        
        # Standardise training data 
        get_floor_count(curs, table)

        train_schema = 'training_data'

        # Create temporary table to store data in
        db_functions.create_temp_table(curs, train_schema, table)

        # Perform automatic data cleaning 
        auto_clean(curs, train_schema, table)

        # Perform manual data cleaning on training data from den haag
        if table == 'c3_dhg':
            clean_dhg(curs, train_schema)

        # Replace original table with unlogged temporary table
        db_functions.replace_temp_table(curs, train_schema, table, pkey='bag_id')

        endtime = time()
        duration = endtime - starttime
        print('\n>> Computation time: ', round(duration, 2), 's \n\n' + 10*'-')
        
    # Close database connection
    db_functions.close_connection(conn, curs)


if __name__ == '__main__':
    main(sys.argv[1])

