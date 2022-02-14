"""
Functions to retrieve and store data for each building in the database. 
Based on: Imke Lansky (https://github.com/ImkeLansky/USA-BuildingHeightInference)

Each "get_" function takes 3 inputs:
cursor -- cursor for database connection 
schema -- schema to store the data in the database 
table -- table to store the data in the database 

Some "get_" functions also include a 4th input called "filter_input" which is used to filter the database tables used.
This variable is set to None by default, meaning that a default filtering statement is applied based on the dataset. 
It can also be changed to a gemeente code (string) or polygon (geopandas.array.GeometryArray) 
in order to perform additional database filtering. 

"""

import json, sys
from time import time
import geopandas as gpd
import db_functions

# ---------------------------------------------------------------
# where_ functions

def where_bag_pand(filter_input=None):

    # Define basic WHERE condition
    where = "WHERE pandstatus LIKE 'Pand in gebruik%' AND einddatumtijdvakgeldigheid IS NULL "

    # Additional WHERE condition based on gemeente code or geometry 
    if type(filter_input) == str: 
        gm_code = filter_input
        filter_output = "AND substring(identificatie, 1, 4) = '" + gm_code + "' "

    elif type(filter_input) == gpd.array.GeometryArray: 
        wkt_geom = gpd.array.to_wkt(filter_input)[0]
        filter_output = "AND ST_intersects(geom, ST_GeomFromText('" + wkt_geom + "', 28992)) "

    else: 
        filter_output = ''

    # Concatenate additional WHERE condition to basic condition
    where = where + filter_output

    return where


def where_bag_vo(filter_input): 

    # Define basic WHERE condition 
    where = "WHERE verblijfsobjectstatus IN ('Verblijfsobject in gebruik', 'Verblijfsobject in gebruik (niet ingemeten)', 'Verblijfsobject gevormd') "

    #  Additional WHERE condition based on gemeente code or geometry 
    if type(filter_input) == str: 
        gm_code = filter_input
        filter_output = "AND substring(pandid, 1, 4) = '" + gm_code + "' "

    elif type(filter_input) == gpd.array.GeometryArray: 
        wkt_geom = gpd.array.to_wkt(filter_input)[0]
        filter_output = "AND ST_intersects(geom, ST_GeomFromText('" + wkt_geom + "', 28992)) "

    else: 
        filter_output = ''

    # Concatenate additional WHERE condition to basic condition
    where = where + filter_output

    return where 


def where_3dbag_pand(filter_input=None):

    # Define basic WHERE condition 
    where = "WHERE CAST(status AS VARCHAR) LIKE 'Pand in gebruik%' AND eindgeldigheid IS NULL "

    #  Additional WHERE condition based on gemeente code or geometry 
    if type(filter_input) == str: 
        gm_code = filter_input
        filter_output = "AND substring(identificatie, 15, 4) = '" + gm_code + "' "

    elif type(filter_input) == gpd.array.GeometryArray: 
        wkt_geom = gpd.array.to_wkt(filter_input)[0]
        filter_output = "AND ST_intersects(geometrie, ST_GeomFromText('" + wkt_geom + "', 28992)) "

    else: 
        filter_output = ''

    # Concatenate additional WHERE condition to basic condition
    where = where + filter_output

    return where


def where_cbs(filter_input=None):

    # Define basic WHERE condition 
    where = "WHERE ST_intersects(stats.geom, buildings.footprint_geom) "

    #  Additional WHERE condition based on gemeente code or geometry 
    if type(filter_input) == str: 
        gm_code = filter_input
        filter_output = "AND substring(stats.gemeentecode, 3) = '" + gm_code + "' "

    elif type(filter_input) == gpd.array.GeometryArray: 
        wkt_geom = gpd.array.to_wkt(filter_input)[0]
        filter_output = "AND ST_intersects(stats.geom, ST_GeomFromText('" + wkt_geom + "', 28992)) "

    else: 
        filter_output = ''

    # Concatenate additional WHERE condition to basic condition
    where = where + filter_output

    return where

# ---------------------------------------------------------------
# get_ functions

def get_bouwjaar(cursor, schema, table, filter_input=None): 

    print('\n>> Dataset {0} -- obtaining construction year'.format(table))

    where = where_bag_pand(filter_input)

    cursor.execute("ALTER TABLE " + schema + "." + table + "_tmp ADD COLUMN IF NOT EXISTS bag_construction_year INTEGER;")

    cursor.execute(
        "UPDATE " + schema + "." + table + "_tmp " + 
        "SET bag_construction_year = subquery.bag_construction_year " + 
        "FROM " + 
            "(SELECT identificatie, bouwjaar AS bag_construction_year " + 
            "FROM bag.pand " + where + ") AS subquery " +
        "WHERE " + schema + "." + table + "_tmp.bag_id = subquery.identificatie;"
    )

    cursor.execute(
        "UPDATE " + schema + "." + table + "_tmp " + 
        "SET bag_construction_year = NULL " + 
        "WHERE " + schema + "." + table + "_tmp.bag_construction_year = 1005;"
    )


def get_footprint(cursor, schema, table, filter_input=None): 

    print('\n>> Dataset {0} -- obtaining building footprint'.format(table))

    where = where_bag_pand(filter_input)

    cursor.execute("ALTER TABLE " + schema + "." + table + "_tmp ADD COLUMN IF NOT EXISTS footprint_geom GEOMETRY;")

    cursor.execute(
        "UPDATE " + schema + "." + table + "_tmp " + 
        "SET footprint_geom = subquery.footprint_geom " + 
        "FROM " + 
            "(SELECT identificatie, geom AS footprint_geom " + 
            "FROM bag.pand " + where + ") AS subquery " +
        "WHERE " + schema + "." + table + "_tmp.bag_id = subquery.identificatie;"
    )

    # Create index on footprint geometry column
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS " + table + "_footprint_idx_tmp " + 
        "ON " + schema + "." + table + "_tmp " + 
        "USING GIST (footprint_geom);"
    )


def get_floor_area(cursor, schema, table, filter_input=None): 

    print('\n>> Dataset {0} -- obtaining net internal floor area'.format(table))

    where = where_bag_vo(filter_input)

    cursor.execute("ALTER TABLE " + schema + "." + table + "_tmp ADD COLUMN IF NOT EXISTS bag_net_internal_area REAL;")

    cursor.execute(
        "UPDATE " + schema + "." + table + "_tmp " + 
        "SET bag_net_internal_area = subquery.bag_net_internal_area " + 
        "FROM " + 
            "(SELECT pandid, SUM(oppervlakteverblijfsobject) AS bag_net_internal_area " + 
            "FROM bag.adres_full " + where + 
            "GROUP BY pandid) AS subquery " + 
        "WHERE " + schema + "." + table + "_tmp.bag_id = subquery.pandid;" 
    )


def get_no_units(cursor, schema, table, filter_input=None): 

    print('\n>> Dataset {0} -- obtaining number of units'.format(table))

    where = where_bag_vo(filter_input)

    cursor.execute("ALTER TABLE " + schema + "." + table + "_tmp ADD COLUMN IF NOT EXISTS bag_no_units INTEGER;")

    cursor.execute(
        "UPDATE " + schema + "." + table + "_tmp " + 
        "SET bag_no_units = subquery.bag_no_units " +
        "FROM " + 
            "(SELECT pandid, COUNT(adresseerbaarobject) AS bag_no_units " + 
            "FROM bag.adres_full " + where +  
            "GROUP BY pandid) AS subquery " + 
        "WHERE " + schema + "." + table + "_tmp.bag_id = subquery.pandid;" 
    )


def get_function(cursor, schema, table, filter_input=None): 

    print('\n>> Dataset {0} -- obtaining building function'.format(table))

    where = where_bag_vo(filter_input)

    cursor.execute("ALTER TABLE " + schema + "." + table + "_tmp ADD COLUMN IF NOT EXISTS uses VARCHAR[];")
    cursor.execute("ALTER TABLE " + schema + "." + table + "_tmp ADD COLUMN IF NOT EXISTS bag_function VARCHAR;")

    cursor.execute(
        "UPDATE " + schema + "." + table + "_tmp " + 
        "SET uses = subquery.uses " + 
        "FROM " + 
            "(SELECT pandid, ARRAY(SELECT DISTINCT unnest(string_to_array(string_agg(verblijfsobjectgebruiksdoel, ', '), ', '))) AS uses " + 
            "FROM bag.adres_full " + where + 
            "GROUP BY pandid) AS subquery " + 
        "WHERE " + schema + "." + table + "_tmp.bag_id = subquery.pandid;" 
    )

    cursor.execute(
        "UPDATE " + schema + "." + table + "_tmp " + 
        "SET bag_function = subquery.bag_function " + 
        "FROM " + 
            "(SELECT bag_id, " + 
            "CASE " + 
                "WHEN uses = '{woonfunctie}' THEN 'Residential' " + 
                "WHEN uses != '{woonfunctie}' AND 'woonfunctie' = ANY(uses) THEN 'Mixed-residential' " + 
                "WHEN 'woonfunctie' != ANY(uses) AND uses != '{overige gebruiksfunctie}' AND cardinality(uses) = 1 THEN 'Non-residential (single-function)' " + 
                "WHEN 'woonfunctie' != ANY(uses) AND uses != '{overige gebruiksfunctie}' AND cardinality(uses) > 1 THEN 'Non-residential (multi-function)' " + 
                "WHEN uses = '{overige gebruiksfunctie}' THEN 'Others' " + 
                "WHEN uses IS NULL THEN 'Unknown' " + 
            "END AS bag_function " +
            "FROM " + schema + "." + table + "_tmp) AS subquery " + 
        "WHERE " + schema + "." + table + "_tmp.bag_id = subquery.bag_id;"
    )

    cursor.execute(
        "ALTER TABLE " + schema + "." + table + "_tmp " +
        "DROP COLUMN uses;"
    )


def get_no_res_units(cursor, schema, table, filter_input=None): 
    """
    This function can only be run after obtaining building function using "get_function". 

    """

    print('\n>> Dataset {0} -- obtaining no. residential units'.format(table))

    where = where_bag_vo(filter_input)

    cursor.execute("ALTER TABLE " + schema + "." + table + "_tmp ADD COLUMN IF NOT EXISTS all_uses VARCHAR[];")
    cursor.execute("ALTER TABLE " + schema + "." + table + "_tmp ADD COLUMN IF NOT EXISTS bag_no_res_units INTEGER;")

    cursor.execute(
        "UPDATE " + schema + "." + table + "_tmp " + 
        "SET all_uses = subquery.all_uses " + 
        "FROM " + 
            "(SELECT pandid, string_to_array(string_agg(verblijfsobjectgebruiksdoel, ', '), ', ') AS all_uses " +  
            "FROM bag.adres_full " + where +  
            "GROUP BY pandid) AS subquery " + 
        "WHERE " + schema + "." + table + "_tmp.bag_id = subquery.pandid;" 
    )

    cursor.execute(
        "UPDATE " + schema + "." + table + "_tmp " + 
        "SET bag_no_res_units = subquery.bag_no_res_units " + 
        "FROM " + 
            "(SELECT bag_id, (SELECT SUM(CASE b WHEN 'woonfunctie' THEN 1 ELSE 0 END) FROM unnest(all_uses) AS dt(b)) AS bag_no_res_units " +
            "FROM " + schema + "." + table + "_tmp " + 
            "WHERE bag_function = 'Residential' " + 
            "OR (bag_function = 'Mixed-use' and bag_no_units != (SELECT SUM(CASE b WHEN 'woonfunctie' THEN 1 ELSE 0 END) FROM unnest(all_uses) AS dt(b)))) AS subquery " + 
        "WHERE " + schema + "." + table + "_tmp.bag_id = subquery.bag_id;"
    )

    cursor.execute(
        "ALTER TABLE " + schema + "." + table + "_tmp " + 
        "DROP COLUMN all_uses;"
    )


def get_roof_type(cursor, schema, table, filter_input=None):

    print('\n>> Dataset {0} -- obtaining roof type'.format(table))

    where = where_3dbag_pand(filter_input)

    cursor.execute("ALTER TABLE " + schema + "." + table + "_tmp ADD COLUMN IF NOT EXISTS bag3d_roof_type VARCHAR;")
    
    cursor.execute(
        "UPDATE " + schema + "." + table + "_tmp " + 
        "SET bag3d_roof_type = subquery.bag3d_roof_type " + 
        "FROM " + 
            "(SELECT substring(identificatie, 15) AS pand_id, dak_type AS bag3d_roof_type " + 
            "FROM bag3d.pand " + where + ") AS subquery " +
        "WHERE " + schema + "." + table + "_tmp.bag_id = subquery.pand_id;"
    )


def get_bag3d_id(cursor, schema, table, filter_input=None): 

    print('\n>> Dataset {0} -- obtaining bag3d id'.format(table))

    where = where_3dbag_pand(filter_input)

    cursor.execute("ALTER TABLE " + schema + "." + table + "_tmp ADD COLUMN IF NOT EXISTS bag3d_id INTEGER;")

    cursor.execute(
        "UPDATE " + schema + "." + table + "_tmp " + 
        "SET bag3d_id = subquery.bag3d_id " +
        "FROM " + 
            "(SELECT substring(identificatie, 15) AS pand_id, fid AS bag3d_id " + 
            "FROM bag3d.pand " + where + ") AS subquery " +
        "WHERE " + schema + "." + table + "_tmp.bag_id = subquery.pand_id;"
    )


def get_h_ground(cursor, schema, table, filter_input=None):

    print('\n>> Dataset {0} -- obtaining ground height'.format(table))

    where = where_3dbag_pand(filter_input)

    cursor.execute("ALTER TABLE " + schema + "." + table + "_tmp ADD COLUMN IF NOT EXISTS bag3d_h_ground REAL;")

    cursor.execute(
        "UPDATE " + schema + "." + table + "_tmp " + 
        "SET bag3d_h_ground = subquery.bag3d_h_ground " +
        "FROM " + 
            "(SELECT substring(identificatie, 15) AS pand_id, h_maaiveld AS bag3d_h_ground " + 
            "FROM bag3d.pand " + where + ") AS subquery " +
        "WHERE " + schema + "." + table + "_tmp.bag_id = subquery.pand_id;"
    )


def get_h_percentiles(cursor, schema, table):
    """
    This function can only be run after obtaining ground height and bag3d id using "get_bag3d_id" and "get_h_ground".

    """

    print('\n>> Dataset {0} -- obtaining point cloud height percentiles'.format(table))

    cursor.execute("ALTER TABLE " + schema + "." + table + "_tmp ADD COLUMN IF NOT EXISTS h_min REAL;")
    cursor.execute("ALTER TABLE " + schema + "." + table + "_tmp ADD COLUMN IF NOT EXISTS h_50p REAL;")
    cursor.execute("ALTER TABLE " + schema + "." + table + "_tmp ADD COLUMN IF NOT EXISTS h_70p REAL;")
    cursor.execute("ALTER TABLE " + schema + "." + table + "_tmp ADD COLUMN IF NOT EXISTS h_max REAL;")

    cursor.execute(
        "UPDATE " + schema + "." + table + "_tmp " + 
        "SET h_min = subquery.h_min, h_50p = subquery.h_50p, h_70p = subquery.h_70p, h_max = subquery.h_max "
        "FROM " +
            "(SELECT t.bag_id, lod12_2d.h_dak_min - t.bag3d_h_ground AS h_min, " + 
            "lod12_2d.h_dak_50p - t.bag3d_h_ground AS h_50p, " + 
            "lod12_2d.h_dak_70p - t.bag3d_h_ground AS h_70p, " +  
            "lod12_2d.h_dak_max - t.bag3d_h_ground AS h_max " + 
            "FROM " + schema + "." + table + "_tmp AS t " + 
            "INNER JOIN bag3d.lod12_2d ON lod12_2d.fid = t.bag3d_id) AS subquery " + 
        "WHERE " + schema + "." + table + "_tmp.bag_id = subquery.bag_id;"
    )


def get_pop_density(cursor, schema, table, filter_input=None): 
    """
    This function can only be run after obtaining building footprints using "get_footprint". 

    """

    print('\n>> Dataset {0} -- obtaining population density'.format(table))

    where = where_cbs(filter_input)

    cursor.execute("ALTER TABLE " + schema + "." + table + "_tmp ADD COLUMN IF NOT EXISTS cbs_pop_per_km2 INTEGER;")

    cursor.execute(
        "UPDATE " + schema + "." + table + "_tmp " + 
        "SET cbs_pop_per_km2 = subquery.cbs_pop_per_km2 " +
        "FROM " + 
            "(SELECT buildings.bag_id as id, stats.bevolkingsdichtheid_inwoners_per_km2 AS cbs_pop_per_km2 " +  
            "FROM cbs.cbs_buurten_2019 as stats, " + schema + "." + table + "_tmp AS buildings " + where + ") AS subquery " + 
        "WHERE " + schema + "." + table + "_tmp.bag_id = subquery.id;"
    )


def get_percent_multihousehold(cursor, schema, table, filter_input=None): 
    """
    This function can only be run after obtaining building footprints using "get_footprint". 

    """

    print('\n>> Dataset {0} -- obtaining percentage of multihousehold buildings'.format(table))

    where = where_cbs(filter_input)

    cursor.execute("ALTER TABLE " + schema + "." + table + "_tmp ADD COLUMN IF NOT EXISTS cbs_percent_multihousehold INTEGER;")

    cursor.execute(
        "UPDATE " + schema + "." + table + "_tmp " + 
        "SET cbs_percent_multihousehold = subquery.cbs_percent_multihousehold " + 
        "FROM " + 
            "(SELECT buildings.bag_id as id, stats.percentage_meergezinswoning AS cbs_percent_multihousehold " + 
            "FROM cbs.cbs_buurten_2019 as stats, " + schema + "." + table + "_tmp AS buildings " + where + ") AS subquery " + 
        "WHERE " + schema + "." + table + "_tmp.bag_id = subquery.id;"
    )


def get_avg_dist(cursor, schema, table, filter_input=None): 
    """
    This function can only be run after obtaining building footprints using "get_footprint". 

    """

    print('\n>> Dataset {0} -- obtaining average distance to shops, supermarkets and cafes'.format(table))

    where = where_cbs(filter_input)

    cursor.execute("ALTER TABLE " + schema + "." + table + "_tmp ADD COLUMN IF NOT EXISTS cbs_shops_avg_dist REAL;")
    cursor.execute("ALTER TABLE " + schema + "." + table + "_tmp ADD COLUMN IF NOT EXISTS cbs_supermarket_avg_dist REAL;")
    cursor.execute("ALTER TABLE " + schema + "." + table + "_tmp ADD COLUMN IF NOT EXISTS cbs_cafe_avg_dist REAL;")

    cursor.execute(
        "UPDATE " + schema + "." + table + "_tmp " + 
        "SET cbs_shops_avg_dist = subquery.cbs_shops_avg_dist, " + 
        "cbs_supermarket_avg_dist = subquery.cbs_supermarket_avg_dist, " + 
        "cbs_cafe_avg_dist = subquery.cbs_cafe_avg_dist " + 
        "FROM " + 
            "(SELECT buildings.bag_id as id, stats.winkels_ov_dagelijkse_levensm_gem_afst_in_km AS cbs_shops_avg_dist, " +  
            "stats.grote_supermarkt_gemiddelde_afstand_in_km AS cbs_supermarket_avg_dist, " + 
            "stats.cafe_gemiddelde_afstand_in_km AS cbs_cafe_avg_dist " + 
            "FROM cbs.cbs_buurten_2019 as stats, " + schema + "." + table + "_tmp AS buildings " + where + ") AS subquery " + 
        "WHERE " + schema + "." + table + "_tmp.bag_id = subquery.id;"
    )


def get_avg_1km(cursor, schema, table, filter_input=None): 
    """
    This function can only be run after obtaining building footprints using "get_footprint". 

    """

    print('\n>> Dataset {0} -- obtaining average no. shops, supermarkets and cafes within 1km'.format(table))

    where = where_cbs(filter_input)

    cursor.execute("ALTER TABLE " + schema + "." + table + "_tmp ADD COLUMN IF NOT EXISTS cbs_shops_avg_1km REAL;")
    cursor.execute("ALTER TABLE " + schema + "." + table + "_tmp ADD COLUMN IF NOT EXISTS cbs_supermarket_avg_1km REAL;")
    cursor.execute("ALTER TABLE " + schema + "." + table + "_tmp ADD COLUMN IF NOT EXISTS cbs_cafe_avg_1km REAL;")

    cursor.execute(
        "UPDATE " + schema + "." + table + "_tmp " + 
        "SET cbs_shops_avg_1km = subquery.cbs_shops_avg_1km, " + 
        "cbs_supermarket_avg_1km = subquery.cbs_supermarket_avg_1km, " + 
        "cbs_cafe_avg_1km = subquery.cbs_cafe_avg_1km " + 
        "FROM " + 
            "(SELECT buildings.bag_id as id, stats.winkels_ov_dagel_levensm_gem_aantal_binnen_1_km AS cbs_shops_avg_1km, " + 
            "stats.grote_supermarkt_gemiddeld_aantal_binnen_1_km AS cbs_supermarket_avg_1km, " + 
            "stats.cafe_gemiddeld_aantal_binnen_1_km AS cbs_cafe_avg_1km " + 
            "FROM cbs.cbs_buurten_2019 as stats, " + schema + "." + table + "_tmp AS buildings " + where + ") AS subquery " + 
        "WHERE " + schema + "." + table + "_tmp.bag_id = subquery.id;"
    )


def get_building_type(cursor, schema, table):

    print('\n>> Dataset {0} -- obtaining ESRI building type'.format(table))

    cursor.execute("ALTER TABLE " + schema + "." + table + "_tmp ADD COLUMN IF NOT EXISTS esri_building_type VARCHAR;")

    cursor.execute(
        "UPDATE " + schema + "." + table + "_tmp " + 
        "SET esri_building_type = subquery.esri_building_type " +
        "FROM " + 
            "(SELECT identificatie, CASE " + 
                "WHEN woningtypering = 'vrijstaande woning' THEN 'detached' " + 
                "WHEN woningtypering = 'twee-onder-een-kap' THEN 'semi-detached' " + 
                "WHEN woningtypering IN ('hoekwoning', 'tussenwoning/geschakeld') THEN 'terraced' " +
                "WHEN woningtypering = 'appartement' THEN 'apartment' " +  
            "END AS esri_building_type " + 
            "FROM bag.esri_buildingtype) AS subquery " + 
        "WHERE " + schema + "." + table + "_tmp.bag_id = subquery.identificatie;"
    )


def main(params):

    # Load parameters 
    jparams = json.load(open(params))

    # Create connection to database
    conn = db_functions.setup_connection()
    conn.autocommit = True

    # Create a cursor (allows PostgreSQL commands to be executed)
    curs = conn.cursor()

    # Loop through each training data table
    for table in jparams["training_tables"]:

        starttime = time()

        # Parameters: 
        gm_code = jparams["gemeente_codes"][table]
        train_schema = 'training_data'
        
        # Create temporary table to store extracted data in
        db_functions.create_temp_table(curs, train_schema, table, pkey='bag_id')

        # Retrieve building function
        get_function(curs, train_schema, table, filter_input=gm_code)

        # Remove any rows where function is not residential/mixed-residential
        print('\n>> Dataset {0} -- removing non-residential buildings'.format(table))
        curs.execute("DELETE FROM training_data." + table + "_tmp WHERE bag_function != 'Residential' AND bag_function != 'Mixed-residential';")

        # Retrieve data from BAG
        get_bouwjaar(curs, train_schema, table, filter_input=gm_code)
        get_footprint(curs, train_schema, table, filter_input=gm_code)
        get_floor_area(curs, train_schema, table, filter_input=gm_code)
        get_no_units(curs, train_schema, table, filter_input=gm_code)
        get_no_res_units(curs, train_schema, table, filter_input=gm_code)

        # Retrieve data from 3D BAG 
        get_roof_type(curs, train_schema, table, filter_input=gm_code)
        get_bag3d_id(curs, train_schema, table, filter_input=gm_code)
        get_h_ground(curs, train_schema, table, filter_input=gm_code)
        get_h_percentiles(curs, train_schema, table)

        # Retrieve data from CBS
        get_pop_density(curs, train_schema, table, filter_input=gm_code)
        get_percent_multihousehold(curs, train_schema, table, filter_input=gm_code)
        get_avg_dist(curs, train_schema, table, filter_input=gm_code)
        get_avg_1km(curs, train_schema, table, filter_input=gm_code)

        # Retrieve builidng type   
        get_building_type(curs, train_schema, table)

        # Replace original table with unlogged temporary tables
        db_functions.replace_temp_table(curs, train_schema, table, pkey='bag_id', geom_index='footprint_geom')

        endtime = time()
        duration = endtime - starttime
        print('\n>> Computation time: ', round(duration, 2), 's \n\n' + 10*'-')

    # Close database connection
    db_functions.close_connection(conn, curs)


if __name__ == '__main__':
    main(sys.argv[1])