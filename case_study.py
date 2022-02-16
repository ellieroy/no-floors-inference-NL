"""
Obtains data required for case study analysis, makes predictions for case study buildings 
and stores results to geojson for further analysis. 

"""

import json, sys, os
from time import time
import numpy as np
import pandas as pd
import db_functions
import retrieve_data
import extract_3d_features
import extract_2d_features
import get_ref_model 
import train_models
import geom_functions
from visualise_data import directory_exists
from joblib import load
import geopandas as gpd


def get_bagid(curs, schema, table, gm_code): 
    """
    Create a new database table as all bag ids present in the input municipality. 

    Parameters: \n
    curs -- cursor for database connection \n
    schema -- schema to store data \n
    table -- table in database schema to store data \n
    gm_code -- 4 digit number corresponding to a municipality in the netherlands (string)

    Returns: None

    """

    print("\n>> Creating table for case study: {0}".format(table))
    
    curs.execute("DROP TABLE IF EXISTS " + schema + "." + table + ";")
    
    curs.execute(
        "CREATE TABLE IF NOT EXISTS " + schema + "." + table + " AS " +
            "(SELECT DISTINCT identificatie AS bag_id " + 
            "FROM bag.pand " + 
            "WHERE substring(identificatie, 1, 4) = '" + gm_code + "' " + 
            "AND pandstatus LIKE 'Pand in gebruik' AND einddatumtijdvakgeldigheid IS NULL);")


def get_data(jparams): 
    """
    Get all data required for case study analysis. 

    Parameters: \n
    jparams -- dictionary of parameters from json parameter file \n

    Returns: None

    """

    # Create connection to database
    conn = db_functions.setup_connection()
    conn.autocommit = True

    # Create a cursor (allows PostgreSQL commands to be executed)
    curs = conn.cursor()

    for table in jparams['case_study_tables']:

        starttime = time()

        # Create a schema to store case study data
        schema = jparams['case_study_schema']
        curs.execute("CREATE SCHEMA IF NOT EXISTS " + schema + ";")

        # Create table for case study containing building ids
        gm_code = jparams["gemeente_codes"][table]
        get_bagid(curs, schema, table, gm_code)

        # Create temporary table to store extracted data in
        db_functions.create_temp_table(curs, schema, table, pkey='bag_id')

        # Retrieve building function
        retrieve_data.get_function(curs, schema, table, filter_input=gm_code)

        # Remove any rows where functions is not residential/mixed-residential
        print('\n>> Dataset {0} -- removing non-residential buildings'.format(table))
        curs.execute("DELETE FROM " + schema + "." + table + "_tmp WHERE bag_function != 'Residential' AND bag_function != 'Mixed-residential';")

        # Retrieve data from BAG
        retrieve_data.get_bouwjaar(curs, schema, table, filter_input=gm_code)
        retrieve_data.get_footprint(curs, schema, table, filter_input=gm_code)
        retrieve_data.get_floor_area(curs, schema, table, filter_input=gm_code)
        retrieve_data.get_no_units(curs, schema, table, filter_input=gm_code)
        retrieve_data.get_no_res_units(curs, schema, table, filter_input=gm_code)

        # Retrieve data from 3D BAG 
        retrieve_data.get_roof_type(curs, schema, table, filter_input=gm_code)
        retrieve_data.get_bag3d_id(curs, schema, table, filter_input=gm_code)
        retrieve_data.get_h_ground(curs, schema, table, filter_input=gm_code)
        retrieve_data.get_h_percentiles(curs, schema, table)

        # Retrieve data from CBS
        retrieve_data.get_pop_density(curs, schema, table, filter_input=gm_code)
        retrieve_data.get_percent_multihousehold(curs, schema, table, filter_input=gm_code)
        retrieve_data.get_avg_dist(curs, schema, table, filter_input=gm_code)
        retrieve_data.get_avg_1km(curs, schema, table, filter_input=gm_code)

        # Retrieve builidng type   
        retrieve_data.get_building_type(curs, schema, table)

        # Get area, perimeter and vertices from footprint geometry 
        extract_2d_features.get_footprint_area(curs, schema, table)
        extract_2d_features.get_footprint_perim(curs, schema, table)
        extract_2d_features.get_footprint_vertices(curs, schema, table)

        # Extract and store required attributes of all buildings within 300m buffer of municipality 
        # (so that no. adjacent and neighbouring buildings can be obtained correctly)
        all_build_table = table + "_all"
        wkt_gm_buffer = geom_functions.get_gemeente_polygon(conn, gm_code, buffer=300)
        extract_2d_features.get_all_buildings_attribs(curs, schema, all_build_table, wkt_gm_buffer)

        # Get no. adjacent and neighbouring buildings 
        dist_adjacent = jparams["distance_adjacent"]
        dist_neighbours = jparams["distance_neighbours"]
        extract_2d_features.get_adjacent_count(curs, schema, table, schema, all_build_table, dist_adjacent)
        extract_2d_features.get_neighbour_count(curs, schema, table, schema, all_build_table, dist_neighbours)

        # Extract 3d features 
        df_lod1 = extract_3d_features.get_lod1_data(conn, schema, table)
        df_lod2 = extract_3d_features.get_lod2_data(conn, schema, table)
        extract_3d_features.get_lod2_hrefs(df_lod2, curs, schema, table, 'lod22')
        extract_3d_features.get_surface_areas(df_lod1, curs, schema, table, 'lod12')
        extract_3d_features.get_surface_areas(df_lod2, curs, schema, table, 'lod22')
        extract_3d_features.get_building_volume(df_lod1, jparams["voxel_scales"]["LOD12"], curs, schema, table, 'lod12')
        extract_3d_features.get_building_volume(df_lod2, jparams["voxel_scales"]["LOD22"], curs, schema, table, 'lod22')

        # Replace original table with unlogged temporary tables
        db_functions.replace_temp_table(curs, schema, table, pkey='bag_id')

        # Drop table containing all buildings
        curs.execute("DROP TABLE " + schema + "." + all_build_table + ";")

        # Ref model results
        ceiling_height = jparams["ceiling_height"]
        get_ref_model.height_based(curs, schema, table, ceiling_height)
        get_ref_model.area_based(curs, schema, table)
        get_ref_model.ref_model(curs, schema, table)

        endtime = time()
        duration = endtime - starttime
        print('\n>> Computation time: ', round(duration, 2), 's \n\n' + 10*'-')

    # Close database connection
    db_functions.close_connection(conn, curs)


def main(params):
    
    # Load parameters 
    jparams = json.load(open(params))

    # Extract required data into new database tables
    extract_features = False
    if extract_features: 
        print('\n>> Extracting feautures')
        get_data(jparams)

    # Filename of best estimator 
    filename = jparams["best_estimator"]
    file_info = filename[:-7].split('_')
    model = 'model_' + file_info[2]

    if os.path.exists('data/train_sets/train_set_' + model + '.csv') and os.path.exists('./models/' + filename):

        # Get the names of the features used by the input model 
        train_set = pd.read_csv('data/train_sets/train_set_' + model + '.csv', index_col=0, dtype={'bag_id':'str'})
        X_train, y_train = train_models.split_features_labels(train_set, jparams["id_column"], jparams["labels_column"])
        features_list = list(X_train.columns)
        feature_names = ''
        for i, item in enumerate(features_list):
            if i == 0:
                feature_names += item
            else: 
                feature_names += ', ' + item

        # Download required columns from database
        building_id = jparams["id_column"]
        other_data = 'ST_AsText(footprint_geom) AS footprint_geom, ref_model, ref_model_unrounded, floors_height_based, floors_area_based'
        columns = building_id + ', ' + feature_names + ', ' + other_data
        case_study_data = train_models.collect_data(jparams['case_study_schema'], jparams['case_study_tables'], cols=columns, where='') 
        case_study_features = case_study_data[features_list]
        
        # Load best model from file 
        print('\n>> Loading model')
        pipeline = load('./models/' + filename)

        # Make predictions and store in df
        print('\n>> Making predictions')
        preds = pipeline.predict(case_study_features)
        preds_rounded = np.rint(pipeline.predict(case_study_features))
        case_study_data = case_study_data.assign(pred_value=preds, pred_value_rounded=preds_rounded)
        case_study_data['diff'] = abs(case_study_data['pred_value_rounded'] - case_study_data['ref_model'])
        
        # Check directory to save data exists
        if not directory_exists('./data/geojson'): 
            os.makedirs('./data/geojson')

        # Create geodataframe and store results to geojson
        print('\n>> Storing case study results to geojson')
        case_study_data['geometry'] = gpd.GeoSeries.from_wkt(case_study_data['footprint_geom'])
        case_study_gdf = gpd.GeoDataFrame(case_study_data, crs='EPSG:28992', geometry='geometry')
        case_study_gdf.to_file('./data/geojson/case_study.geojson', driver='GeoJSON')
        

if __name__ == '__main__':
    main(sys.argv[1])