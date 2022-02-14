"""
Functions to extract and store features based on the 3D geometry of each building in the database. 

"""

import json, sys
from time import time
import geopandas as gpd
import numpy as np
import pyvista as pv 
from scipy.spatial import ConvexHull
from tqdm import tqdm
import db_functions
import geom_functions
import val3ditypy


def get_lod1_data(connection, schema, table):
    """
    Store required data from 3D BAG LOD1.2 models in pandas dataframe.

    Parameters: \n
    connection -- database connection \n
    schema -- schema to store the data in the database \n
    table -- table to store the data in the database \n
    
    Returns: GeoDataFrame containing all data required to extract 3D features from LOD1

    """

    print('\n>> Dataset {0} -- obtaining 3D BAG data (LOD1)'.format(table))

    query = ("SELECT DISTINCT t.bag_id, " + 
            "lod12_3d._semantics_values AS semantic_values, " + 
            "lod12_3d.geometrie AS geometry " + 
            "FROM " + schema + "." + table + "_tmp AS t " + 
            "INNER JOIN bag3d.lod12_3d ON lod12_3d.fid = t.bag3d_id " + 
            "WHERE lod12_3d.val3dity_codes IS NOT NULL;")

    data = gpd.GeoDataFrame.from_postgis(query, connection, geom_col='geometry')

    return data 


def get_lod2_data(connection, schema, table):
    """
    Store required data from 3D BAG LOD2.2 models in pandas dataframe.

    Parameters: \n
    connection -- database connection \n
    schema -- schema to store the data in the database \n
    table -- table to store the data in the database \n
    
    Returns: GeoDataFrame containing all data required to extract 3D features from LOD2

    """

    print('\n>> Dataset {0} -- obtaining 3D BAG data (LOD2)'.format(table))

    query = ("SELECT DISTINCT t.bag_id, t.bag3d_roof_type, t.bag3d_h_ground, " + 
            "lod22_3d._semantics_values AS semantic_values, " + 
            "lod22_3d.geometrie AS geometry " +
            "FROM " + schema + "." + table + "_tmp AS t " + 
            "INNER JOIN bag3d.lod22_3d ON lod22_3d.fid = t.bag3d_id " + 
            "WHERE lod22_3d.val3dity_codes IS NOT NULL;")

    data = gpd.GeoDataFrame.from_postgis(query, connection, geom_col='geometry')

    return data


def get_lod2_hrefs(df, cursor, schema, table, lod): 
    """
    Extract lod2 height references (ridge/eaves) and store results in database. 

    Parameters: \n
    df -- dataframe containing required data from 3D BAG
    cursor -- cursor to execute database queries
    schema -- schema to store the data in the database \n
    table -- table to store the data in the database \n
    lod -- string providing information on lod being processed (lod22) \n

    Returns: none

    """

    print('\n>> Dataset {0} -- extracting {1} height references'.format(table, lod))

    h_ridge = []
    h_eaves = []

    for geometry, semantic_values, roof_type, h_ground in tqdm(zip(df.geometry, df.semantic_values, df.bag3d_roof_type, df.bag3d_h_ground), total=len(df.index)):

        # Initialise list to store polygon vertices 
        vertices = []

        # Initialise lists to store z-coords 
        zlist_walls = []
        zlist_roof = []

        # Loop through each polygon stored in the multipolygon geometry
        for polygon, semantic_value in zip(geometry, semantic_values): 

            for xyz in polygon.exterior.coords[:-1]:
                    
                # Check if cooordinates not in polygon vertices list
                if xyz not in vertices: 
                    
                    # If not in list, append coordinates to vertices list 
                    vertices.append(xyz)

            # Check for roof surfaces 
            if semantic_value == 1:

                # Append the max z-coord of each polygon to roof list
                z_max = max([c[-1] for c in polygon.exterior.coords[:-1]])
                zlist_roof.append(z_max)

            # Check for exterior wall surfaces
            elif semantic_value == 2:

                # Append z-coords to walls list 
                zlist_walls.extend([c[-1] for c in polygon.exterior.coords[:-1]])
            
        if len(zlist_roof) > 0: 

            # Round the z-coords to 3 decimal places 
            zlist_walls = list(np.around(np.array(zlist_walls), 3))
            zlist_roof = list(np.around(np.array(zlist_roof), 3))

            if roof_type == 'slanted':
                # Set the ridge height as the 90th percentile of roof z-coords
                ridge_height = np.around(np.percentile(zlist_roof, 90) - h_ground, 3) 

                # Find the wall z-coords that do not coincide with any z-coords in roof list
                walls_without_ridge = list(set(zlist_walls) - set(zlist_roof))

                # Remove any z-coords that are less than or equal to the ground height
                walls_without_ridge = [x for x in walls_without_ridge if x > h_ground]

                if len(walls_without_ridge) > 0: 
                    # Set eave height as the median (50th percentile) of the remaining z-coords 
                    eave_height = np.around(np.percentile(walls_without_ridge, 50) - h_ground, 3)

                else:  # for cases where roof type is actually horizontal 

                    # Set eave height equal to ridge height
                    eave_height = ridge_height

            elif roof_type == 'multiple horizontal':
                
                # Set ridge height as 90th percentile of roof coords
                ridge_height = np.around(np.percentile(zlist_roof, 90) - h_ground, 3) 

                # Set eave height as 70th percentile of roof coords 
                eave_height = np.around(np.percentile(zlist_roof, 75) - h_ground, 3) 

            else: # single horizontal roof type

                # Set the ridge height as the 90th percentile of roof z-coords
                ridge_height = np.around(np.percentile(zlist_roof, 90) - h_ground, 3)

                # Set the eave height equal to the ridge height
                eave_height = ridge_height
                
        else:
            # If roof z-coords list is empty then set values equal to None
            ridge_height = None
            eave_height = None

        h_ridge.append(ridge_height)
        h_eaves.append(eave_height)

    # Store results in dataframe
    df['h_ridge'] = h_ridge
    df['h_eaves'] = h_eaves
    df['href_diff'] = df['h_ridge'] - df['h_eaves']

    rows = zip(df.bag_id, df.h_ridge, df.h_eaves, df.href_diff)

    # Store results in database 
    print('\n>> Dataset {0} -- storing {1} height references in database'.format(table, lod))

    cursor.execute("DROP TABLE IF EXISTS " + schema + ".data_import;")

    cursor.execute(
            "CREATE UNLOGGED TABLE " + schema + ".data_import(bag_id VARCHAR, " + 
            lod + "_h_ridge REAL, " + lod + "_h_eaves REAL, " + lod + "_href_diff REAL);")
    
    cursor.executemany(
        "INSERT INTO " + schema + ".data_import " + 
        "(bag_id, " + lod + "_h_ridge, " + lod + "_h_eaves, " + lod + "_href_diff) " + 
        "VALUES(%s, %s, %s, %s);", rows)

    cursor.execute("ALTER TABLE " + schema + "." + table + "_tmp ADD COLUMN IF NOT EXISTS " + lod + "_h_ridge REAL;")
    cursor.execute("ALTER TABLE " + schema + "." + table + "_tmp ADD COLUMN IF NOT EXISTS " + lod + "_h_eaves REAL;")
    cursor.execute("ALTER TABLE " + schema + "." + table + "_tmp ADD COLUMN IF NOT EXISTS " + lod + "_href_diff REAL;")

    cursor.execute(
            "UPDATE " + schema + "." + table + "_tmp " +
            "SET " + lod + "_h_ridge = d." + lod + "_h_ridge, " + 
            lod + "_h_eaves = d." + lod + "_h_eaves, " + 
            lod + "_href_diff = d." + lod + "_href_diff " + 
            "FROM " + schema + ".data_import AS d " + 
            "WHERE d.bag_id = " + schema + "." + table + "_tmp.bag_id;")

    cursor.execute("DROP TABLE " + schema + ".data_import;")


def get_surface_areas(df, cursor, schema, table, lod): 
    """
    Extract surface areas of roof and exterior walls and store results in database. 

    Parameters: \n
    df -- dataframe containing required data from 3D BAG
    cursor -- cursor to execute database queries
    schema -- schema to store the data in the database \n
    table -- table to store the data in the database \n
    lod -- string providing information on lod being processed (lod12 or lod22) \n

    Returns: none

    """

    print('\n>> Dataset {0} -- extracting {1} surface areas'.format(table, lod))

    area_roof_list = []
    area_walls_list = []
    
    for geometry, semantic_values in tqdm(zip(df.geometry, df.semantic_values), total=len(df.index)):

        # Initialise variables to store surface areas
        area_walls_sum = 0
        area_roof_sum = 0

        # Loop through each polygon stored in the multipolygon geometry
        for polygon, semantic_value in zip(geometry, semantic_values): 

            # Check for roof surfaces 
            if semantic_value == 1:

                # Calculate area and add to sum
                area_roof_sum = area_roof_sum + geom_functions.compute_area(polygon.exterior.coords)

            # Check for exterior wall surfaces
            elif semantic_value == 2:

                # Calculate area and add to sum
                area_walls_sum = area_walls_sum + geom_functions.compute_area(polygon.exterior.coords)

        # Round surface areas to 3 decimal places
        area_roof = np.around(area_roof_sum, 3)
        area_walls = np.around(area_walls_sum, 3)

        area_roof_list.append(area_roof)
        area_walls_list.append(area_walls)

    # Store results in dataframe
    df['area_roof'] = area_roof_list
    df['area_walls'] = area_walls_list

    rows = zip(df.bag_id, df.area_roof, df.area_walls)

    # Store results in database 
    print('\n>> Dataset {0} -- storing {1} surface areas in database'.format(table, lod))

    cursor.execute("DROP TABLE IF EXISTS " + schema + ".data_import;")

    cursor.execute(
            "CREATE UNLOGGED TABLE " + schema + ".data_import(bag_id VARCHAR, " + 
            lod + "_area_walls REAL, " + lod + "_area_roof REAL);")
    
    cursor.executemany(
        "INSERT INTO " + schema + ".data_import " + 
        "(bag_id, " + lod + "_area_walls, " + lod + "_area_roof) " + 
        "VALUES(%s, %s, %s);", rows)

    cursor.execute("ALTER TABLE " + schema + "." + table + "_tmp ADD COLUMN IF NOT EXISTS " + lod + "_area_walls REAL;")
    cursor.execute("ALTER TABLE " + schema + "." + table + "_tmp ADD COLUMN IF NOT EXISTS " + lod + "_area_roof REAL;")

    cursor.execute(
            "UPDATE " + schema + "." + table + "_tmp " + 
            "SET " + lod + "_area_walls = d." + lod + "_area_walls, " + lod + "_area_roof = d." + lod + "_area_roof " + 
            "FROM " + schema + ".data_import AS d " + 
            "WHERE d.bag_id = " + schema + "." + table + "_tmp.bag_id;")

    cursor.execute("DROP TABLE " + schema + ".data_import;")


def get_building_volume(df, voxel_scale, cursor, schema, table, lod):
    """"
    Extract volume of building from mesh geometry and store results in database. 
    Geometrically invalid meshes are voxelised. 

    Parameters: \n
    df -- dataframe containing required data from 3D BAG
    voxel_scale -- size of voxels
    cursor -- cursor to execute database queries
    schema -- schema to store the data in the database \n
    table -- table to store the data in the database \n
    lod -- string providing information on lod being processed (lod12 or lod22) \n

    Returns: none

    """

    print('\n>> Dataset {0} -- extracting {1} building volume'.format(table, lod))

    building_volume = []
    invalid_count = 0
    error_codes = []
    
    for geometry, bag_id in tqdm(zip(df.geometry, df.bag_id), total=len(df.index)):

        # Initialise lists to store polygon faces and vertices 
        faces = []
        boundaries = []
        vertices = []

        # Loop through each polygon stored in the multipolygon geometry
        for polygon in geometry: 

            # Initialise list to store pointers to face vertices, first value indicates face contains 3 vertices
            f = [3] 

            for xyz in polygon.exterior.coords[:-1]:
                    
                # Check if cooordinates not in polygon vertices list
                if xyz not in vertices: 
                    
                    # If not in list, append coordinates to vertices list 
                    vertices.append(xyz)

                # Find the index of each xyz coordinate vertices list and create face from indexes
                f.append(vertices.index(xyz))
                
            # Append face to list of polygon faces
            faces.append(f)
            boundaries.append([f[1:]])

        if len(vertices) > 3: 

            # Create mesh from faces and vertices 
            try: 
                mesh = pv.PolyData(np.array(vertices), np.hstack(faces))

                # Check that mesh was created
                if mesh is not None: 
                    
                    # Create tu3djson object 
                    geom_obj = create_tu3djson_object(boundaries, [list(x) for x in vertices])
                    
                    # Check validy of tu3djson object
                    if val3ditypy.is_valid_onegeom(geom_obj) == True:
                        
                        # If mesh valid, compute volume
                        volume_mesh = np.around(mesh.volume, 3)

                    else:
                        
                        # Store val3dity error codes
                        for error in val3ditypy.validate_onegeom(geom_obj)["all_errors"]: 
                            if error not in error_codes:
                                error_codes.append(error)
                        
                        # Keep track of total no. invalid meshes
                        invalid_count = invalid_count + 1

                        # Else if mesh invalid, voxelize and compute volume
                        voxels = pv.voxelize(mesh, density=mesh.length/voxel_scale, check_surface=False)

                        # Compute volume of voxels if mesh could be voxelized 
                        if voxels is not None:
                            volume_mesh = np.around(voxels.volume, 3)

                            try: 
                                # Compute volume of convex hull for comparison
                                convex_hull_volume, hull = get_conhull_volume(vertices)

                                # Check if voxelised volume > convex hull volume
                                vol_diff = volume_mesh - convex_hull_volume

                                if vol_diff > 0.1 * convex_hull_volume:

                                    # Set mesh volume equal to convex hull volume
                                    volume_mesh = convex_hull_volume

                                    print('\n >> Val3dity error code for incorrectly voxelised:', val3ditypy.validate_onegeom(geom_obj)["all_errors"])
                                    
                                    # # To visualise convex hull and voxelised mesh
                                    # print('\n\n>> BAG ID: ', bag_id, ' - ', np.around(vol_diff/convex_hull_volume * 100, 1), '% > convex hull volume \n')
                                    # plot_convex_hull(hull)
                                    # voxels.plot()

                            except Exception as error:
                                print('\nError: {0}'.format(str(error)))
                                
                        else:
                            volume_mesh = None
                    
                else:
                    print('Mesh is None')
                    volume_mesh = None

            except Exception as error:
                print('\nError: {0}'.format(str(error)))
                volume_mesh = None

        else: 
            print('Length vertices list: ', len(vertices))
            volume_mesh = None

        building_volume.append(volume_mesh)

    print('\n>> Total invalid: ', invalid_count)
    print('\n>> Percent invalid ' + lod + ': ', np.around(invalid_count / len(building_volume) * 100, 1), ' %') 
    print('\n>> Val3dity error codes: ', error_codes)

    # Store results in dataframe
    df['building_volume'] = building_volume

    rows = zip(df.bag_id, df.building_volume)

    # Store results in database  
    print('\n>> Dataset {0} -- storing {1} building volume in database'.format(table, lod))

    cursor.execute("DROP TABLE IF EXISTS " + schema + ".data_import;")

    cursor.execute("CREATE UNLOGGED TABLE " + schema + ".data_import(bag_id VARCHAR, " + lod + "_volume REAL);")
    
    cursor.executemany(
        "INSERT INTO " + schema + ".data_import " + 
        "(bag_id, " + lod + "_volume) " + 
        "VALUES(%s, %s);", rows)

    cursor.execute("ALTER TABLE " + schema + "." + table + "_tmp ADD COLUMN IF NOT EXISTS " + lod + "_volume REAL;")

    cursor.execute(
            "UPDATE " + schema + "." + table + "_tmp " + 
            "SET " + lod + "_volume = d." + lod + "_volume " + 
            "FROM " + schema + ".data_import AS d " + 
            "WHERE d.bag_id = " + schema + "." + table + "_tmp.bag_id;")

    cursor.execute("DROP TABLE " + schema + ".data_import;")


def get_conhull_volume(vertices):
    """
    Compute the convex hull from the input vertices and determine its volume. 

    Parameters: \n
    vertices -- list of 3d vertices 

    Returns: \n
    volume_ch --- volume of convex hull 
    con_hull -- geometry of convex hull 

    """

    x = []
    y = []
    z = []

    if len(vertices) > 0:

        for vertex in vertices:
            x.append(vertex[0])
            y.append(vertex[1])
            z.append(vertex[2])

        con_hull = ConvexHull(np.column_stack((x, y, z)))
        volume_ch = np.around(con_hull.volume, 3)

    else:
        # Set convex hull volume equal to None 
        volume_ch = None

    return volume_ch, con_hull


def plot_convex_hull(hull):
    """
    Plot a convex hull. 
    Based on https://gist.github.com/flutefreak7/bd621a9a836c8224e92305980ed829b9

    """

    faces = np.column_stack((3*np.ones((len(hull.simplices), 1), dtype=np.int64), hull.simplices)).flatten()
    poly = pv.PolyData(hull.points, faces)

    poly.plot()


def create_tu3djson_object(boundaries, vertices):
    """
    Create a tu3djson object from the boundaries and vertices of a mesh. 
    Based on https://github.com/tudelft3d/tu3djson#tu3djson-object

    """ 

    tu3djson_geom_obj =  {"type": "CompositeSurface", "boundaries": boundaries, "vertices": vertices}

    return tu3djson_geom_obj


def main(params):

    # Load parameters 
    jparams = json.load(open(params))

    # Schema where training data is stored
    train_schema = 'training_data'

    for table in jparams["training_tables"]:

        starttime = time()

        # Create connection to database
        conn = db_functions.setup_connection()
        conn.autocommit = True

        # Create a cursor (allows PostgreSQL commands to be executed)
        curs = conn.cursor()

        # Create temporary table to store extracted data in
        db_functions.create_temp_table(curs, train_schema, table, pkey='bag_id')

        # Get required data from database
        df_lod1 = get_lod1_data(conn, train_schema, table)
        df_lod2 = get_lod2_data(conn, train_schema, table)

        # Get LOD2 height references (ridge/eaves)
        get_lod2_hrefs(df_lod2, curs, train_schema, table, 'lod22')

        # Get surface areas
        get_surface_areas(df_lod1, curs, train_schema, table, 'lod12')
        get_surface_areas(df_lod2, curs, train_schema, table, 'lod22')

        # Get building volume
        get_building_volume(df_lod1, jparams["voxel_scales"]["LOD12"], curs, train_schema, table, 'lod12')
        get_building_volume(df_lod2, jparams["voxel_scales"]["LOD22"], curs, train_schema, table, 'lod22')

        # Replace original table with unlogged temporary table
        db_functions.replace_temp_table(curs, train_schema, table, pkey='bag_id')

        # Close database connection
        db_functions.close_connection(conn, curs)

        endtime = time()
        duration = endtime - starttime
        print('\n>> Computation time: ', round(duration, 2), 's \n\n' + 10*'-')


if __name__ == '__main__':
    main(sys.argv[1])