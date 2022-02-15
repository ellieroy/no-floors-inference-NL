"""
Script to generate plots used to determine optimal voxel size to use.
Based on comparison of median difference between voxelised volume and mesh volume 
and median runtime per building for valid geometries. 

"""

import json, sys
import os
from time import time
import pandas as pd
import geopandas as gpd
import numpy as np
import pyvista as pv
import val3ditypy
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
import db_functions 
from extract_3d_features import create_tu3djson_object
from visualise_data import directory_exists

def main(train_schema, train_tables, get_data, voxel_sizes, no_rows):
    """
    If get_data is True: compute the volume of each geometrically valid mesh, 
    voxelise each geometrically valid mesh using different voxel sizes and compute the voxelised volumes,
    record the time taken to voxelise each mesh and store results to csv. 

    If get_data is False: load results from csv and make plots of median volume difference 
    and median runtime per building for lod 1.2 and 2.2. 

    Parameters: \n
    train_schema -- name of database schema where trainin data stored \n
    train_tables -- names of tables where training data stored \n
    get_data -- True/False, whether to obtain required data from database or load data from csv \n
    voxel_sizes -- voxel sizes to test \n
    no_rows -- number of rows to read from each training data table in the database \n

    Returns: none
    
    """

    if get_data == True:

        # Create connection to database
        conn = db_functions.setup_connection()
        conn.autocommit = True

        # Create a cursor (allows PostgreSQL commands to be executed)
        curs = conn.cursor()

        # Create empty geopandas GeoDataFrames to store all training data
        lod1_df = gpd.GeoDataFrame()
        lod2_df = gpd.GeoDataFrame()

        for table in train_tables:

            print('\n>> Dataset {0} -- obtaining 3D BAG LOD1 data'.format(table))

            query1 = ("SELECT lod12_3d.fid, t.bag_id, lod12_3d.geometrie " + 
                    "FROM " + train_schema + "." + table + " AS t " + 
                    "INNER JOIN bag3d.lod12_3d ON lod12_3d.fid = t.bag3d_id " + 
                    "WHERE t.lod12_volume != 'NaN' "
                    "LIMIT " + str(no_rows) + ";") 

            lod1_data = gpd.GeoDataFrame.from_postgis(query1, conn, geom_col='geometrie')

            lod1_df = lod1_df.append(lod1_data)

            print('\n>> Dataset {0} -- obtaining 3D BAG LOD2 data'.format(table))

            query2 = ("SELECT lod22_3d.fid, t.bag_id, lod22_3d.geometrie " + 
                    "FROM " + train_schema + "." + table + " AS t " + 
                    "INNER JOIN bag3d.lod22_3d ON lod22_3d.fid = t.bag3d_id " + 
                    "WHERE t.lod22_volume != 'NaN' " + 
                    "LIMIT " + str(no_rows) + ";")  

            lod2_data = gpd.GeoDataFrame.from_postgis(query2, conn, geom_col='geometrie')

            lod2_df = lod2_df.append(lod2_data)

        # Close database connection
        db_functions.close_connection(conn, curs)

        i = 0 
        
        for df in [lod1_df, lod2_df]:

            i += 1
            print('\n>> {0} -- extracting mesh surfaces and calculating voxelised volumes'.format('LOD' + str(i)))

            # Create lists to store calculated volumes and runtimes
            volume_lists = [ [] for _ in range(len(voxel_sizes))]
            runtime_lists = [ [] for _ in range(len(voxel_sizes))]
            mesh_volumes = []
            
            # Loop through each row (building) in the dataframe
            for geom in df.geometrie: 

                # Create lists to store polygon faces and vertices for volume calculation
                polygon_faces = []
                polygon_vertices = []
                polygon_boundaries = []

                # Loop through each polygon stored in the multipolygon geometry
                for polygon in geom: 

                    # Declare list to store polygon face in 
                    face = [3]

                    # Loop through each polygon coordinate 
                    for xyz in polygon.exterior.coords[:-1]:
                            
                        # Check if cooordinates not in polygon vertices list
                        if xyz not in polygon_vertices: 
                            
                            # If not in list, append coordinates to vertices list 
                            polygon_vertices.append(xyz)
                        
                        # Find the index of each xyz coordinate vertices list and create face from indexes
                        face.append(polygon_vertices.index(xyz))
                        
                    # Append face to list of polygon faces
                    polygon_faces.append(face)
                    polygon_boundaries.append([face[1:]])

                # Create tu3djson object
                geom_obj = create_tu3djson_object(polygon_boundaries, [list(x) for x in polygon_vertices])

                # Check validity of geometry
                if val3ditypy.is_valid_onegeom(geom_obj) == True:
                    
                    # Create mesh from vertices and faces 
                    mesh = pv.PolyData(np.array(polygon_vertices), np.hstack(polygon_faces))

                    # Calculate volume of mesh 
                    mesh_volume = np.around(mesh.volume, 3)
                    mesh_volumes.append(mesh_volume)
            
                    # Loop through all input voxel sizes 
                    for voxel_size, volume_list, runtime_list in zip(voxel_sizes, volume_lists, runtime_lists): 
                        
                        starttime = time()

                        # Voxelise surface, compute volume of voxels and store in volume list
                        voxels = pv.voxelize(mesh, density=mesh.length/voxel_size, check_surface=False)
                        voxel_volume = np.around(voxels.volume, 3)

                        endtime = time()
                        duration = round(endtime - starttime, 2)

                        volume_list.append(voxel_volume)
                        runtime_list.append(duration)

            # Create new dataframe to store calculated volumes and runtimes for each voxel size
                df_out = pd.DataFrame(mesh_volumes, columns=['mesh_volume'])

            # Loop through each voxel size and store data 
            for voxel_size, volume_list, runtime_list in zip(voxel_sizes, volume_lists, runtime_lists): 
                
                df_out['%s' % ('volume_' + str(voxel_size))] = volume_list
                df_out['%s' % ('runtime_' + str(voxel_size))] = runtime_list

            # Check directory to save data exists
            if not directory_exists('./data/voxel_size'):
                os.mkdir('./data/voxel_size')

            df_out.to_csv('./data/voxel_size/lod{0}2_volumes.csv'.format(i))

    else: 

        # Read CSV files 
        file_names = ['lod12_volumes.csv', 'lod22_volumes.csv']
        
        for file_name in file_names:
            
            if os.path.exists('./data/voxel_size/' + file_name):

                # Read data from file 
                df = pd.read_csv('./data/voxel_size/{0}'.format(file_name))

                vol_diff_50p = []
                runtime_50p = []

                # Compute median volume difference runtime for each voxel size
                for voxel_size in voxel_sizes:

                    vol_diff = abs(df['mesh_volume'] - df['volume_{0}'.format(voxel_size)])
                    vol_diff_50p.append(np.percentile(vol_diff, 50))

                    runtime = df['runtime_{0}'.format(voxel_size)]
                    runtime_50p.append(np.percentile(runtime, 50))

                # Plot voxel size against median volume difference and runtime
                fig, ax = plt.subplots()

                x = voxel_sizes
                y1 = vol_diff_50p
                y2 = runtime_50p

                X_Y_Spline1 = make_interp_spline(x, y1)
                X_Y_Spline2 = make_interp_spline(x, y2)

                # Returns evenly spaced number over a specified interval
                X_ = np.linspace(min(x), max(x), 500)
                Y_1 = X_Y_Spline1(X_)
                Y_2 = X_Y_Spline2(X_)

                lns1 = ax.plot(X_, Y_1, 'darkslategrey', label='Median volume difference')
                ax.set_ylabel('Median volume difference (m$^3$)')
                ax.set_xlabel('Number of voxels in mesh length')
        
                ax2 = ax.twinx()
                lns2 = ax2.plot(X_, Y_2, 'cadetblue', label='Median runtime')
                ax2.set_ylabel('Median runtime (s)')

                lns = lns1 + lns2
                labs = [l.get_label() for l in lns]
                ax.legend(lns, labs, loc='upper center')

                # Check directory to save plots exists
                if not directory_exists('./plots/voxel_size'):
                    os.mkdir('./plots/voxel_size')

                plt.savefig('plots/voxel_size/' + file_name[0:5] + '_voxelsize.png', dpi=300)
                plt.close()
            

if __name__ == '__main__':

    voxel_sizes = [50, 100, 150, 200, 250]

    no_rows = 1

    get_data = False

    jparams = json.load(open(sys.argv[1]))

    train_schema = jparams["training_schema"]
    train_tables = jparams["training_tables"]
    
    main(train_schema, train_tables, get_data, voxel_sizes, no_rows)