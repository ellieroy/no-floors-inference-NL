"""
Various functions to compute intermediate geometric features.
Partly based on: Imke Lansky (https://github.com/ImkeLansky/USA-BuildingHeightInference)

"""

import db_functions


def compute_buffers(cursor, schema, table, size):
    """
    Compute buffers around all footprint geometries and store results in database. 
    
    Parameters: \n
    cursor -- cursor for database connection \n
    schema -- schema to store the features in the database \n
    table -- table to store the features in the database \n
    size -- size of buffer (meters) \n

    Returns: none

    """

    print('\n>> Dataset {0} -- computing buffers of {1}m around footprints'.format(table, size))

    cursor.execute("ALTER TABLE " + schema + "." + table + "_tmp ADD COLUMN IF NOT EXISTS footprint_buffer GEOMETRY;")

    cursor.execute(
        "UPDATE " + schema + "." + table + "_tmp " + 
        "SET footprint_buffer = subquery.buffer " + 
        "FROM " + 
            "(SELECT bag_id, ST_Buffer(footprint_geom, " + str(size) + ", 'join=mitre') AS buffer " + 
            "FROM " + schema + "." + table + "_tmp) AS subquery " + 
        "WHERE " + schema + "." + table + "_tmp.bag_id = subquery.bag_id;"  
    )

    cursor.execute(
        "CREATE INDEX IF NOT EXISTS " + table + "_buf_idx_tmp " + 
        "ON " + schema + "." + table + "_tmp " + 
        "USING GIST (footprint_buffer);"
    )


def compute_centroids(cursor, schema, table): 
    """
    Compute centroids of all footprint geometries and store results in database. 
    
    Parameters: 
    cursor -- cursor for database connection \n
    schema -- schema to store the features in the database \n
    table -- table to store the features in the database \n

    Returns: none
    
    """

    print('\n>> Dataset {0} -- computing centroids of footprints'.format(table))

    cursor.execute("ALTER TABLE " + schema + "." + table + "_tmp ADD COLUMN IF NOT EXISTS footprint_centroid GEOMETRY;")   

    cursor.execute(
        "UPDATE " + schema + "." + table + "_tmp " +  
        "SET footprint_centroid = subquery.centroid " + 
        "FROM " + 
            "(SELECT bag_id, ST_Centroid(footprint_geom) as centroid " + 
            "FROM " + schema + "." + table + "_tmp) AS subquery " + 
        "WHERE " + schema + "." + table + "_tmp.bag_id = subquery.bag_id;"  
    )

    cursor.execute(
        "CREATE INDEX IF NOT EXISTS " + table + "_centroid_idx_tmp " + 
        "ON " + schema + "." + table + "_tmp " + 
        "USING GIST (footprint_centroid);"
    )


def get_gemeente_polygon(connection, gemeente_code, buffer=0): 
    """
    Obtain the WKT representation of the polygon corresponding to the input municipality. 
    It is possible to set an optional buffer around the polygon. 

    Parameters: 
    connection -- database connection \n
    gemeente_code -- code of input municipality \n
    buffer -- optional buffer (default = 0) (meters) \n

    Returns: WKT representation of geometry corresponding to municipality code (string)

    """

    if buffer != 0 :
        gm_df = db_functions.read_spatial_data(connection, 'cbs', 'gemeenten2019', where="WHERE gemeentecode = concat('GM', '" + gemeente_code + "')", columns="gemeentecode, ST_Buffer(geom, " + str(buffer) + ", 'join=mitre') AS geom")
    else: 
        gm_df = db_functions.read_spatial_data(connection, 'cbs', 'gemeenten2019', where="WHERE gemeentecode = concat('GM', '" + gemeente_code + "')", columns="gemeentecode, geom")
        
    # Dissolve polygons belonging to the same municipality 
    gm_df = gm_df.dissolve(by='gemeentecode')

    # Simplify the geometry using Douglas-Peucker
    gm_df['geom'] = gm_df['geom'].simplify(100)

    # Return geometry
    return gm_df.geom.values


def det(a):
    """
    Compute determinant of matrix a.

    """

    return a[0][0]*a[1][1]*a[2][2] + a[0][1]*a[1][2]*a[2][0] + a[0][2]*a[1][0]*a[2][1] - a[0][2]*a[1][1]*a[2][0] - a[0][1]*a[1][0]*a[2][2] - a[0][0]*a[1][2]*a[2][1]


def unit_normal(a, b, c):
    """
    Compute unit normal vector of plane defined by points a, b, and c.

    """

    x = det([[1,a[1],a[2]],
             [1,b[1],b[2]],
             [1,c[1],c[2]]])
    y = det([[a[0],1,a[2]],
             [b[0],1,b[2]],
             [c[0],1,c[2]]])
    z = det([[a[0],a[1],1],
             [b[0],b[1],1],
             [c[0],c[1],1]])
    magnitude = (x**2 + y**2 + z**2)**.5
    return (x/magnitude, y/magnitude, z/magnitude)


def dot(a, b):
    """
    Compute dot product of vectors a and b.
    
    """

    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]


def cross(a, b):
    """
    Compute cross product of vectors a and b. 

    """
    
    x = a[1] * b[2] - a[2] * b[1]
    y = a[2] * b[0] - a[0] * b[2]
    z = a[0] * b[1] - a[1] * b[0]
    return (x, y, z)


def compute_area(poly):
    """ 
    Compute area of polygon in 3D.
    
    """

    if len(poly) < 3: # not a plane - no area
        return 0

    total = [0, 0, 0]
    for i in range(len(poly)):
        vi1 = poly[i]
        if i is len(poly)-1:
            vi2 = poly[0]
        else:
            vi2 = poly[i+1]
        prod = cross(vi1, vi2)
        total[0] += prod[0]
        total[1] += prod[1]
        total[2] += prod[2]
    result = dot(total, unit_normal(poly[0], poly[1], poly[2]))
    return abs(result/2)
