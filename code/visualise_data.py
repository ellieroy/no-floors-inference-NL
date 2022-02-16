"""
Reads training data from database into dataframe and visualises results using different plots. 

"""

from os import path
import json, sys
import db_functions
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd


def directory_exists(name):
    """
    Check whether a directory exists. 
    
    Parameters: \n
    name -- relative path of directory \n

    Returns: True / False 

    """

    # Check if directory to save figures exists
    if path.exists(name) and path.isdir(name):
        return True

    else: 
        return False


def no_floors_hist(conn):
    """
    Plot histogram of number of floors in training dataset. 

    Parameters: \n
    conn -- connection to database \n
    
    Returns: None 

    """

    # Read training data from database 
    c1_ams = db_functions.read_data(conn, 'training_data', 'c1_ams', where='where is_clean is not null')
    c2_rot = db_functions.read_data(conn, 'training_data', 'c2_rot', where='where is_clean is not null')
    c3_dhg = db_functions.read_data(conn, 'training_data', 'c3_dhg', where='where is_clean is not null')
    c4_rho = db_functions.read_data(conn, 'training_data', 'c4_rho', where='where is_clean is not null')

    frames = [c1_ams, c2_rot, c3_dhg, c4_rho]

    df_all = pd.concat(frames)

    df_all.hist(bins=np.arange(max(df_all.clean_floors)+2)-0.5, column=['clean_floors'], log=True, grid=False, edgecolor='white', linewidth=0.8, color='steelblue', figsize=[8.6, 4.8], alpha=0.7)
    x_ticks = np.arange(1, max(df_all.clean_floors)+1, 2)
    plt.xticks(x_ticks)
    plt.xlabel('Number of floors')
    plt.ylabel('Count')
    plt.title('')
    plt.savefig('plots/histograms/hist_floors_all.pdf', dpi=300)
    plt.close()

    print('No. floors 90th percentile: {0}'.format(np.percentile(df_all.clean_floors, 90)))

    for df in frames:

        df.hist(bins=np.arange(max(df.clean_floors)+2)-0.5, column=['clean_floors'], log=True, grid=False, edgecolor='white', linewidth=0.8, color='steelblue', figsize=[8.6, 4.8], alpha=0.7)
        x_ticks = np.arange(1, max(df.clean_floors)+1, 2)
        plt.xticks(x_ticks)
        plt.xlabel('Number of floors')
        plt.ylabel('Count')
        plt.title('')
        plt.show()


def pie_chart_functions(conn, jparams):
    """
    Plot pie chart showing different building functions in training dataset. 
    Do not run unless all building functions are present in database. 

    Parameters: \n
    conn -- connection to database \n
    jparams -- dictionary of parameters from json parameter file \n
    
    Returns: None 

    """

    cols = 'bag_id, bag_function'
    schema = jparams["training_schema"]

    c1_ams = db_functions.read_data(conn, schema, 'c1_ams', columns=cols)
    c2_rot = db_functions.read_data(conn, schema, 'c2_rot', columns=cols)
    c3_dhg = db_functions.read_data(conn, schema, 'c3_dhg', columns=cols)
    c4_rho = db_functions.read_data(conn, schema, 'c4_rho', columns=cols)
    frames = [c1_ams, c2_rot, c3_dhg, c4_rho]

    df_all = pd.concat(frames)

    counts = df_all.groupby("bag_function")['bag_id'].count()
    counts['Unknown'] = counts['Others'] + counts['Unknown']
    counts['Non-residential'] = counts['Non-residential (single-function)'] + counts['Non-residential (multi-function)']
    counts.drop(labels='Others', inplace=True)
    counts.drop(labels=['Non-residential (single-function)', 'Non-residential (multi-function)'], inplace=True)
    labels = counts.keys()

    colours = sns.color_palette("Blues", len(labels))

    pie, ax = plt.subplots(figsize=[10,8])
    plt.pie(x=counts, autopct="%.1f%%", labels=labels, explode=[0.05]*len(labels), colors=colours, pctdistance=0.7)
    # plt.show()
    pie.savefig('plots/functions_all_data.pdf', dpi=300)

    for table in jparams["training_tables"]: 

        data = db_functions.read_data(conn, schema, table, columns=cols)

        counts = data.groupby("bag_function")['bag_id'].count()
        counts['Unknown'] = counts['Others'] + counts['Unknown']
        counts['Non-residential'] = counts['Non-residential (single-function)'] + counts['Non-residential (multi-function)']
        counts.drop(labels='Others', inplace=True)
        counts.drop(labels=['Non-residential (single-function)', 'Non-residential (multi-function)'], inplace=True)

        if table == 'c3_dhg': 
            counts['Other'] = counts['Non-residential'] + counts['Mixed-residential'] + counts['Unknown']
            counts.drop(labels=['Non-residential', 'Mixed-residential', 'Unknown'], inplace=True)

        elif table == 'c4_rho':
            counts['Other'] = counts['Non-residential'] + counts['Unknown']
            counts.drop(labels=['Non-residential', 'Unknown'], inplace=True)
        
        labels = counts.keys()

        colours = sns.color_palette("Blues", len(labels))

        pie, ax = plt.subplots(figsize=[10,8])
        plt.pie(x=counts, autopct="%.1f%%", labels=labels, explode=[0.05]*len(labels), colors=colours, pctdistance=0.7)
        # plt.show()
        pie.savefig('plots/functions_' + table + '.pdf', dpi=300)


def pie_chart_td_cities(conn, jparams):
    """
    Plot piechart showing the fraction of training data originating from each municipality. 

    Parameters: \n
    conn -- connection to database \n
    jparams -- dictionary of parameters from json parameter file \n
    
    Returns: None 

    """

    cols = 'bag_id'
    schema = jparams["training_schema"]

    c1_ams = db_functions.read_data(conn, schema, 'c1_ams', columns=cols, where='where is_clean is not null')
    c2_rot = db_functions.read_data(conn, schema, 'c2_rot', columns=cols, where='where is_clean is not null')
    c3_dhg = db_functions.read_data(conn, schema, 'c3_dhg', columns=cols, where='where is_clean is not null')
    c4_rho = db_functions.read_data(conn, schema, 'c4_rho', columns=cols, where='where is_clean is not null')

    c1_ams['city'] = 'amsterdam'
    c2_rot['city'] = 'rotterdam'
    c3_dhg['city'] = 'den haag'
    c4_rho['city'] = 'rijssen-holten'

    frames = [c1_ams, c2_rot, c3_dhg, c4_rho]

    df_all = pd.concat(frames)

    counts = df_all.groupby("city")['bag_id'].count()
    print(counts)
    labels = counts.keys()

    colours = sns.color_palette("Blues", len(labels))

    pie, ax = plt.subplots(figsize=[10,8])
    plt.rcParams.update({'font.size': 18})
    plt.pie(x=counts, autopct="%.1f%%", labels=labels, explode=[0.05]*len(labels), colors=colours, pctdistance=0.7)
    plt.show()
    pie.savefig('plots/td_city_split.pdf', dpi=300)


def main(params):

    # Load parameters 
    jparams = json.load(open(params))

    # Set up connection to database
    conn = db_functions.setup_connection()
    curs = conn.cursor()

    # Make plots 
    no_floors_hist(conn)
    # pie_chart_functions(conn, jparams)
    pie_chart_td_cities(conn, jparams)
    
    # Close connection 
    db_functions.close_connection(conn, curs)


if __name__ == '__main__':

    main(sys.argv[1])