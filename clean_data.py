import sys, os
import db_functions
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from time import time
from visualise_data import directory_exists


def remove_outer_percentiles(df, p_lower, p_upper, attribute):
    """
    Function to remove outliers per floor based on input pecentiles. 

    Parameters: \n
    df -- dataframe containing outliers \n
    p_lower -- lower percentile / lower boundary to remove outliers from \n
    p_upper -- upper percentile / upper boundary to remove outliers from \n
    attribute -- attribute based on which outliers are removed \n

    Returns: dataframe with column describing whether points fall inside the upper and lower boundaries 

    """

    df_output= pd.DataFrame()

    for value in sorted(df['no_floors'].unique()): 

        df_per_floor = df.loc[df['no_floors'] == value]
        
        y = df_per_floor[attribute]
        removed_outliers = y.between(y.quantile(p_lower), y.quantile(p_upper))
        df_per_floor['is_inlier'] = removed_outliers

        df_output = df_output.append(df_per_floor)
    
    return df_output


def remove_outliers(df, attribute):
    """
    Function to remove outliers per floor based on 1.5 IQR. 
    Points are considered outliers if they fall outside Q1 - 1.5 IQR or Q3 + 1.5 IQR 

    Parameters: \n
    df -- dataframe containing outliers \n
    attribute -- attribute based on which outliers are removed \n

    Returns: dataframe with column describing whether points are outliers 
    
    """

    df_output = pd.DataFrame()

    for value in sorted(df['no_floors'].unique()): 

        df_per_floor = df.loc[df['no_floors'] == value].copy()

        q75, q25 = np.percentile(df_per_floor[attribute], [75, 25])
        iqr = 1.5 * (q75 - q25)

        df_per_floor['is_outlier'] = df_per_floor[attribute].apply(lambda x: True if x < q25 - iqr or x > q75 + iqr else False)
        
        df_output = df_output.append(df_per_floor)

    return df_output


def plot_distribution(df, attribute, title, ylabel):
    """
    Plot distribution of data using box plots and violin plots. 

    Parameters: \n
    df -- dataframe containing all data \n
    attribute -- attribute used to generate plots (must be column of df) \n
    title -- tile of plots \n
    ylabel -- label of y-axis \n

    Returns: None

    """

    if attribute == 'h_70p':
        y_ticks = np.arange(0, max(df[attribute]+1), 10)

    elif attribute == 'avg_h_storey':
        y_ticks = np.arange(0, max(df[attribute])+1, 2)

    # Check directory for plots exists
    if not directory_exists('plots/data_cleaning/'):
        os.makedirs('./plots/data_cleaning')

    # Box plot
    ax = sns.boxplot(x="no_floors", y=attribute, data=df, palette='Blues', whis=1.5, linewidth=0.5)
    for patch in ax.artists:
        r, g, b, a = patch.get_facecolor()
        patch.set_facecolor((r, g, b, .6))
    plt.xlabel('Number of floors')
    plt.ylabel(ylabel)
    plt.yticks(y_ticks)
    plt.savefig('plots/data_cleaning/box_' + title + '.pdf', dpi=300)
    plt.close()

    # Violin plot
    sns.violinplot(x='no_floors', y=attribute, hue='bag_function', hue_order=['Residential', 'Mixed-residential'], data=df, linewidth=0.75, palette='PuBu', split=True)
    plt.xlabel('Number of floors')
    plt.ylabel(ylabel)
    plt.yticks(y_ticks)
    plt.legend(title='Building function')
    plt.savefig('plots/data_cleaning/violin_function_' + title + '.pdf', dpi=300)
    plt.close()

    # Violin plot
    sns.violinplot(x='no_floors', y=attribute, hue='bag3d_roof_type', hue_order=['slanted','horizontal'], data=df, linewidth=0.75, palette='GnBu', split=True)
    plt.xlabel('Number of floors')
    plt.ylabel(ylabel)
    plt.yticks(y_ticks)
    plt.legend(title='Roof type')
    plt.savefig('plots/data_cleaning/violin_roof_' + title + '.pdf', dpi=300)
    plt.close()


def main(params):

    conn = db_functions.setup_connection()
    conn.autocommit = True
    curs = conn.cursor()

    cols = 'bag_id, no_floors, h_70p, bag_function, bag3d_roof_type'

    # Read training data from database 
    c1_ams = db_functions.read_data(conn, 'training_data', 'c1_ams', columns=cols)
    c2_rot = db_functions.read_data(conn, 'training_data', 'c2_rot', columns=cols)
    c3_dhg = db_functions.read_data(conn, 'training_data', 'c3_dhg', columns=cols)
    c4_rho = db_functions.read_data(conn, 'training_data', 'c4_rho', columns=cols)

    # Create column to store data source
    c1_ams['data_source'] = 'c1_ams'
    c2_rot['data_source'] = 'c2_rot'
    c3_dhg['data_source'] = 'c3_dhg'
    c4_rho['data_source'] = 'c4_rho'

    # Combine all dataframes
    frames = [c1_ams, c2_rot, c3_dhg, c4_rho]
    data_all = pd.concat(frames)

    # Drop any null values
    df = data_all.dropna()

    print('\n' + 10*'-' + '\n\n>> Performing height-based cleaning steps \n\n' + 10*'-')

    # Filter dataframe for intervals with enough data 
    df_1 = df.loc[df['no_floors'] <= 17]

    # Get roof type 
    df_1 = df_1.loc[(df_1['bag3d_roof_type'] == 'slanted') | (df_1['bag3d_roof_type'] == 'multiple horizontal') | (df_1['bag3d_roof_type'] == 'single horizontal')]
    df_1['bag3d_roof_type'] = df_1['bag3d_roof_type'].apply(lambda x: 'slanted' if x == 'slanted' else 'horizontal')

    # Compute average storey height
    df_1['avg_h_storey'] = df_1['h_70p'] / df_1['no_floors']
    
    # Visualise data before data cleaning 
    plot_distribution(df_1, 'h_70p', 'height_0', 'Building height (m)')
    plot_distribution(df_1, 'avg_h_storey', 'storey_0', 'Average storey height (m)')
    total_in = len(df_1)    

    # Remove buildings where average storey height < 1.5 m 
    df_1['avg_storey_>1.5m'] = df_1['avg_h_storey'].apply(lambda x: True if x > 1.5 else False)
    df_cleaned_1 = df_1.loc[df_1['avg_storey_>1.5m'] == True]
    df_outlier_1 = df_1.loc[df_1['avg_storey_>1.5m'] == False]
    plot_distribution(df_cleaned_1, 'h_70p', 'height_1', 'Building height (m)')
    plot_distribution(df_cleaned_1, 'avg_h_storey', 'storey_1', 'Average storey height (m)')
    
    # Remove building height outliers (using 1.5 IQR) 
    df_2 = remove_outliers(df_cleaned_1, 'h_70p')
    df_cleaned_2 = df_2.loc[df_2['is_outlier'] == False].drop(['is_outlier'], axis=1)
    df_outlier_2 = df_2.loc[df_2['is_outlier'] == True]
    plot_distribution(df_cleaned_2, 'h_70p', 'height_2', 'Building height (m)')
    plot_distribution(df_cleaned_2, 'avg_h_storey', 'storey_2', 'Average storey height (m)')

    # Remove building height outliers (using 1.5 IQR) 
    df_3 = remove_outliers(df_cleaned_2, 'h_70p')
    df_cleaned_3 = df_3.loc[df_3['is_outlier'] == False].drop(['is_outlier'], axis=1)
    df_outlier_3 = df_3.loc[df_3['is_outlier'] == True]
    plot_distribution(df_cleaned_3, 'h_70p', 'height_3', 'Building height (m)')
    plot_distribution(df_cleaned_3, 'avg_h_storey', 'storey_3', 'Average storey height (m)')

    total_out = len(df_cleaned_3)

    # Check directory for plots exists
    if not directory_exists('plots/data_cleaning/'):
        os.makedirs('./plots/data_cleaning')

    # Plot kept vs. removed
    ax1 = df_outlier_1.plot(kind='scatter', x='no_floors', y='h_70p', color='lightcoral', alpha=0.4, label='Removed')    
    ax2 = df_outlier_2.plot(kind='scatter', x='no_floors', y='h_70p', color='lightcoral', alpha=0.4, ax=ax1, legend=False)    
    ax3 = df_outlier_3.plot(kind='scatter', x='no_floors', y='h_70p', color='lightcoral', alpha=0.4, ax=ax1, legend=False)
    ax4 = df_cleaned_3.plot(kind='scatter', x='no_floors', y='h_70p', color='steelblue', alpha=0.4, ax=ax1, label='Kept')
    plt.legend()
    plt.xlabel('No. floors')
    plt.ylabel('Building height (m)')
    plt.savefig('plots/data_cleaning/kept_vs_removed.png', dpi=300)
    plt.close()

    print('\n>> Step 1 removed:', len(df_outlier_1))
    print('\n>> Step 2 removed:', len(df_outlier_2))
    print('\n>> Step 3 removed:', len(df_outlier_3))
    print('\n' + 10*'-')

    # Split cleaned data into original data sources 
    c1_ams_clean = df_cleaned_3.loc[df_cleaned_3['data_source'] == 'c1_ams'].copy()
    c2_rot_clean = df_cleaned_3.loc[df_cleaned_3['data_source'] == 'c2_rot'].copy()
    c3_dhg_clean = df_cleaned_3.loc[df_cleaned_3['data_source'] == 'c3_dhg'].copy()
    c4_rho_clean = df_cleaned_3.loc[df_cleaned_3['data_source'] == 'c4_rho'].copy()

    clean_frames = [c1_ams_clean, c2_rot_clean, c3_dhg_clean, c4_rho_clean]

    train_schema = 'training_data'
    
    for clean_frame in clean_frames:

        starttime = time()

        # Get name of table to store results in
        table = clean_frame['data_source'].unique()[0]

        # Add column indicating that data is clean
        clean_frame['is_clean'] = 'y'

        # Drop any unnecessary columns
        clean_frame.drop(['h_70p', 'bag_function', 'bag3d_roof_type', 'data_source', 'avg_h_storey', 'avg_storey_>1.5m'], axis=1, inplace=True)

        # If csv exists, perform manual cleaning for amsterdam and rotterdam
        if (table == 'c1_ams' or table == 'c2_rot') and os.path.exists('data/cleaning/' + table + '.csv'): 

            print('\n>> Dataset {0} -- performing manual cleaning steps'.format(table))

            # Obtain manual cleaning results (buildings > 17 floors)
            df_manual = pd.read_csv('data/cleaning/' + table + '.csv', dtype={'bag_id':str, 'td_floors':int, 'is_clean':str, 'actual_floors':int})
            df_manual['no_floors'] = np.where(df_manual['is_clean'] == 'y', df_manual['td_floors'], df_manual['actual_floors'])
            df_manual.drop(['td_floors', 'actual_floors'], axis=1, inplace=True)
            
            # Join semi-automatic and manual cleaning results
            clean_output = pd.concat([clean_frame, df_manual])

        else: # other cities didn't require manual cleaning
            
            print('\n>> Dataset {0} -- no manual cleaning steps performed'.format(table))

            # Clean output is the same as original df
            clean_output = clean_frame

        # Create temporary table to store extracted data in
        db_functions.create_temp_table(curs, train_schema, table, pkey='bag_id')

        # Create temporary table to import data
        print('\n>> Dataset {0} -- storing data cleaning results in database'.format(table))
        curs.execute("DROP TABLE IF EXISTS " + train_schema + ".data_import;")
        curs.execute("CREATE UNLOGGED TABLE " + train_schema + ".data_import (bag_id VARCHAR, is_clean VARCHAR, clean_floors INTEGER);")

        # Insert data into temporary import table
        rows = zip(clean_output.bag_id, clean_output.is_clean, clean_output.no_floors)
        curs.executemany(
            "INSERT INTO " + train_schema + ".data_import (bag_id, is_clean, clean_floors) " + 
            "VALUES(%s, %s, %s);", rows)
        
        # Create primary key on temporary import table
        curs.execute("ALTER TABLE " + train_schema + ".data_import ADD PRIMARY KEY (bag_id);")

        # Create new columns in training data temp table 
        curs.execute("ALTER TABLE " + train_schema + "." + table + "_tmp DROP COLUMN IF EXISTS is_clean;")
        curs.execute("ALTER TABLE " + train_schema + "." + table + "_tmp ADD COLUMN is_clean VARCHAR;")
        curs.execute("ALTER TABLE " + train_schema + "." + table + "_tmp DROP COLUMN IF EXISTS clean_floors;")
        curs.execute("ALTER TABLE " + train_schema + "." + table + "_tmp ADD COLUMN clean_floors INTEGER;")

        # Update new column with is_clean and clean_floors values
        curs.execute(
            "UPDATE " + train_schema + "." + table + "_tmp " +
            "SET is_clean = d.is_clean, clean_floors = d.clean_floors " + 
            "FROM " + train_schema + ".data_import AS d " + 
            "WHERE d.bag_id = " + train_schema + "." + table + "_tmp.bag_id;")

        # Perform other cleaning steps: 
        print('\n>> Dataset {0} -- performing additional cleaning steps'.format(table))

        # Clean building height above 17 floors
        curs.execute(
            "UPDATE " + train_schema + "." + table + "_tmp " + 
            "SET is_clean = NULL " + 
            "WHERE no_floors > 17 " + 
            "AND (h_70p/clean_floors > 4.5 OR h_70p/clean_floors < 2);")

        # Clean volume (lod1.2)
        curs.execute(
            "UPDATE " + train_schema + "." + table + "_tmp " + 
            "SET is_clean = NULL, clean_floors = NULL " +
            "WHERE is_clean IS NOT NULL " + 
            "AND (footprint_area*h_70p > 2*lod12_volume OR footprint_area*h_70p*2 < lod12_volume);")
        
        # Clean net internal area
        curs.execute(
            "UPDATE " + train_schema + "." + table + "_tmp " + 
            "SET is_clean = NULL " +
            "WHERE is_clean IS NOT NULL " + 
            "AND (bag_net_internal_area/clean_floors < 10 OR bag_net_internal_area/clean_floors > 2.5*footprint_area);")

        # Replace original table with unlogged temporary table
        db_functions.replace_temp_table(curs, train_schema, table, pkey='bag_id', geom_index='footprint_geom')

        endtime = time()
        duration = endtime - starttime
        print('\n>> Computation time: ', round(duration, 2), 's \n\n' + 10*'-')

    db_functions.close_connection(conn, curs)


if __name__ == '__main__':
    main(sys.argv[1])