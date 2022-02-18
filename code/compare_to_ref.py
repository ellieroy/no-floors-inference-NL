import json, sys, os
import numpy as np
import csv
import ml_plots
import pandas as pd
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, max_error
import train_models
import test_models
import db_functions
from visualise_data import directory_exists


def compare_to_ref_model(model_name, algorithm, tuning_description, jparams):
    """
    Compute error metrics for reference model, best model and model that always predicts mean no. floors.
    Results are stored to csv for comparison. 

    Parameters: \n
    model_name -- name of model \n
    algorithm -- name of algorithm used by model \n
    tuning_description -- describes whether model uses default parameters or tuned parameters \n
    jparams -- dictionary of parameters from json parameter file \n

    Returns: None

    """

    # Define input parameters
    train_schema = jparams["training_schema"]
    train_tables = jparams["training_tables"]
    building_id = jparams["id_column"]
    no_floors = jparams["labels_column"]

    # Columns to select from database
    columns = building_id + ', ref_model, ref_model_unrounded'

    # Get reference model results 
    ref_model_results = train_models.collect_data(train_schema, train_tables, cols=columns)
    ref_model_results.dropna(inplace=True)
    if directory_exists('../thesis_report/plots'):
        ref_model_results.to_csv('../thesis_report/plots/ref_model_results.csv')

    train_set_path = 'data/train_sets/train_set_' + model_name + '.csv'
    test_set_path = 'data/test_sets/test_set_' + model_name + '.csv'
    pred_train_path = 'data/predictions/rounded/trainset_' + model_name + '_' + algorithm + '_' + tuning_description + '.csv'
    pred_test_path = 'data/predictions/rounded/testset_' + model_name + '_' + algorithm + '_' + tuning_description + '.csv'

    if os.path.exists(train_set_path) and os.path.exists(test_set_path) and os.path.exists(pred_train_path) and os.path.exists(pred_test_path):
        
        # Read train / test sets 
        train_set = pd.read_csv(train_set_path, index_col=0, dtype={'bag_id':'str'})
        test_set = pd.read_csv(test_set_path, index_col=0, dtype={'bag_id':'str'})
        
        # Read predictions of train / test sets 
        train_pred = pd.read_csv(pred_train_path)
        test_pred = pd.read_csv(pred_test_path)

        # Create one dataframe with building id + plus prediction
        train_set = train_set.assign(pred_value=train_pred.values)
        test_set = test_set.assign(pred_value=test_pred.values)

        # Remove unnecessary data 
        train_set = train_set[[building_id, no_floors, 'pred_value']]
        test_set = test_set[[building_id, no_floors, 'pred_value']]
    
        # Combine ref model and predictions into one dataframe
        train_results = pd.merge(train_set, ref_model_results, on=building_id, how='left')
        test_results = pd.merge(test_set, ref_model_results, on=building_id, how='left')

        # Mean number of floors in the test set
        test_results['mean_no_floors'] = test_results[no_floors].mean()

        # Value to separate data into classes
        x_floors = jparams["class_threshold"]

        # Compute accuracy 
        accuracy_ref = np.around(accuracy_score(test_results[no_floors], test_results.ref_model)*100, 1)
        accuracy_test = np.around(accuracy_score(test_results[no_floors], np.rint(test_results.pred_value))*100, 1)
        accuracy_mean = np.around(accuracy_score(test_results[no_floors], np.rint(test_results.mean_no_floors))*100, 1)

        # Compute accuracy per class
        accuracy_above_ref, accuracy_below_ref = test_models.custom_accuracy_per_class(test_results[no_floors], test_results.ref_model, x_floors)
        accuracy_above_ref, accuracy_below_ref = np.around(accuracy_above_ref*100, 1), np.around(accuracy_below_ref*100, 1)
        accuracy_above_test, accuracy_below_test = test_models.custom_accuracy_per_class(test_results[no_floors], np.rint(test_results.pred_value), x_floors)
        accuracy_above_test, accuracy_below_test = np.around(accuracy_above_test*100, 1), np.around(accuracy_below_test*100, 1)
        accuracy_above_mean, accuracy_below_mean = test_models.custom_accuracy_per_class(test_results[no_floors], np.rint(test_results.mean_no_floors), x_floors)
        accuracy_above_mean, accuracy_below_mean = np.around(accuracy_above_mean*100, 1), np.around(accuracy_below_mean*100, 1)

        # Compute MAE
        mae_ref = np.around(mean_absolute_error(test_results[no_floors], test_results.ref_model), 2)
        mae_test = np.around(mean_absolute_error(test_results[no_floors], np.rint(test_results.pred_value)), 2)
        mae_mean = np.around(mean_absolute_error(test_results[no_floors], np.rint(test_results.mean_no_floors)), 2)

        ml_plots.plot_abs_error(test_results[no_floors], test_results.ref_model, jparams, model_name, algorithm, '_', 'ref')

        # Compute MAE per class
        mae_above_ref, mae_below_ref = test_models.custom_mae_per_class(test_results[no_floors], test_results.ref_model, x_floors)
        mae_above_ref, mae_below_ref = np.around(mae_above_ref, 2), np.around(mae_below_ref, 2)
        mae_above_test, mae_below_test = test_models.custom_mae_per_class(test_results[no_floors], np.rint(test_results.pred_value), x_floors)
        mae_above_test, mae_below_test = np.around(mae_above_test, 2), np.around(mae_below_test, 2)
        mae_above_mean, mae_below_mean = test_models.custom_mae_per_class(test_results[no_floors], np.rint(test_results.mean_no_floors), x_floors)
        mae_above_mean, mae_below_mean = np.around(mae_above_mean, 2), np.around(mae_below_mean, 2)

        # Compute RMSE 
        rmse_ref = np.around(np.sqrt(mean_squared_error(test_results[no_floors], test_results.ref_model)), 2)
        rmse_test = np.around(np.sqrt(mean_squared_error(test_results[no_floors], np.rint(test_results.pred_value))), 2)
        rmse_ref = np.around(np.sqrt(mean_squared_error(test_results[no_floors], np.rint(test_results.mean_no_floors))), 2)
        
        # Compute RMSE per class
        rmse_above_ref, rmse_below_ref = test_models.custom_rmse_per_class(test_results[no_floors], test_results.ref_model, x_floors)
        rmse_above_ref, rmse_below_ref = np.around(rmse_above_ref, 2), np.around(rmse_below_ref, 2)
        rmse_above_test, rmse_below_test = test_models.custom_rmse_per_class(test_results[no_floors], np.rint(test_results.pred_value), x_floors)
        rmse_above_test, rmse_below_test = np.around(rmse_above_test, 2), np.around(rmse_below_test, 2)
        rmse_above_mean, rmse_below_mean = test_models.custom_rmse_per_class(test_results[no_floors], np.rint(test_results.mean_no_floors), x_floors)
        rmse_above_mean, rmse_below_mean = np.around(rmse_above_mean, 2), np.around(rmse_below_mean, 2)

        # Compute max error 
        max_error_ref = max_error(test_results[no_floors], test_results.ref_model)
        max_error_test = max_error(test_results[no_floors], np.rint(test_results.pred_value))
        max_error_mean = max_error(test_results[no_floors], np.rint(test_results.mean_no_floors))
        
        # Compute max error per class 
        max_above_ref, max_below_ref = test_models.custom_max_err_per_class(test_results[no_floors], test_results.ref_model, x_floors)
        max_above_test, max_below_test = test_models.custom_max_err_per_class(test_results[no_floors], np.rint(test_results.pred_value), x_floors)
        max_above_mean, max_below_mean = test_models.custom_max_err_per_class(test_results[no_floors], np.rint(test_results.mean_no_floors), x_floors)

        # Store results to csv
        evaluation_results_ref = ['ref_model', mae_above_ref, mae_below_ref, rmse_above_ref, rmse_below_ref, accuracy_above_ref, accuracy_below_ref, max_above_ref, max_below_ref] 
        evaluation_results_test = ['best_model', mae_above_test, mae_below_test, rmse_above_test, rmse_below_test, accuracy_above_test, accuracy_below_test, max_above_test, max_below_test]
        evaluation_results_mean = ['mean', mae_above_mean, mae_below_mean, rmse_above_mean, rmse_below_mean, accuracy_above_mean, accuracy_below_mean, max_above_mean, max_below_mean]

        fname = 'model_comparison_to_ref.csv'

        with open('models/' + fname, 'w', encoding='UTF8') as f:
            fieldnames = ['model', 'mae_above', 'mae_below', 'rmse_above', 'rmse_below', 'accuracy_above', 'accuracy_below', 'max_above', 'max_below']
            writer = csv.writer(f)
            writer.writerow(fieldnames)
            writer.writerow(evaluation_results_ref)
            writer.writerow(evaluation_results_test)
            writer.writerow(evaluation_results_mean)


def export_results(model_name, algorithm, tuning_description, jparams):
    """
    Store the model predictions to a geojson file and to the database. 

    Parameters: \n
    model_name -- name of model \n
    algorithm -- name of algorithm used by model \n
    tuning_description -- describes whether model uses default parameters or tuned parameters \n
    jparams -- dictionary of parameters from json parameter file \n

    Returns: None

    """
    
    # Paths to files
    train_set_path = 'data/train_sets/train_set_' + model_name + '.csv'
    test_set_path = 'data/test_sets/test_set_' + model_name + '.csv'
    pred_train_path_rounded = 'data/predictions/rounded/trainset_' + model_name + '_' + algorithm + '_' + tuning_description + '.csv'
    pred_test_path_rounded = 'data/predictions/rounded/testset_' + model_name + '_' + algorithm + '_' + tuning_description + '.csv'
    pred_train_path = 'data/predictions/trainset_' + model_name + '_' + algorithm + '_' + tuning_description + '.csv'
    pred_test_path = 'data/predictions/testset_' + model_name + '_' + algorithm + '_' + tuning_description + '.csv'

    if os.path.exists(train_set_path) and os.path.exists(test_set_path) and os.path.exists(pred_train_path) and os.path.exists(pred_test_path) and os.path.exists(pred_train_path_rounded) and os.path.exists(pred_test_path_rounded):

        # Read train/test sets  
        train_set = pd.read_csv(train_set_path, index_col=0, dtype={'bag_id':'str'})
        test_set = pd.read_csv(test_set_path, index_col=0, dtype={'bag_id':'str'})

        # Read predictions
        pred_train_set = pd.read_csv(pred_train_path)
        pred_train_set_rounded = pd.read_csv(pred_train_path_rounded)
        pred_test_set = pd.read_csv(pred_test_path)
        pred_test_set_rounded = pd.read_csv(pred_test_path_rounded)
        
        # Merge train/test sets + predictions
        test_set = test_set.assign(pred_value=pred_test_set.values, pred_value_rounded=pred_test_set_rounded.values)
        train_set = train_set.assign(pred_value=pred_train_set.values, pred_value_rounded=pred_train_set_rounded.values)

        # Create connection to database
        conn = db_functions.setup_connection()

        # Create a cursor (allows PostgreSQL commands to be executed)
        curs = conn.cursor()

        all_data = np.array([])

        schema = jparams['training_schema']
        cols = 'bag_id, bag3d_roof_type, footprint_geom, ref_model, ref_model_unrounded'
        where = 'WHERE is_clean IS NOT NULL'

        # Extract all data and store into a pandas DataFrame.
        for i, table in enumerate(jparams["training_tables"]):

            # Initialise pandas dataframe on first iteration
            if i == 0: 
                all_data = db_functions.read_spatial_data(conn, schema, table, columns=cols, where=where, geom_col='footprint_geom') 

            # Append data to dataframe on all other iterations
            else: 
                df = db_functions.read_spatial_data(conn, schema, table, columns=cols, where=where, geom_col='footprint_geom') 
                all_data = all_data.append(df, ignore_index=True)

        # Close database connection
        db_functions.close_connection(conn, curs)

        # Merge dataframes 
        all_data_test = all_data.merge(test_set, on='bag_id')
        all_data_test['pred-label'] = all_data_test.pred_value_rounded - all_data_test.clean_floors

        all_data_train = all_data.merge(train_set, on='bag_id')
        all_data_train['pred-label'] = all_data_train.pred_value_rounded - all_data_train.clean_floors

        # Check directory to save data exists
        if not directory_exists('./data/geojson'): 
            os.makedirs('./data/geojson')

        # Save to geojson
        all_data_test.to_file('./data/geojson/predictions_test.geojson', driver='GeoJSON')
        all_data_train.to_file('./data/geojson/predictions_train.geojson', driver='GeoJSON')

        # Export to database
        conn = db_functions.setup_connection()
        conn.autocommit = True
        curs = conn.cursor()
        rows_test = zip(all_data_test.bag_id, all_data_test.pred_value, all_data_test.pred_value_rounded, all_data_test.clean_floors)
        rows_train = zip(all_data_train.bag_id, all_data_train.pred_value, all_data_train.pred_value_rounded, all_data_train.clean_floors)
        bulk_insert(curs, 'predictions', rows_test, 'best_pred_test')
        bulk_insert(curs, 'predictions', rows_train, 'best_pred_train')
        db_functions.close_connection(conn, curs)


def bulk_insert(cursor, schema, rows, table): 
    """
    Insert the (rounded) predictions into the database. 

    Parameters: \n
    cursor -- cursor for database connection \n
    schema -- schema to store the data \n
    rows -- data to insert as rows to the database (bag ids, pred. values, rounded pred. values and labels stored in a zip object) \n
    table -- table to store the data \n

    Returns: None
    
    """

    print('\n>> Bulk insert table: {0}.{1}'.format(schema, table))

    cursor.execute("CREATE SCHEMA IF NOT EXISTS " + schema + ";")
    cursor.execute("DROP TABLE IF EXISTS " + schema + "." + table + ";")
    cursor.execute("CREATE TABLE " + schema + "." + table + " (bag_id VARCHAR, pred_value REAL, pred_value_rounded REAL, clean_floors REAL);")

    cursor.executemany(
        "INSERT INTO " + schema + "." + table + " " + 
        "(bag_id, pred_value, pred_value_rounded, clean_floors) " + 
        "VALUES(%s, %s, %s, %s);", rows)


def main(params):
    
    # Load parameters 
    jparams = json.load(open(params))

    # Get info about best estimator 
    best_estimator_info = jparams["best_estimator"][:-7].split('_')
    model = 'model_' + best_estimator_info[2]
    algorithm = best_estimator_info[3]
    tuning_descrip = best_estimator_info[4]

    # Compare model results to reference model 
    compare_to_ref_model(model, algorithm, tuning_descrip, jparams)

    # Export data to geojson and database to make maps of predictions
    export_results(model, algorithm, tuning_descrip, jparams)


if __name__ == '__main__':
    main(sys.argv[1])