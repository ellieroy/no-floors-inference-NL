import json, sys
from time import time
import os
import numpy as np
from zlib import crc32
import pandas as pd
from sklearn_pandas import DataFrameMapper
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import LinearSVR
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.model_selection import StratifiedShuffleSplit
from joblib import load, dump
import db_functions
import tune_models
from visualise_data import directory_exists


def get_feature_names(features_dict, model):
    """
    Store the names of the features used by each model as a string. 

    Parameters: \n
    features_dict -- dictionary of features used by each model (obtained from json parameter file) \n
    model -- name of model (dictionary key) \n

    Returns: string of feature names separated by commas

    """

    features = features_dict[model]

    feature_names = ''

    for i, item in enumerate(features):
        
        if i == 0:
            feature_names += item

        else: 
            feature_names += ', ' + item

    return feature_names


def collect_data(schema, tables=None, cols='*', where="WHERE is_clean IS NOT NULL"):
    """
    Collect the required data from the database. 

    Parameters: \n
    schema -- database schema where data is stored \n
    tables -- tables to obtain data from, if None then all unique tables in the database schema are used \n
    cols -- columns to select data from (string), if '*' then all columns in the table are used \n
    where -- where statement to filter data selection, set to only select clean data by default \n

    Returns: dataframe containing data collected from all tables queried

    """

    print('\n>> Collecting data from schema: {0}'.format(schema))

    # Create connection to database
    conn = db_functions.setup_connection()

    # Create a cursor (allows PostgreSQL commands to be executed)
    curs = conn.cursor()

    # If tables not specified then get all unique tables in input schema
    if not tables:
        tables = db_functions.unique_tables(curs, schema)

    all_data = np.array([])

    # Extract all data and store into a pandas DataFrame.
    for i, table in enumerate(tables):

        # Initialise pandas dataframe on first iteration
        if i == 0: 
            all_data = db_functions.read_data(conn, schema, table, columns=cols, where=where) 

        # Append data to dataframe on all other iterations
        else: 
            df = db_functions.read_data(conn, schema, table, columns=cols, where=where) 
            all_data = all_data.append(df, ignore_index=True)

    db_functions.close_connection(conn, curs)

    return all_data


def preprocess_mapper(X_train): 
    """
    Obtain a DataFrameMapper which can perform all pre-processing steps on the training data. 
    The pre-processing steps used are defined separately for numerical and categorical features. 

    Parameters: \n
    X_train -- dataframe of features from train set \n

    Returns: DataFrameMapper which can be applied to data to perform all pre-processing steps
    
    """

    num_features = list(X_train.select_dtypes(exclude=['object']))
    cat_features = list(X_train.select_dtypes(include=['object']))

    cat = [([c], [SimpleImputer(strategy='most_frequent'), SimpleImputer(strategy='most_frequent', missing_values=None), OneHotEncoder(drop='if_binary')]) for c in cat_features]
    num = [([n], [IterativeImputer(random_state=0), StandardScaler()]) for n in num_features]
    mapper = DataFrameMapper(num + cat, df_out=True)
    
    return mapper


def get_estimator(algorithm_name):
    """
    Get the estimator corresponding to the input algorithm name. 

    Parameters: \n
    algorithm -- name of algorithm to obtain estimator for \n

    Returns: an estimator from the scikit-learn libary (either random forest regressor, linear svr or gradient boosting regressor)

    """

    if algorithm_name == 'rfr': 

        rgr = RandomForestRegressor(random_state=0)

    elif algorithm_name == 'svr': 

        rgr =  LinearSVR(random_state=0)

    elif algorithm_name == 'gbr':

        rgr = GradientBoostingRegressor(random_state=0)
    
    else: 
        print('Input ML algorithm not recognised ! Choose from: <rfr>, <svr> or <gbr>')
        sys.exit()

    return rgr


def test_set_check(identifier, test_ratio):
    """
    Check whether a building is in the test set or not. 

    Parameters: \n
    identifier: id of building \n
    test_ratio: relative size of test set \n

    Returns: True/False

    """

    return crc32(np.int64(identifier)) & 0xffffffff < test_ratio * 2**32


def split_train_test_by_id(data, test_ratio, id_column):
    """
    Create train and test sets based on building id. 

    Parameters: \n
    data -- training data used to create train and test sets \n
    test_ratio -- relative size of test set \n
    id_column -- name of column used to store building ids \n

    Returns: train set and test set (DataFrames)

    """

    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio))
    
    return data.loc[~in_test_set], data.loc[in_test_set]


def train_test_split(training_data, test_ratio, id_column, features):
    """
    Generate a train-test split based on either building height (70th percentile) or id. 

    Parameters: \n
    training_data -- training data used to create train and test sets \n
    test_ratio -- relative size of test set \n
    id_column -- name of column used to store building ids \n
    features -- training data features \n

    Returns: train and test sets (DataFrames)

    """

    if 'h_70p' in features:

        # Create stratified train/test sets based on building height
        training_data.dropna(subset=['h_70p'], inplace=True)
        training_data.reset_index(drop=True, inplace=True)
        training_data['h_cat'] = pd.cut(training_data['h_70p'], bins=[0,5,6,8,10,np.inf], labels=[1,2,3,4,5])

        split = StratifiedShuffleSplit(n_splits=1, test_size=test_ratio, random_state=42)
        for train_index, test_index in split.split(training_data, training_data['h_cat']):
            train_set = training_data.loc[train_index]
            test_set = training_data.loc[test_index]

        for set_ in (train_set, test_set): 
            set_.drop('h_cat', axis=1, inplace=True)
    
    else: 

        # Create training and test sets based on building ID
        train_set, test_set = split_train_test_by_id(training_data, test_ratio, id_column)

    return train_set, test_set


def split_features_labels(data, id_column, labels_column): 
    """
    Split training data into features and labels + remove building ids. 

    Parameters: \n
    data -- training data \n
    id_column -- name of column used to store building ids \n
    labels_column -- name of column used to store training data labels \n

    Returns: DataFrames of features and corresponding labels 

    """
    
    building_id = id_column
    no_floors = labels_column

    X, y = data.drop([building_id, no_floors], axis=1), data[no_floors]

    return X, y


def training_steps(X_train, y_train, model, algorithm, tune_params=False, estimator=None):
    """
    Fit a machine learning pipeline to the input data (consists of both training and pre-processing steps).  
    The trained model is stored to file for later use. 

    Parameters: \n
    X_train -- training features \n
    y_train -- training labels \n
    model -- name of output model \n
    algorithm -- name of algorithm to use \n
    tune_params -- whether to tune the algorithm's hyperparameters (True/False) \n
    estimator -- (optional) an estimator to train, if None then the algorithm name is used to select an estimator \n

    Returns: None

    """

    print('\n>> Model name: {0} \n\n   Machine learning algorithm: {1}'.format(model, algorithm))

    # Create pre-processing steps for training data features
    mapper = preprocess_mapper(X_train)

    if estimator is None:

        if tune_params:
            # Run hyperparameter search to get best estimator 
            rgr = tune_models.get_best_estimator(X_train, y_train, algorithm, mapper)
            tuning_description = 'tuned'
        
        else: 
            # Get basic estimator (no hyperparameter tuning)
            rgr = get_estimator(algorithm)
            tuning_description = 'untuned'
    
    else: 
        # Use the input estimator with pre-tuned hyperparameters
        rgr = estimator
        tuning_description = 'tuned'
    
    # Define complete machine learning pipeline
    pipeline = Pipeline([
        ('preprocess', mapper),
        ('rgr', rgr)
    ])

    # Fit pipeline to training set
    print('\n   >> Training model')
    starttime = time()
    pipeline.fit(X_train, y_train)
    endtime = time()
    duration = endtime - starttime
    print('\n   >> Training time: ', round(duration, 2), 's')

    # Check model directory exists
    if not directory_exists('models/'):
        os.mkdir('./models')
    
    # Save pipeline to file for later use
    dump(pipeline, 'models/pipeline_' + model + '_' + algorithm + '_' + tuning_description + '.joblib')


def main(params):

    # Load parameters 
    jparams = json.load(open(params))

    # Parameters
    models = jparams["models_to_train"]
    train_schema = jparams["training_schema"]
    train_tables = jparams["training_tables"]
    building_id = jparams["id_column"]
    no_floors = jparams["labels_column"]
    best_estimator_path = './models/' + jparams["best_estimator"]

    for model in models: 

        # Get name of features used by input model 
        features = get_feature_names(jparams["features"], model)

        # Collect all training data into one dataframe 
        columns = building_id + ', ' + no_floors + ', ' + features
        training_data = collect_data(train_schema, train_tables, cols=columns) 

        # Create train / test split
        train_set, test_set = train_test_split(training_data, 0.2, building_id, features)

        # Check data directories exist
        if not directory_exists('data/train_sets'):
            os.makedirs('./data/train_sets')

        if not directory_exists('data/test_sets'):
            os.makedirs('./data/test_sets')

        # Save training and test sets to file 
        train_set.to_csv('data/train_sets/train_set_' + model + '.csv')
        test_set.to_csv('data/test_sets/test_set_' + model + '.csv')
        
        # Split labels from features
        X_train, y_train = split_features_labels(train_set, building_id, no_floors)

        # Determine whether model should use pre-tuned hyperparameters of best estimator
        if model in jparams["use_tuned_params"] and os.path.exists(best_estimator_path):
            
            # Get name of algorithm used by best estimator
            algorithm = best_estimator_path.split('_')[3]

            # Define a new regressor with the same hyperparameters as the best estimator 
            best_estimator = load(best_estimator_path)['rgr']
            rgr = clone(best_estimator)

            # Perform model training steps 
            training_steps(X_train, y_train, model, algorithm, estimator=rgr)
        
        # Train model using basic estimator for each input algorithm 
        elif model not in jparams["use_tuned_params"]: 

            # Loop through each ML algorithm 
            for algorithm in jparams["ml_algorithms"]: 

                # Perform model training steps 
                training_steps(X_train, y_train, model, algorithm)
        
        else: 

            print('\n>> Model {0} could not be trained! Check input parameters.'.format(model))
        
       
if __name__ == '__main__':

    main(sys.argv[1])

