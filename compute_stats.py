"""
Computes statistical measures on training data. 

"""

import json
import os, sys
import pandas as pd
import warnings
from statsmodels.stats.outliers_influence import variance_inflation_factor
from visualise_data import directory_exists
import ml_plots
import train_models


def compute_correlation(train_set, method, jparams, model): 
    """
    Compute pairwise correlation of columns in training set. 

    Parameters \n
    train_set -- data from the train set \n
    method -- method used to compute correlation \n
    jparams -- dictionary of parameters from json parameter file \n
    model -- name of model \n

    Returns: correlation matrix

    """

    # Check directory to save data exists
    if not directory_exists('./data/feature_selection/corr'):
        os.makedirs('./data/feature_selection/corr')

    no_floors = jparams["labels_column"]

    corr_matrix = train_set.corr(method=method)
    corr_floors = corr_matrix[no_floors].sort_values(ascending=False)
    corr_floors = corr_floors.drop(['clean_floors'])
    corr_floors.to_csv('data/feature_selection/corr/corr_' + model + '_' + method + '.csv')

    return corr_matrix


def compute_vif(X_train, model):
    """
    Compute variance inflation factor for each feature in the training set. 

    Parameters: \n
    X_train -- training features \n
    model -- name of model \n

    Returns: None

    """

    # Check directory to save data exists
    if not directory_exists('./data/feature_selection/vif'):
        os.makedirs('./data/feature_selection/vif')

    # Prepare data (remove categorical features and NULL values)
    X_prep = X_train.select_dtypes(exclude=['object'])
    X_prep = X_prep.dropna()

    # Compute VIF scores 
    X_prep['intercept'] = 1 # add constant to df
    vif = pd.DataFrame()
    vif['variables'] = X_prep.columns
    vif['VIF'] = [variance_inflation_factor(X_prep.values, i) for i in range(X_prep.shape[1])]
    vif_sorted = vif.sort_values(by='VIF', ascending=False)
    vif_sorted.to_csv('data/feature_selection/vif/vif_' + model + '.csv')


def main(params):

    # Load parameters 
    jparams = json.load(open(params))

    if directory_exists('./data/train_sets') and directory_exists('./data/test_sets'):

        # Define path to train sets and order files by date/time created
        search_dir = './data/train_sets'
        files = os.listdir(search_dir)
        files.sort(key=lambda fn: os.path.getmtime(os.path.join(search_dir, fn)))

        files = [file for file in files if '.csv' in file]

        for filename, i in zip(files, range(len(files))):

            # Get information about model from file name
            file_info = filename[:-4].split('_')
            model = 'model_' + file_info[3]

            # Check path to test set also exists
            if os.path.exists('data/test_sets/test_set_' + model + '.csv'):

                print('\n>> Computing stats for model: {0}'.format(model))

                # Read train and test sets from csv into dataframe
                train_set = pd.read_csv('data/train_sets/train_set_' + model + '.csv', index_col=0, dtype={'bag_id':'str'})
                test_set = pd.read_csv('data/test_sets/test_set_' + model + '.csv', index_col=0, dtype={'bag_id':'str'})

                # Split labels from features
                X_train, y_train = train_models.split_features_labels(train_set, jparams["id_column"], jparams["labels_column"])
                X_test, y_test = train_models.split_features_labels(test_set, jparams["id_column"], jparams["labels_column"])

                # Create histograms of no. floors in training and test sets 
                ml_plots.plot_hist(y_train, model, 'train')
                ml_plots.plot_hist(y_test, model, 'test')

                # Plot correlation matrices           
                corr_matrix = compute_correlation(train_set, 'pearson', jparams, model)
                ml_plots.plot_corr_matrix(corr_matrix, model, jparams, method='pearson')

                corr_matrix = compute_correlation(train_set, 'spearman', jparams, model)
                ml_plots.plot_corr_matrix(corr_matrix, model, jparams, method='spearman')

                # Get VIF scores
                warnings.filterwarnings("ignore")
                compute_vif(X_train, model)


if __name__ == '__main__':
    main(sys.argv[1])