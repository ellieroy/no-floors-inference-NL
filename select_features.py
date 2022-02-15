import json, sys
import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import load
from sklearn.feature_selection import SelectKBest, SelectFromModel, f_regression, mutual_info_regression
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
from mpl_toolkits.axes_grid1 import make_axes_locatable
import train_models
from visualise_data import directory_exists


def select_K_best_features(train_set, model, jparams, score, K=10):
    """
    Select K best features based on their correlation/relationship to the target variable. 

    Parameters: \n
    train_set -- train set to select K best features from \n
    model -- name of model corresponding to train set \n
    jparams -- dictionary of parameters from json parameter file \n
    score -- score function to use (either f_regression or mutual_info_regression) \n
    K -- no. features to select \n

    Returns: K best features from train set based on score function used (list of strings)

    """

    if score == 'f_regression': 
        score_function = f_regression

    elif score == 'mutual_info_regression': 
        score_function = mutual_info_regression

    # Drop columns containing NaN for statistical analysis
    df = train_set.dropna()
    X, y = train_models.split_features_labels(df, jparams["id_column"], jparams["labels_column"])
    X_num = X.select_dtypes(exclude=['object'])

    # Use selector to get K best features
    selector = SelectKBest(score_func=score_function, k=K)
    selector.fit(X_num, y)

    # Get names of K best features 
    cols = selector.get_support(indices=True)
    k_best = list(X_num.iloc[:,cols].columns)

    plot_K_best(X[k_best], y, model, score, jparams)

    return k_best


def plot_K_best(X, y, model, score, params): 
    """
    Generate scatter plots of K best features against the target variable.
    Based on: 
    https://scikit-learn.org/stable/auto_examples/feature_selection/plot_f_test_vs_mi.html#sphx-glr-auto-examples-feature-selection-plot-f-test-vs-mi-py

    Parameters: \n
    X -- features \n
    y -- labels \n
    model -- name of model \n
    score -- name of score function \n
    params -- dictionary of parameters from json parameter file \n

    Returns: None

    """

    f_test, _ = f_regression(X, y)
    f_test /= np.max(f_test)

    mi = mutual_info_regression(X, y)
    mi /= np.max(mi)

    # Check directory to save plots exists
    if not directory_exists('plots/feature_selection'):
        os.makedirs('./plots/feature_selection')

    xlabels = []
    for item in list(X.columns): 
        xlabels.append(params['plot_labels'][item])

    plt.figure(figsize=(15, 10))
    for i in range(len(X.columns)):
        plt.subplot(2, math.ceil(len(X.columns)/2), i + 1)
        plt.scatter(X.iloc[:, i], y, s=20, alpha=0.2)
        if xlabels[i] == 'net internal area' or xlabels[i] == 'volume (lod1.2)' or xlabels[i] == 'volume (lod2.2)' or xlabels[i] == 'roof surface area (lod1.2)' or xlabels[i] == 'roof surface area (lod2.2)':
            plt.xscale('log')
        plt.xlabel(xlabels[i], fontsize=14)
        if i == 0 or i == 5:
            plt.ylabel("No. floors", fontsize=14)
        plt.title("F-test={:.2f}, MI={:.2f}".format(f_test[i], mi[i]), fontsize=16)
    plt.tight_layout()
    plt.savefig('plots/feature_selection/k_best_' + model + '_' + score + '.png', dpi=300)
    plt.close()


def select_K_most_important(train_set, pipeline, jparams, K=10): 
    """
    Select a subset of K best features based on importance weights defined by model. 

    Parameters: \n
    train_set -- train set to select K best features from \n
    pipeline -- machine learning pipeline (consisting of both pre-processing and training steps) \n
    jparams -- dictionary of parameters from json parameter file \n
    K -- no. features to select \n

    Returns: K most important features from train set based on algorithm used (list of strings)

    """

    # Split features from labels
    X_train, y_train = train_models.split_features_labels(train_set, jparams["id_column"], jparams["labels_column"])

    # Perform pre-processing steps 
    X_train_pp = pipeline['preprocess'].transform(X_train)

    # Select K most important features from model 
    rgr_new = SelectFromModel(pipeline['rgr'], prefit=True, threshold=-np.inf, max_features=K)

    # Get names of K most important features 
    feature_idx = rgr_new.get_support()
    feature_names = X_train_pp.columns[feature_idx]
    
    # Get original column names
    column_names = []

    for column in X_train_pp[feature_names].columns: 
        
        split_value = '_x0_'

        if split_value in column: 

            original_name = column.split(split_value)[0]

            if original_name not in column_names: 

                column_names.append(original_name)

        else: 
            column_names.append(column)

    return column_names


def select_uncorrelated_features(df, jparams, threshold, model, method='pearson'):
    """
    Select subset of features with low correlation to other features based on hierarchical clustering.  

    Parameters: \n
    df -- DataFrame to select features from \n 
    jparams -- dictionary of parameters from json parameter file \n 
    threshold -- threshold distance for clustering together similar features \n 
    model -- name of model \n 
    method -- method used to compute correlation between features \n 

    Returns: subset of features (list of strings)

    """

    no_floors = jparams["labels_column"]

    # Correlation matrix based on input method of correlation
    corr_matrix = df.corr(method=method)

    # Correlation to no. floors
    corr_floors = corr_matrix[no_floors].sort_values(ascending=False)
    corr_floors = corr_floors.drop(['clean_floors'])
    
    # Remove no. floors from correlation matrix
    corr_features = corr_matrix.drop([no_floors], axis=0).drop([no_floors], axis=1)

    # Perform hierarchical clustering and return leaf indexes + colours of dendrogram
    leaves, leaves_color_list = perform_clustering(corr_features, threshold, model, method, jparams)

    # Automatically select best feature from each cluster (based on correlation to no. floors)
    features = list(corr_features.columns)
    clustered_features = []

    for leaf in leaves: 
        # Find feature corresponding with leaf index
        clustered_features.append(features[leaf])

    # Create dataframe containing features with corresponding cluster number 
    feature_clusters = pd.DataFrame(leaves_color_list, index=clustered_features, columns=['cluster'])

    # Merge correlation dataframe with cluster dataframe based on feature names
    feature_stats = pd.merge(feature_clusters, np.abs(corr_floors), left_index=True, right_index=True)

    # Split dataframe into clustered and non-clustered features (cluster = C0)
    features_uncorr, features_corr = feature_stats.loc[feature_stats['cluster'] == 'C0'], feature_stats.loc[feature_stats['cluster'] != 'C0']

    # Per group of correlated features select feature with highest correlation to no. floors
    idx_corr = features_corr.groupby(['cluster'])['clean_floors'].transform(max) == features_corr['clean_floors']
    output_corr = features_corr[idx_corr].index.tolist()

    # Get name of features that have low correlation to other features 
    output_uncorr = features_uncorr.index.tolist()

    # Merge lists of feature names
    best_features = output_corr + output_uncorr

    return best_features


def perform_clustering(corr_features, threshold, model, method, params): 
    """
    Convert the input correlation matrix to a distance matrix and perform hierarchical clustering using Ward's linkage.
    Visualise clusters using a dendrogram and correlation matrix. 
    The correlation matrix is sorted to place features from the same cluster next to each other.
    Based on: https://scikit-learn.org/stable/auto_examples/inspection/plot_permutation_importance_multicollinear.html

    Parameters: \n
    corr_features -- correlation matrix of features in dataset \n
    threshold -- threshold distance for clustering together similar features \n 
    model -- name of model \n
    method -- method used to compute correlation between features \n 
    params -- dictionary of parameters from json parameter file \n

    Returns: \n
    leaves -- list of indices of features in the correlation matrix ordered by their position in the dendrogram \n
    leaves_color_list -- a list of color names where the k-th element represents the color of the k-th leaf

    """

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8), gridspec_kw={'width_ratios': [1.5, 1]})

    plot_labels = []
    for item in corr_features.columns:
        plot_labels.append(params['plot_labels'][item])

    # Convert the correlation matrix to a distance matrix before performing hierarchical clustering using Ward's linkage.
    distance_matrix = 1 - np.abs(corr_features)
    dist_linkage = hierarchy.linkage(squareform(distance_matrix), method='ward')
    dendro = hierarchy.dendrogram(dist_linkage, labels=list(plot_labels), ax=ax1, orientation='right', color_threshold=threshold)
    dendro_idx = np.arange(0, len(dendro['ivl']))

    corr_array = np.abs(corr_features.to_numpy())
    corr_array = corr_array[dendro['leaves'], :][:, dendro['leaves']]

    # Check directory to save plots exists
    if not directory_exists('plots/dendrograms'):
        os.makedirs('./plots/dendrograms')

    # Plot dendrogram
    im1 = ax2.imshow(corr_array, cmap='viridis')
    ax2.set_xticks(dendro_idx)
    ax2.set_yticks(dendro_idx)
    ax2.set_xticklabels(dendro['ivl'], rotation=45, ha='right')
    ax2.set_yticklabels(dendro['ivl'])
    divider = make_axes_locatable(ax2)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im1, cax=cax, orientation='vertical')
    fig.tight_layout()
    plt.savefig("./plots/dendrograms/dendrogram_" + model + "_" + method + ".pdf", bbox_inches="tight", dpi=300)
    plt.close()

    return dendro['leaves'], dendro['leaves_color_list']


def main(params):

    # Load parameters 
    jparams = json.load(open(params))
    
    # Loop through each model 
    for model in jparams["feature_selection_models"]:

        if os.path.exists('data/train_sets/train_set_' + model + '.csv') and os.path.exists('data/test_sets/test_set_' + model + '.csv'):
            
            # Read train and test sets from csv into dataframe
            train_set = pd.read_csv('data/train_sets/train_set_' + model + '.csv', index_col=0, dtype={'bag_id':'str'})
            test_set = pd.read_csv('data/test_sets/test_set_' + model + '.csv', index_col=0, dtype={'bag_id':'str'})

            print('\n>> Selecting K best features')

            # Select K best features based on univariate linear regression tests
            k_best_f = select_K_best_features(train_set, model, jparams, 'f_regression', K=10)

            # Select K best features based on mutual information (non-linear)
            k_best_mi = select_K_best_features(train_set, model, jparams, 'mutual_info_regression', K=10)

            # Define columns subset (including labels and bag_id)
            columns_kBest = [jparams["id_column"], jparams["labels_column"]] + k_best_mi
            
            # Select subset of features from train and test sets
            train_set_kBest = train_set[columns_kBest]
            test_set_kBest = test_set[columns_kBest]

            # Define additional part of model/file name
            counter = 1
            model_addition = '.' + str(counter) 
            counter += 1

            # Save new training and test sets to file 
            train_set_kBest.to_csv('data/train_sets/train_set_' + model + model_addition + '.csv') 
            test_set_kBest.to_csv('data/test_sets/test_set_' + model + model_addition + '.csv')

            # Split features from labels
            X_train_kBest, y_train_kBest = train_models.split_features_labels(train_set_kBest, jparams["id_column"], jparams["labels_column"])

            # Train model based on K best features
            for algorithm in jparams["ml_algorithms"]: 
                train_models.training_steps(X_train_kBest, y_train_kBest, model + model_addition, algorithm)

            # Select K most important features based on model (coefficients or impurity-based importance)
            print('\n>> Selecting K most important features')

            if directory_exists('models/'):

                # Load the corresponding model pipeline
                for filename in os.listdir('./models'):

                    file_info = filename[:-7].split('_')
                    
                    # Only perform selection for base model 
                    if filename.endswith('.joblib') and 'model_' + file_info[2] == model and file_info[4] != 'tuned':
                        
                        algorithm_name = file_info[3]

                        # Load pipeline
                        pipeline = load('./models/' + filename)

                        # Select K most important features 
                        k_important = select_K_most_important(train_set, pipeline, jparams)

                        # Define columns subset (including labels and bag_id)
                        columns_kImportant = [jparams["id_column"], jparams["labels_column"]] + k_important 
                        
                        # Select subset of features from train and test sets
                        train_set_kImportant = train_set[columns_kImportant]
                        test_set_kImportant = test_set[columns_kImportant]

                        # Define additional part of model/file name
                        model_addition = '.' + str(counter) 
                        counter += 1

                        # Save new training and test sets to file 
                        train_set_kImportant.to_csv('data/train_sets/train_set_' + model + model_addition + '.csv') 
                        test_set_kImportant.to_csv('data/test_sets/test_set_' + model + model_addition + '.csv')

                        # Split features from labels
                        X_train_kImportant, y_train_kImportant = train_models.split_features_labels(train_set_kImportant, jparams["id_column"], jparams["labels_column"])
                        
                        # Train model based on K most important features
                        train_models.training_steps(X_train_kImportant, y_train_kImportant, model + model_addition, algorithm_name)

            # Select best subset of features to maximise correlation to no. floors / minimise correlation between features
            print('\n>> Selecting features to maximise correlation to no. floors / minimise correlation between features')
            best_uncorr = select_uncorrelated_features(train_set, jparams, 0.4, model, method='pearson') 

            # Define columns subset (all features with low multicollinearity) 
            columns_uncorr = [jparams["id_column"], jparams["labels_column"]] + best_uncorr
            
            # Select subset of features with low multicollinearity from train set
            train_set_uncorr = train_set[columns_uncorr]

            # Select 10 best features from train set with low multicollinearity
            kBest_mi_uncorr = select_K_best_features(train_set_uncorr, model+'_uncorr', jparams, 'mutual_info_regression', K=10)

            # Define columns subset (10 best features with low multicollinearity + categorical features except bag_function) 
            text_columns = jparams["text_columns"]
            text_columns.remove('bag_function')
            columns_kBest_uncorr = [jparams["id_column"], jparams["labels_column"]] + text_columns + kBest_mi_uncorr

            # Select subset of features from train and test sets
            train_set_kBest_uncorr = train_set[columns_kBest_uncorr]
            test_set_kBest_uncorr = test_set[columns_kBest_uncorr]

            # Define additional part of model/file name
            model_addition = '.' + str(counter) 
            counter += 1

            # Save new training and test sets to file 
            train_set_kBest_uncorr.to_csv('data/train_sets/train_set_' + model + model_addition + '.csv') 
            test_set_kBest_uncorr.to_csv('data/test_sets/test_set_' + model + model_addition + '.csv')

            # Split features from labels
            X_train_kBest_uncorr, y_train_kBest_uncorr = train_models.split_features_labels(train_set_kBest_uncorr, jparams["id_column"], jparams["labels_column"])

            # Train model based on K best features with low multicollinearity
            for algorithm in jparams["ml_algorithms"]: 
                train_models.training_steps(X_train_kBest_uncorr, y_train_kBest_uncorr, model + model_addition, algorithm)   


if __name__ == '__main__':
    main(sys.argv[1])