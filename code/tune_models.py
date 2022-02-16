import json, sys, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import LinearSVR
from sklearn.metrics import r2_score
from sklearn.model_selection import RandomizedSearchCV, validation_curve, ShuffleSplit
import train_models
from visualise_data import directory_exists


def get_best_estimator(X_train, y_train, algorithm, preprocess_mapper):
    """
    Apply cross validation and a randomised grid search to find optimal hyperparameters based on the training data.

    Parameters: \n
    X_train -- training features \n
    y_train -- training labels \n
    algorithm -- name of algorithm to perform hyperparameter tuning \n
    preprocess_mapper -- DataFrameMapper consisting of all pre-processing steps to perform on data before training \n 

    Returns: best estimator obtained during search 

    """

    print('\n   >> Determining optimal hyperparameters')

    # Find optimal random forest hyperparameters
    if algorithm == 'rfr':

        # No. trees in random forest
        n_estimators = np.linspace(start=50, stop=350, num=13, dtype=int)

        # No. features to consider at every split of a node 
        max_features = ['auto', 'sqrt', 'log2']

        # Max. depth of the trees
        max_depth = [int(x) for x in np.linspace(10, 42, num=9, dtype=int)]
        max_depth.append(None)

        # Min. no. samples required to split a node
        min_samples_split = np.linspace(2, 50, num=13, dtype=int)

        # Min. no. samples required at each leaf node
        min_samples_leaf = np.linspace(1, 45, num=10, dtype=int)

        # Method for selecting the samples for each individual tre.
        bootstrap = [True, False]

        # Create a random grid with all parameters
        random_grid = {'n_estimators': n_estimators,
                    'max_features': max_features,
                    'max_depth': max_depth,
                    'min_samples_split': min_samples_split,
                    'min_samples_leaf': min_samples_leaf,
                    'bootstrap': bootstrap}

        # Create base model to tune
        regressor = RandomForestRegressor(random_state=0)

        # Define scoring function 
        scoring = 'neg_root_mean_squared_error'
     
    elif algorithm == 'svr': 

        epsilon = [0.0, 0.01, 0.05, 0.1, 0.5, 1]
        tol = [1e-3, 1e-4, 1e-5]
        C = np.linspace(0.1, 1.1, 6)
        loss = ['epsilon_insensitive', 'squared_epsilon_insensitive']
        dual = [True, False]
        max_iter = np.linspace(1000, 5000, 9, dtype=int) 

        # Create a random grid with all parameters.
        random_grid = {'epsilon': epsilon,
                        'tol': tol,
                        'C': C,
                        'loss': loss,
                        'dual': dual,
                        'max_iter': max_iter}

        # Create base model to tune
        regressor = LinearSVR(random_state=0)

        # Define scoring function 
        scoring = None # default

    elif algorithm == 'gbr': 

        n_estimators = np.linspace(start=150, stop=850, num=15, dtype=int)
        max_features = ['auto', 'sqrt', 'log2']
        max_depth = [int(x) for x in np.linspace(2, 24, num=12, dtype=int)]
        min_samples_split = np.linspace(2, 50, num=13, dtype=int)
        min_samples_leaf = np.linspace(1, 45, num=10, dtype=int)
        learning_rate = np.linspace(0.01, 0.1, num=10)

        # Create a random grid with all parameters
        random_grid = {'n_estimators': n_estimators,
                    'max_features': max_features,
                    'max_depth': max_depth,
                    'min_samples_split': min_samples_split,
                    'min_samples_leaf': min_samples_leaf, 
                    'learning_rate': learning_rate}

        # Create base model to tune
        regressor = GradientBoostingRegressor(random_state=0)

        # Define scoring function 
        scoring = 'neg_root_mean_squared_error'

    else: 
        print('Input ML algorithm not recognised ! Choose from: <rfr>, <svr> or <gbr>')
        sys.exit()
    
    # Random search of parameters  
    print('\n       >> Peforming randomised search of hyperparameters')
    cv_random = RandomizedSearchCV(estimator=regressor,
                                   param_distributions=random_grid,
                                   n_iter=75, cv=5, verbose=2,
                                   random_state=0, n_jobs=-1, error_score=0.0, 
                                   scoring=scoring)

    # Apply pre-processing steps to training features
    X_prep = preprocess_mapper.fit_transform(X_train)

    # Fit the random search model
    search = cv_random.fit(X_prep, y_train)

    # Print best parameters
    print("\n      Best hyperparameters of best estimator: ", search.best_estimator_.get_params())
    print("\n      Best hyperparameters of search obj: ", search.best_params_)

    return search.best_estimator_


def plot_validation_curve(X_train, y_train, algorithm, preprocess_mapper, param_name, param_range): 
    """
    Plot validation curves to show influence of each hyperparameter on model performance. 

    Parameters: \n
    X_train -- training features \n
    y_train -- training labels \n
    algorithm -- name of algorithm to perform hyperparameter tuning \n
    preprocess_mapper -- DataFrameMapper consisting of all pre-processing steps to perform on data before training \n 
    param_name -- name of parameter to alter \n
    param_range -- values of parameter to test \n

    Returns: None

    """

    if algorithm == 'rfr': 
        estimator = RandomForestRegressor(random_state=0)
        scoring = 'neg_root_mean_squared_error'
        ylabel = 'RMSE'

    elif algorithm == 'gbr': 
        estimator = GradientBoostingRegressor(random_state=0)
        scoring = 'neg_root_mean_squared_error'
        ylabel = 'RMSE'

    elif algorithm == 'svr': 
        estimator = LinearSVR(random_state=0)
        scoring = None
        ylabel = 'R2 score'

    else: 
        print('Input ML algorithm not recognised ! Choose from: <rfr>, <svr> or <gbr>')
        sys.exit()

    # Apply pre-processing steps to training features
    X_train_prep = preprocess_mapper.fit_transform(X_train)

    # Plot validation curves
    rs = ShuffleSplit(n_splits=5, random_state=0, test_size=0.2, train_size=None)
    
    train_scores, test_scores = validation_curve(estimator, X_train_prep, y_train, param_name=param_name, param_range=param_range, cv=rs, n_jobs=-1, verbose=2, scoring=scoring)

    if ylabel == 'RMSE':
        train_scores_mean = -1 * np.nanmean(train_scores, axis=1)
        test_scores_mean = -1 * np.nanmean(test_scores, axis=1)

    elif ylabel == 'R2 score': 
        train_scores_mean = np.nanmean(train_scores, axis=1)
        test_scores_mean = np.nanmean(test_scores, axis=1)

    # Check directory to save plots exists
    if not directory_exists('./plots/hyperparameters'):
        os.makedirs('./plots/hyperparameters')

    plt.xlabel(param_name)
    plt.ylabel(ylabel)
    lw = 2
    plt.plot(param_range, train_scores_mean, label="Training score", color="darkorange", lw=lw)
    plt.plot(param_range, test_scores_mean, label="Cross-validation score", color="navy", lw=lw)
    plt.legend(loc="best")
    plt.savefig('plots/hyperparameters/' + algorithm + '_' + param_name + '.pdf', dpi=300) 
    plt.close()


def svr_maxiter_tolerance(X_train, y_train, X_test, y_test, preprocess_mapper):
    """
    Plot a combination of the maximum number of iterations and the tolerance against the accuracy.
    Based on: Imke Lansky (https://github.com/ImkeLansky/USA-BuildingHeightInference)

    Parameters: \n
    X_train -- train set features \n
    y_train -- train set labels \n
    X_test -- test set features \n
    y_test -- test set labels \n
    preprocess_mapper -- DataFrameMapper consisting of all pre-processing steps to perform on data before training \n 

    Returns: None

    """

    train_results = []
    test_results = []

    tolerances = [1e-3, 1e-4, 1e-5]
    tol_labels = ['1e-3', '1e-4', '1e-5']
    max_iter = [100, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000]

    # Apply pre-processing steps
    X_train_pp = preprocess_mapper.fit_transform(X_train)
    X_test_pp = preprocess_mapper.transform(X_test)

    for tolerance in tolerances:

        temp_train = []
        temp_test = []

        print("         >> Tolerance:", tolerance)

        for iteration in max_iter:
            print("         >> Max. iterations:", iteration)

            svr = LinearSVR(tol=tolerance, max_iter=iteration, random_state=0)
            svr.fit(X_train_pp, y_train)
            y_train_predict = svr.predict(X_train_pp)

            # Score train set 
            r2_train = r2_score(y_train, y_train_predict)
            temp_train.append(r2_train)

            y_test_predict = svr.predict(X_test_pp)

            # Score test set
            r2_test = r2_score(y_test, y_test_predict)
            temp_test.append(r2_test)

        train_results.append(temp_train)
        test_results.append(temp_test)

    lw = 2
    for i in range(len(train_results)):
        label_train = 'Train (tol' + tol_labels[i] +')'
        plt.plot(max_iter, train_results[i], label=label_train, lw=lw)
        label_test = 'Test (tol' + tol_labels[i] +')'
        plt.plot(max_iter, test_results[i], label=label_test, lw=lw)

    plt.legend(loc='best')
    plt.xlabel('Maximum number of iterations')
    plt.ylabel('R2 Score')

    # Check directory to save plots exists
    if not directory_exists('./plots/hyperparameters'):
        os.makedirs('./plots/hyperparameters')

    plt.savefig('plots/hyperparameters/svr_maxiter_tol.pdf', dpi=300) 
    plt.close()


def main(params):

    # Load parameters 
    jparams = json.load(open(params))

    for algorithm in jparams["tuned_models"].keys():

        model = jparams["tuned_models"][algorithm]

        if os.path.exists('data/train_sets/train_set_' + model + '.csv') and os.path.exists('data/test_sets/test_set_' + model + '.csv'):
           
            # Read train and test sets from csv into dataframe
            train_set = pd.read_csv('data/train_sets/train_set_' + model + '.csv', index_col=0, dtype={'bag_id':'str'})
            test_set = pd.read_csv('data/test_sets/test_set_' + model + '.csv', index_col=0, dtype={'bag_id':'str'})
            
            # Split labels from features
            X_train, y_train = train_models.split_features_labels(train_set, jparams["id_column"], jparams["labels_column"])
            X_test, y_test = train_models.split_features_labels(test_set, jparams["id_column"], jparams["labels_column"])

            # Perform model training steps (including parameter tuning)
            train_models.training_steps(X_train, y_train, model, algorithm, tune_params=True)
            
            # Plot validation curves (if True)
            plot_validation_curves = False

            if plot_validation_curves: 

                # Pre-processing steps
                mapper = train_models.preprocess_mapper(X_train)

                print('\n   >> Plotting validation curves: {0}'.format(algorithm))

                if algorithm == 'rfr': 

                    print('\n       >> n_estimators')
                    plot_validation_curve(X_train, y_train, algorithm, mapper, "n_estimators", [1, 2, 4, 8, 16, 32, 64, 128, 256, 512])

                    print('\n       >> max_depth')
                    plot_validation_curve(X_train, y_train, algorithm, mapper, "max_depth", np.linspace(1, 35, 35, dtype=int))
                    
                    print('\n       >> min_samples_split')
                    samples_start = np.linspace(2, 24, 12, dtype=int)
                    samples_end = np.linspace(50, 100, num=6, dtype=int)
                    plot_validation_curve(X_train, y_train, algorithm, mapper, "min_samples_split", np.hstack((samples_start, samples_end)))

                    print('\n       >> min_samples_leaf')
                    samples_start = np.linspace(1, 23, 12, dtype=int)
                    samples_end = np.linspace(50, 100, num=6, dtype=int)
                    plot_validation_curve(X_train, y_train, algorithm, mapper, "min_samples_leaf", np.hstack((samples_start, samples_end)))

                    print('\n       >> max_features')
                    plot_validation_curve(X_train, y_train, algorithm, mapper, "max_features", np.linspace(0.1, 1.0, 10))

                    print('\n       >> max_samples')
                    plot_validation_curve(X_train, y_train, algorithm, mapper, "max_samples", [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99])

                elif algorithm == 'gbr': 

                    print('\n       >> n_estimators')
                    plot_validation_curve(X_train, y_train, algorithm, mapper, "n_estimators", [1, 2, 4, 8, 16, 32, 64, 128, 256, 512])

                    print('\n       >> learning_rate')
                    plot_validation_curve(X_train, y_train, algorithm, mapper, "learning_rate", [1, 0.5, 0.25, 0.1, 0.05, 0.025, 0.01])

                    print('\n       >> max_depth')
                    plot_validation_curve(X_train, y_train, algorithm, mapper, "max_depth", np.linspace(1, 35, 35, dtype=int))
                    
                    print('\n       >> min_samples_split')
                    samples_start = np.linspace(2, 24, 12, dtype=int)
                    samples_end = np.linspace(50, 100, num=6, dtype=int)
                    plot_validation_curve(X_train, y_train, algorithm, mapper, "min_samples_split", np.hstack((samples_start, samples_end)))

                    print('\n       >> min_samples_leaf')
                    samples_start = np.linspace(1, 23, 12, dtype=int)
                    samples_end = np.linspace(50, 100, num=6, dtype=int)
                    plot_validation_curve(X_train, y_train, algorithm, mapper, "min_samples_leaf", np.hstack((samples_start, samples_end)))

                    print('\n >> max_features')
                    plot_validation_curve(X_train, y_train, algorithm, mapper, "max_features", np.linspace(0.1, 1.0, 10))

                elif algorithm == 'svr': 

                    print('\n       >> epsilon')
                    plot_validation_curve(X_train, y_train, algorithm, mapper, "epsilon", np.linspace(0, 5, 11))

                    print('\n       >> C')
                    plot_validation_curve(X_train, y_train, algorithm, mapper, "C", np.linspace(1e-4, 1, 10))

                    print('\n       >> max iterations / tolerance')
                    svr_maxiter_tolerance(X_train, y_train, X_test, y_test, mapper)


if __name__ == '__main__':
    main(sys.argv[1])