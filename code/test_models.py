import json, sys
import os
import csv
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, max_error, make_scorer
from sklearn.model_selection import ShuffleSplit
from joblib import load
import ml_plots
import train_models
from visualise_data import directory_exists


def custom_accuracy(y_true, y_pred):
    """
    Compute accuracy after rounding predictions. 

    y_true -- training labels \n
    y_pred -- predicted values \n

    Returns: accuracy 

    """

    y_pred = np.rint(y_pred)
    accuracy = accuracy_score(y_true, y_pred)

    return accuracy


def custom_accuracy_per_class(y_true, y_pred, x_floors):
    """
    Compute accuracy after rounding predictions for buildings above (>) and below (<=) a certain number of floors.  
    
    y_true -- training labels \n
    y_pred -- predicted values \n
    x_floors -- no. floors to use as threshold value for classes \n

    Returns: accuracy obtained for buildings above and below a certain number of floors

    """

    y_true = y_true.to_numpy()
    y_pred = np.rint(y_pred)
    indexes_class1 = np.where(y_true > x_floors)[0].tolist()
    indexes_class2 = np.where(y_true <= x_floors)[0].tolist()

    y_true_class1 = y_true[indexes_class1]
    y_pred_class1 = y_pred[indexes_class1]
    abs_diff_class1 = abs(y_true_class1 - y_pred_class1)
    accuracy_class1 = np.count_nonzero(abs_diff_class1==0) / len(y_true_class1) 

    y_true_class2 = y_true[indexes_class2]
    y_pred_class2 = y_pred[indexes_class2]
    abs_diff_class2 = abs(y_true_class2 - y_pred_class2)
    accuracy_class2 = np.count_nonzero(abs_diff_class2==0) / len(y_true_class2) 

    return accuracy_class1, accuracy_class2


def custom_mae(y_true, y_pred):
    """
    Compute MAE after rounding predictions. 

    y_true -- training labels \n
    y_pred -- predicted values \n

    Returns: MAE

    """

    y_pred = np.rint(y_pred)
    mae = mean_absolute_error(y_true, y_pred)

    return mae


def custom_mae_per_class(y_true, y_pred, x_floors):
    """
    Compute MAE after rounding predictions for buildings above (>) and below (<=) a certain number of floors.  
    
    y_true -- training labels \n
    y_pred -- predicted values \n
    x_floors -- no. floors to use as threshold value for classes \n

    Returns: MAE obtained for buildings above and below a certain number of floors

    """

    y_true = y_true.to_numpy()
    y_pred = np.rint(y_pred)
    indexes_class1 = np.where(y_true > x_floors)[0].tolist()
    indexes_class2 = np.where(y_true <= x_floors)[0].tolist()

    y_true_class1 = y_true[indexes_class1]
    y_pred_class1 = y_pred[indexes_class1]
    mae_class1 = mean_absolute_error(y_true_class1, y_pred_class1)

    y_true_class2 = y_true[indexes_class2]
    y_pred_class2 = y_pred[indexes_class2]
    mae_class2 = mean_absolute_error(y_true_class2, y_pred_class2)

    return mae_class1, mae_class2


def custom_rmse(y_true, y_pred):
    """
    Compute RMSE after rounding predictions. 

    y_true -- training labels \n
    y_pred -- predicted values \n

    Returns: RMSE

    """

    y_pred = np.rint(y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)

    return rmse


def custom_rmse_per_class(y_true, y_pred, x_floors):
    """
    Compute RMSE after rounding predictions for buildings above (>) and below (<=) a certain number of floors.  
    
    y_true -- training labels \n
    y_pred -- predicted values \n
    x_floors -- no. floors to use as threshold value for classes \n

    Returns: RMSE obtained for buildings above and below a certain number of floors

    """

    y_true = y_true.to_numpy()
    y_pred = np.rint(y_pred)
    indexes_class1 = np.where(y_true > x_floors)[0].tolist()
    indexes_class2 = np.where(y_true <= x_floors)[0].tolist()

    y_true_class1 = y_true[indexes_class1]
    y_pred_class1 = y_pred[indexes_class1]
    mse_class1 = mean_squared_error(y_true_class1, y_pred_class1)
    rmse_class1 = np.sqrt(mse_class1)

    y_true_class2 = y_true[indexes_class2]
    y_pred_class2 = y_pred[indexes_class2]
    mse_class2 = mean_squared_error(y_true_class2, y_pred_class2)
    rmse_class2 = np.sqrt(mse_class2)

    return rmse_class1, rmse_class2


def custom_max_err(y_true, y_pred):
    """
    Compute maximum error after rounding predictions. 

    y_true -- training labels \n
    y_pred -- predicted values \n

    Returns: maximum error

    """

    y_pred = np.rint(y_pred)
    max_err = max_error(y_true, y_pred)

    return max_err


def custom_max_err_per_class(y_true, y_pred, x_floors):
    """
    Compute maximum error after rounding predictions for buildings above (>) and below (<=) a certain number of floors.  
    
    y_true -- training labels \n
    y_pred -- predicted values \n
    x_floors -- no. floors to use as threshold value for classes \n

    Returns: maximum error obtained for buildings above and below a certain number of floors

    """

    y_true = y_true.to_numpy()
    y_pred = np.rint(y_pred)
    indexes_class1 = np.where(y_true > x_floors)[0].tolist()
    indexes_class2 = np.where(y_true <= x_floors)[0].tolist()

    y_true_class1 = y_true[indexes_class1]
    y_pred_class1 = y_pred[indexes_class1]
    max_class1 = max_error(y_true_class1, y_pred_class1)

    y_true_class2 = y_true[indexes_class2]
    y_pred_class2 = y_pred[indexes_class2]
    max_class2 = max_error(y_true_class2, y_pred_class2)

    return max_class1, max_class2


def main(params):

    # Load parameters 
    jparams = json.load(open(params))

    if directory_exists('./models'):

        # Define path to models and order files by date/time created
        search_dir = './models'
        files = os.listdir(search_dir)
        files.sort(key=lambda fn: os.path.getmtime(os.path.join(search_dir, fn)))

        files = [file for file in files if '.joblib' in file]

        for filename, i in zip(files, range(len(files))):
        
            # Get information about model from file name
            file_info = filename[:-7].split('_')
            model = 'model_' + file_info[2]
            algorithm = file_info[3]
            tuning_description = file_info[4]
            
            if os.path.exists('data/train_sets/train_set_' + model + '.csv') and os.path.exists('data/test_sets/test_set_' + model + '.csv'):
                
                print('\n>> Testing model: {0} ({1} / {2})'.format(model, algorithm, tuning_description))

                # Read train and test sets from csv into dataframe
                train_set = pd.read_csv('data/train_sets/train_set_' + model + '.csv', index_col=0, dtype={'bag_id':'str'})
                test_set = pd.read_csv('data/test_sets/test_set_' + model + '.csv', index_col=0, dtype={'bag_id':'str'})

                # Split labels from features
                X_train, y_train = train_models.split_features_labels(train_set, jparams["id_column"], jparams["labels_column"])
                X_test, y_test = train_models.split_features_labels(test_set, jparams["id_column"], jparams["labels_column"])

                # Load model pipeline
                print('\n   >> Loading model')
                pipeline = load('./models/' + filename)

                print('\n   >> Performing model evaluation') 

                # Use pipeline to make predictions on train/test sets
                y_pred_train = pipeline.predict(X_train)
                y_pred_test = pipeline.predict(X_test)

                # Class threshold 
                x_floors = jparams["class_threshold"]

                # Compute accuracy 
                accuracy_train = np.around(custom_accuracy(y_train, y_pred_train)*100, 1)
                accuracy_test = np.around(custom_accuracy(y_test, y_pred_test)*100, 1)

                # Compute accuracy per class
                accuracy_above_train, accuracy_below_train = custom_accuracy_per_class(y_train, y_pred_train, x_floors)
                accuracy_above_train, accuracy_below_train = np.around(accuracy_above_train*100, 1), np.around(accuracy_below_train*100, 1)
                accuracy_above_test, accuracy_below_test = custom_accuracy_per_class(y_test, y_pred_test, x_floors)
                accuracy_above_test, accuracy_below_test = np.around(accuracy_above_test*100, 1), np.around(accuracy_below_test*100, 1)

                # Compute MAE and plot absolute error
                mae_train = np.around(custom_mae(y_train, y_pred_train), 2)
                mae_test = np.around(custom_mae(y_test, y_pred_test), 2)
                ml_plots.plot_abs_error(y_train, y_pred_train, jparams, model, algorithm, tuning_description, 'train')
                ml_plots.plot_abs_error(y_test, y_pred_test, jparams, model, algorithm, tuning_description, 'test')

                # Compute MAE per class
                mae_above_train, mae_below_train = custom_mae_per_class(y_train, y_pred_train, x_floors)
                mae_above_train, mae_below_train = np.around(mae_above_train, 2), np.around(mae_below_train, 2)
                mae_above_test, mae_below_test = custom_mae_per_class(y_test, y_pred_test, x_floors)
                mae_above_test, mae_below_test = np.around(mae_above_test, 2), np.around(mae_below_test, 2)

                # Compute RMSE
                rmse_train = np.around(custom_rmse(y_train, y_pred_train), 2)
                rmse_test = np.around(custom_rmse(y_test, y_pred_test), 2)

                # Compute RMSE per class
                rmse_above_train, rmse_below_train = custom_rmse_per_class(y_train, y_pred_train, x_floors)
                rmse_above_train, rmse_below_train = np.around(rmse_above_train, 2), np.around(rmse_below_train, 2)
                rmse_above_test, rmse_below_test = custom_rmse_per_class(y_test, y_pred_test, x_floors)
                rmse_above_test, rmse_below_test = np.around(rmse_above_test, 2), np.around(rmse_below_test, 2)

                # Compute max error 
                max_err_train = custom_max_err(y_train, y_pred_train)
                max_err_test = custom_max_err(y_test, y_pred_test)

                # Compute max error per class
                max_err_above_train, max_err_below_train = custom_max_err_per_class(y_train, y_pred_train, x_floors)
                max_err_above_test, max_err_below_test = custom_max_err_per_class(y_test, y_pred_test, x_floors)

                # Save results to csv file
                evaluation_results = [model, algorithm, tuning_description, mae_train, mae_test, rmse_train, rmse_test, max_err_train, max_err_test, accuracy_train, accuracy_test] 
                fname = 'model_evaluation.csv'
                
                if i == 0:
                # Write to csv file on first iteration
                    with open('models/' + fname, 'w', encoding='UTF8') as f:
                        fieldnames = ['model', 'algorithm', 'tuning_description', 'mae_train', 'mae_test', 'rmse_train', 'rmse_test', 'max_err_train', 'max_err_test', 'accuracy_train', 'accuracy_test']
                        writer = csv.writer(f)
                        writer.writerow(fieldnames)
                        writer.writerow(evaluation_results)
                else:
                    # Append to existing csv file on all other iterations
                    with open('models/' + fname, 'a') as f:
                        writer = csv.writer(f)
                        writer.writerow(evaluation_results)

                # Save results per class to csv file
                evaluation_results_per_class = [model, algorithm, tuning_description, mae_above_test, mae_below_test, rmse_above_test, rmse_below_test, max_err_above_test, max_err_below_test, accuracy_above_test, accuracy_below_test] 
                fname_per_class = 'model_evaluation_per_class.csv'

                if i == 0:
                # Write to csv file on first iteration
                    with open('models/' + fname_per_class, 'w', encoding='UTF8') as f:
                        fieldnames_per_class = ['model', 'algorithm', 'tuning_description', 'mae_above_test', 'mae_below_test', 'rmse_above_test', 'rmse_below_test', 'max_err_above_test', 'max_err_below_test', 'accuracy_above_test', 'accuracy_below_test']
                        writer = csv.writer(f)
                        writer.writerow(fieldnames_per_class)
                        writer.writerow(evaluation_results_per_class)
                else:
                    # Append to existing csv file on all other iterations
                    with open('models/' + fname_per_class, 'a') as f:
                        writer = csv.writer(f)
                        writer.writerow(evaluation_results_per_class)

                # Create custom scorers that round predictions to closest integer
                custom_accuracy_score = make_scorer(custom_accuracy)
                custom_mae_error = make_scorer(custom_mae)
                custom_max_error = make_scorer(custom_max_err)
                custom_rmse_error = make_scorer(custom_rmse)
                
                # Impurity-based feature importance
                if algorithm == 'rfr' or algorithm == 'gbr': 
                    print('\n   >> Determining impurity-based feature importance')
                    feature_names = pipeline['preprocess'].transform(X_train).columns
                    ml_plots.plot_impurity_importance(pipeline, feature_names, algorithm, model, tuning_description, jparams)
                
                # Coefficients of SVR
                if algorithm == 'svr': 
                    print('\n   >> Determing model coefficients')
                    feature_names = pipeline['preprocess'].transform(X_train).columns
                    weights = abs(pipeline['rgr'].coef_)
                    weights_df = pd.DataFrame(sorted(zip(weights, feature_names), reverse=True), columns=['weight', 'feature'])

                    # Check directory to save data exists
                    if not directory_exists('./data/feature_selection/coefs'):
                        os.makedirs('./data/feature_selection/coefs')
        
                    weights_df.to_csv('data/feature_selection/coefs/coefs_' + model + '_' + algorithm + '_' + tuning_description +'.csv')

                # Calculate permutation importance
                calc_perm_importance = False
                if calc_perm_importance:
                    if (model == 'model_1' and algorithm == 'gbr') or (model == 'model_1.2') or (model == 'model_1.4') or (model == 'model_1.5' and algorithm == 'svr'):
                        print('\n   >> Calculating permutation importance')
                        ml_plots.plot_permutation_importance(pipeline, X_train, y_train, custom_accuracy_score, algorithm, model, tuning_description, jparams)

                # Plot learning curves
                plot_learning_curves = False
                if plot_learning_curves:
                    learning_curve_models = ['model_1.2', 'model_1.4', 'model_1.5']
                    if model in learning_curve_models:
                        cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0) # cross validation, each time with 20% data randomly selected as a validation set
                        print('\n   >> Plotting learning curves')
                        ml_plots.plot_learning_curve(pipeline, X_train, y_train, model, algorithm, ylim=[-0.1,1.59], cv=cv, scoring=custom_rmse_error, label='RMSE', tuning=tuning_description)

                # Round predictions
                y_pred_train_rounded = np.rint(y_pred_train)
                y_pred_test_rounded = np.rint(y_pred_test)

                # Check if directories to save predictions exist
                if not directory_exists('data/predictions'):
                    os.makedirs('./data/predictions')

                if not directory_exists('data/predictions/rounded'):
                    os.makedirs('./data/predictions/rounded')

                # Save predictions to csv
                print('\n   >> Saving model predictions')
                np.savetxt('data/predictions/trainset_' + model + '_' + algorithm + '_' + tuning_description + '.csv', y_pred_train, delimiter=',', header='predicted_floors')
                np.savetxt('data/predictions/rounded/trainset_' + model + '_' + algorithm + '_' + tuning_description + '.csv', y_pred_train_rounded, delimiter=',', header='predicted_floors')
                np.savetxt('data/predictions/testset_' + model + '_' + algorithm + '_' + tuning_description + '.csv', y_pred_test, delimiter=',', header='predicted_floors')
                np.savetxt('data/predictions/rounded/testset_' + model + '_' + algorithm + '_' + tuning_description + '.csv', y_pred_test_rounded, delimiter=',', header='predicted_floors')


if __name__ == '__main__':
    main(sys.argv[1])