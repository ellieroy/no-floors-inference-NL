import os
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable 
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import learning_curve
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.inspection import permutation_importance
from visualise_data import directory_exists


def plot_hist(series, model, train_test):
    """
    Plot histogram of number of floors in data used by the input model. 

    Parameters: \n
    series -- pandas Series containing data on no. floors \n
    model -- name of model \n
    train_test -- describes whether the data is from the train or test set \n

    Returns: None

    """

    # Check directory to save plots exists
    if not directory_exists('./plots/histograms'):
        os.makedirs('./plots/histograms')

    series.hist(bins=np.arange(max(series)+2)-0.5, log=True, grid=False, edgecolor='white', linewidth=0.8, color='cadetblue', figsize=[8.6, 4.8])
    x_ticks = np.arange(1, max(series)+1, 2)
    plt.xticks(x_ticks)
    plt.xlabel('Number of floors')
    plt.ylabel('Count')
    plt.title('')
    plt.savefig('plots/histograms/hist_' + train_test + '_' + model + '.pdf', dpi=300)
    plt.close()


def plot_confusion_matrix(y_true, y_pred, model, algorithm, tuning, train_test): 
    """
    Plot a confusion matrix. 
    
    Parameters: \n
    y_true -- training labels \n
    y_pred -- predictions \n
    model -- name of model \n
    algorithm -- name of algorithm used by model \n
    tuning -- describes whether model uses default parameters or tuned parameters \n
    train_test -- describes whether the data is from the train or test set \n
    
    Returns: None
    
    """

    # Check directory to save plots exists
    if not directory_exists('./plots/confusion_matrices'):
        os.makedirs('./plots/confusion_matrices')
    
    np.set_printoptions(precision=1) 
    max_value = max(y_true.max(), y_pred.max())
    labels = np.arange(1, max_value+1)
    # np.unique(np.concatenate([np.unique(y_true), np.unique(y_pred)]))
    cm = confusion_matrix(y_true, y_pred, normalize='true', labels=labels.astype(int))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels.astype(int)) 
    fig, ax = plt.subplots(figsize=(15,15))
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(16)
    disp.plot(ax=ax, cmap=plt.cm.Blues, include_values=False, colorbar=False)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.25)
    cbar = fig.colorbar(ScalarMappable(cmap=plt.cm.Blues), cax=cax, orientation='vertical')
    for t in cbar.ax.get_yticklabels():
        t.set_fontsize(16)
    fig.tight_layout()
    plt.savefig('plots/confusion_matrices/cm_' + algorithm + '_' + train_test + '_' + model + '_' + tuning + '.pdf', dpi=300)
    plt.close()


def plot_corr_matrix(corr_matrix, model, params, method):
    """
    Plot correlation matrix. 

    Parameters: \n
    corr_matrix -- correlation matrix to plot \n
    model -- name of model \n
    params -- dictionary of parameters from json parameter file \n
    method -- method used to compute correlation \n
    
    Returns: None

    """

    sns.set_style("ticks")

    fig = plt.figure(figsize=(12, 10))

    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

    heatmap = sns.heatmap(np.abs(corr_matrix), cmap='viridis',
                          annot=False,
                          linewidth=0.5, square=True, mask=mask,
                          linewidths=.5,
                          cbar_kws={"shrink": 0.4, "orientation": "horizontal",
                                    "label": "Correlation"},
                          vmin=0, vmax=1)
    xlabels = []
    for item in heatmap.get_xticklabels():
        xlabels.append(params['plot_labels'][item.get_text()])
    heatmap.set_xticklabels(xlabels, rotation=45, horizontalalignment='right')
    heatmap.set_yticklabels(xlabels)
    heatmap.tick_params(left=False, bottom=False)
    fig.tight_layout()

    # Check directory to save plots exists
    if not directory_exists('./plots/correlation_matrices'):
        os.makedirs('./plots/correlation_matrices')

    plt.savefig("./plots/correlation_matrices/corr_matrix_" + model + "_" + method + ".pdf", bbox_inches="tight", dpi=300)
    plt.close()
    

def plot_learning_curve(pipeline, X, y, model, algorithm, ylim=None, cv=None, n_jobs=None, train_sizes=np.linspace(0.0001, 1, 5), scoring='accuracy', label='Accuracy (%)', tuning='tuned'):
    """
    Generate 3 plots: the test and training learning curve, the training
    samples vs fit times curve, the fit times vs score curve.

    Source: https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html

    Parameters: \n
    pipeline -- machine learning pipeline (including training and pre-processing steps) \n
    X -- training features \n
    y -- training labels \n
    model -- name of model \n
    algorithm -- name of algorithm used by model \n
    ylim -- (optional) to define the limits of the y-axis of the plots \n
    cv -- determines the cross-validation splitting strategy (None means using the default 5-fold cross-validation) \n
    train_sizes -- relative or absolute number of training examples used to generate the learning curve \n
    scoring -- method used to evaluate model performance \n
    label -- plot label corresponding to the scoring method chosen \n
    tuning -- describes whether model uses default parameters or tuned parameters \n

    Returns: None

    """

    sns.set_style("whitegrid")
    fig, axes = plt.subplots(3, 1, figsize=(10, 15))
    
    X_prep = pipeline['preprocess'].fit_transform(X)

    train_sizes, train_scores, test_scores, fit_times, _ = learning_curve(pipeline['rgr'], X_prep, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, return_times=True, scoring=scoring)
    
    if scoring == 'neg_mean_squared_error':
        train_scores = np.sqrt(-train_scores)
        test_scores = np.sqrt(-test_scores)
    
    elif scoring == 'neg_mean_absolute_error':
        train_scores = -train_scores
        test_scores = -test_scores

    elif scoring == 'accuracy':
        train_scores = train_scores * 100
        test_scores = test_scores * 100 

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)

    # Check directory to save plots exists
    if not directory_exists('./plots/learning_curves'):
        os.makedirs('./plots/learning_curves')

    # Plot learning curve
    if ylim is not None:
        axes[0].set_ylim(*ylim)
    axes[0].set_xlabel("Training examples")
    axes[0].set_ylabel(label)
    axes[0].fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
    axes[0].fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1,
                         color="g")
    axes[0].plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
    axes[0].plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")
    axes[0].legend(loc="best")

    # Plot n_samples vs fit_times
    axes[1].plot(train_sizes, fit_times_mean, 'o-')
    axes[1].fill_between(train_sizes, fit_times_mean - fit_times_std,
                         fit_times_mean + fit_times_std, alpha=0.1)
    axes[1].set_xlabel("Training examples")
    axes[1].set_ylabel("Fit times (s)")
    axes[1].set_title("Scalability of the model")
    axes[1].set_ylim([-5,215])

    # Plot fit_time vs score
    axes[2].plot(fit_times_mean, test_scores_mean, 'o-')
    axes[2].fill_between(fit_times_mean, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1)
    axes[2].set_xlabel("Fit times (s)")
    axes[2].set_ylabel(label)
    axes[2].set_title("Performance of the model")
    
    fig.tight_layout(pad=3.0)

    plt.savefig('plots/learning_curves/' + algorithm + '_' + model + '_' + tuning + '.pdf', dpi=300) 
    plt.close()

    return train_scores, test_scores


def plot_impurity_importance(pipeline, feature_names, algorithm, model, tuning, params):
    """
    Plot impurity importance of all features as a bar chart. 
    
    Parameters: \n
    pipeline -- machine learning pipeline \n
    feature_names -- names of features used by model (list) \n
    algorithm -- name of algorithm used by model \n 
    model -- name of model \n
    tuning -- describes whether model uses default parameters or tuned parameters \n 
    params -- dictionary of parameters from json parameter file \n
    
    Returns: None 
    
    """

    impurity_importances = pipeline['rgr'].feature_importances_
    impurity_importance_sorted_idx = np.argsort(impurity_importances)
    impurity_indices = np.arange(0, len(impurity_importances)) + 0.5

    # Check directory to save plots exists
    if not directory_exists('./plots/feature_importance'):
        os.makedirs('./plots/feature_importance')

    fig, ax = plt.subplots()
    ax.barh(impurity_indices, impurity_importances[impurity_importance_sorted_idx], height=0.7)
    ax.set_yticks(impurity_indices)
    ylabels = []
    for item in feature_names[impurity_importance_sorted_idx]:
        ylabels.append(params['plot_labels'][item])
    ax.set_yticklabels(ylabels)
    ax.set_xlabel("Mean decrease in impurity")
    ax.set_ylim((0, len(impurity_importances)))
    ax.set_xlim((0, 1))
    axes = plt.gca()
    axes.xaxis.grid()
    plt.subplots_adjust(left=0.4, bottom=0.1, right=0.95, top=0.95)

    plt.savefig('plots/feature_importance/impurity_importance_' + model + '_' + algorithm + '_' + tuning + '.pdf', dpi=300) 
    plt.close() 

    # Check directory to save data exists
    if not directory_exists('./data/feature_selection/impurity'):
        os.makedirs('./data/feature_selection/impurity')

    importances_df = pd.DataFrame(sorted(zip(impurity_importances, feature_names), reverse=True), columns=['impurity_importance', 'feature'])
    importances_df.to_csv('data/feature_selection/impurity/impurity_importance_' + model + '_' + algorithm + '_' + tuning +'.csv')


def plot_permutation_importance(pipeline, X_train, y_train, scoring, algorithm, model, tuning, params):
    """
    Plot permutation importance of all features as a bar chart. 
    
    Parameters: \n
    pipeline -- machine learning pipeline (including training and pre-processing steps) \n
    X_train -- training features \n
    y_train -- training labels \n
    scoring -- method used to evaluate model performance \n
    algorithm -- name of algorithm used by model \n 
    model -- name of model \n
    tuning -- describes whether model uses default parameters or tuned parameters \n 
    params -- dictionary of parameters from json parameter file \n
    
    Returns: None 
    
    """ 

    X_prep = pipeline['preprocess'].transform(X_train)
    feature_names = X_prep.columns

    result = permutation_importance(pipeline['rgr'], X_prep, y_train, scoring=scoring, n_repeats=10, random_state=42)
    perm_sorted_idx = result.importances_mean.argsort()

    permutation_importances = pd.Series(result.importances_mean, index=feature_names)
    permutation_indices = np.arange(0, len(permutation_importances)) + 0.5

    # Check directory to save plots exists
    if not directory_exists('./plots/feature_importance'):
        os.makedirs('./plots/feature_importance')

    fig, ax = plt.subplots()
    ax.barh(permutation_indices, permutation_importances[perm_sorted_idx], height=0.7)
    ax.set_xlabel("Mean accuracy decrease")
    ax.set_yticks(permutation_indices)
    ylabels = []
    for item in feature_names[perm_sorted_idx]: 
        ylabels.append(params['plot_labels'][item])
    ax.set_yticklabels(ylabels)
    ax.set_ylim((0, len(permutation_importances)))
    ax.set_xlim((0, 1))
    axes = plt.gca()
    axes.xaxis.grid()
    plt.subplots_adjust(left=0.4, bottom=0.1, right=0.95, top=0.95)

    plt.savefig('plots/feature_importance/permutation_importance_' + model + '_' + algorithm + '_' + tuning + '.pdf', dpi=300) 
    plt.close() 

    # Check directory to save data exists
    if not directory_exists('./data/feature_selection/permutation'):
        os.makedirs('./data/feature_selection/permutation')

    importances_df = pd.DataFrame(sorted(zip(result.importances_mean, feature_names), reverse=True), columns=['permutation_importance', 'feature'])
    importances_df.to_csv('data/feature_selection/permutation/permutation_importance_' + algorithm + '_' + model + '_' + tuning +'.csv')


def plot_abs_error(y_true, y_pred, jparams, model, algorithm, tuning, train_test):
    """
    Plot frequency of absolute errors using a histogram and plot the mean absolute error for each number of floors. 

    Parameters: \n
    y_true -- training labels \n
    y_pred -- predicted labels \n
    jparams -- dictionary of parameters from json parameter file \n
    model -- name of model \n 
    algorithm -- name of algorithm used by model \n 
    tuning -- describes whether model uses default parameters or tuned parameters \n 
    train_test -- describes whether the data is from the train or test set \n

    Returns: None

    """

    # Check directory to save plots exists
    if not directory_exists('./plots/mean_abs_error'): 
        os.mkdir('./plots/mean_abs_error')

    abs_error = abs(y_true - np.rint(y_pred))
    abs_error.rename('abs_error', inplace=True)

    abs_error.hist(log=True, grid=False, bins=np.arange(max(abs_error)+2)-0.5, edgecolor='white', linewidth=0.8, color='steelblue')
    x_ticks = np.arange(0, max(abs_error)+1, 1)
    plt.xticks(x_ticks)
    plt.xlabel('Absolute error')
    plt.ylabel('Frequency')
    plt.title('')
    plt.savefig('plots/mean_abs_error/abs_error_' + algorithm + '_' + train_test + '_' + model + '_' + tuning + '.pdf', dpi=300) 
    plt.close()

    sns.set(style="ticks")
    sns.set_style("darkgrid")
    df = pd.concat([y_true, abs_error], axis=1, join='inner')
    mae_per_floor = df.groupby(jparams["labels_column"])['abs_error'].mean()
    mae_per_floor.plot.bar(width=1.0, edgecolor='white', linewidth=0.8, color='steelblue')
    plt.xlabel('Number of floors')
    plt.ylabel('Mean absolute error')
    axes = plt.gca()
    axes.xaxis.grid()
    plt.savefig('plots/mean_abs_error/mae_' + algorithm + '_' + train_test + '_' + model + '_' + tuning + '.pdf', dpi=300) 
    plt.close()
