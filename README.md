# Inference of building floor count in the Netherlands

This repository contains the code developed during the graduation project of Ellie Roy for the MSc in Geomatics at TU Delft. 
For further details, the thesis report is available [here](http://resolver.tudelft.nl/uuid:6de4255c-ab2b-49c2-a282-ed779de092a1). 

## Introduction 

Data on the number of floors is required for a variety of applications, ranging from energy demand estimation to flood response plans. 
Despite this, open data on the number of floors is currently not available at a nationwide level in the Netherlands.
This means that it must be inferred from other available data. 
Automatic methods usually involve dividing the estimated height of a building by an assumed storey height. 
In some cases, this simple approach limits the accuracy of the results. 
Therefore, the goal of this thesis was to develop an alternative method to automatically infer the number of floors. 

The alternative method was based on machine learning. 
Three different algorithms were tested and compared: Random Forest, Gradient Boosting and Support Vector Regression.
These algorithms were trained using data on the number of floors obtained from four municipalities in the Netherlands.
In addition, 25 features were derived from cadastral attributes, building geometry and neighbourhood census data.
These features were tested in different combinations in order to determine whether a specific subset yielded better results. 
Furthermore, a comparison was made between features derived from 3D building models at different levels of detail.

An overview of the methdology is provided by the flowchart below: 

<p float="left">
  <img src="./images/flowchart.jpg" width="95%"/> 
</p>

## Dependencies 
The code was written in `Python (v3.9.7)`.
A `PostgreSQL (v10.19)` database extended with `PostGIS (v3.0.1)` was used to store the data used during the analysis. 

The implementation depends on the following Python libraries: 

* [geopandas v0.9.0](https://pypi.org/project/geopandas/0.9.0/)
* [pandas v1.3.1](https://pypi.org/project/pandas/1.3.1/)
* [numpy v1.21.2](https://pypi.org/project/numpy/1.21.2/)
* [psycopg2 v2.9.1](https://pypi.org/project/psycopg2/2.9.1/)
* [pyvista v0.31.3](https://pypi.org/project/pyvista/0.31.3/)
* [scipy v1.7.1](https://pypi.org/project/scipy/1.7.1/)
* [statsmodels v0.13.0](https://pypi.org/project/statsmodels/0.13.0/)
* [scikit-learn v0.24.2](https://pypi.org/project/scikit-learn/0.24.2/)
* [sklearn-pandas v2.2.0](https://pypi.org/project/sklearn-pandas/2.2.0/)
* [joblib v1.1.0](https://pypi.org/project/joblib/1.1.0/)
* [seaborn v0.11.1](https://pypi.org/project/seaborn/0.11.1/)
* [matplotlib v3.4.3](https://pypi.org/project/matplotlib/3.4.3/)
* [tqdm v4.62.0](https://pypi.org/project/tqdm/4.62.0/)
* [val3ditypy v0.2](https://github.com/tudelft3d/val3ditypy/releases/tag/0.2)

The conda environment used during the project can be recreated from the `environment.yml` file using the following command: `conda env create -f environment.yml`. However, the `val3ditypy` library must be installed separately (see installation details [here](https://github.com/tudelft3d/val3ditypy)). 

## Datasets

The implementation relies on three main datasets: 

* [BAG](https://data.overheid.nl/en/dataset/0ff83e1e-3db5-4975-b2c1-fbae6dd0d8e0) (04-2020)
* [3D BAG](https://3dbag.nl/en/download) (v21.03.1)
* [CBS wijken en buurten](https://data.overheid.nl/dataset/09f5479a-50f9-45ed-b727-91bf141d14f4) (2019)

Additional data on building type was also obtained from [this](https://www.arcgis.com/home/item.html?id=fa01ef63321e482e9b2c55620e554ffc) dataset maintained by ESRI. 

## Usage

### Data preparation 

* **Training data standardisation**: `python3 get_train_data.py params.json` is used to standardise the format of the training data on the number of floors obtained from different municipalities.
* **Data retrieval**: `python3 retrieve_data.py params.json` is used to retrieve data from the datasets list above. This data is either used directly as features or required to compute other features (e.g. those based on building geometry). 
* **Extract 2D features**: `python3 extract_2d_features.py params.json` is used to extract features from the 2D footprint geometry. 
* **Extract 3D features**: `python3 extract_3d_features.py params.json` is used to extract features from the 3D building geometry.
* **Data cleaning**: `python3 clean_data.py params.json` is used to perform the main data cleaning steps. 
* **Extract reference model**: `python3 get_ref_model.py params.json` is used to calculate the number of floors using height- and area-based approaches to generate a reference model to compare the predictions to. 

All data preparation steps can be performed by running: `python3 data_prep.py params.json`. 

### Modelling and prediction 

* **Training**: `python3 train_models.py params.json` is used to train the models listed by the `models_to_train` parameter (see section below). 
* **Feature selection**: `python3 select_features.py params.json` is used to select different feature subsets for the models listed by the `feature_selection_models` parameter (see section below). 
* **Hyperparameter tuning**: `python3 tune_models.py params.json` is used to tune the hyperparameters of the models listed by the `tuned models` parameter (see section below). 
* **Model evaluation**: `python3 test_models.py params.json` is used to make predictions on the test set and compute error metrics. 

### Analysis 
* **Visualise training data**: `python3 visualise_data.py params.json` is used to generate different plots to visualise the training data. 
* **Compute statistical measures**: `python3 compute_stats.py params.json` is used to compute the correlation coefficient and VIF score of each feature. 
* **Analysis of results**: `python3 analyse_results.py params.json` is used to generate plots to analyse the gross errors and impact of rounding for the best predictive model. 
* **Compare predictions to reference model**: `python3 compare_to_ref.py params.json` is used to compute error metrics to compare the performance of the best predictive model, reference model and a model that always predicts the mean number of floors. 
* **Case study analysis**: `python3 case_study.py params.json` is used to predict the number of floors of (mixed-)residential buildings located in the municipalities listed by the `case_study_tables` parameter (see section below). 

## Parameters

The `params.json` file contains all parameters that can be set by the user. These parameters are defined as follows: 

* `models_to_train`: list of models to train
* `feature_selection_models`: list of models to perform feature selection for
* `tuned models`: name of model that should be tuned for each algorithm
* `use_tuned_params`: list of models that should use the same hyperparameters as the best estimator
* `best_estimator`: name of joblib file used to store the pipeline of the best estimator
* `features`: list of features used by each model 
* `ml_algorithms`: list of machine learning algorithms to use during training (select from `rfr/gbr/svr`)
* `training_schema`: database schema used to store the training data 
* `training_tables`: database tables used to store the training data in the above schema
* `case_study_schema`: database schema used to store the case study dataa
* `case_study_tables`: database tables used to store the case study data in the above schema
* `id_column`: name of the database column corresponding to the building id
* `labels_column`: name of the database column corresponding to the training labels 
* `text_columns`: names of the database columns corresponding to text/categorical features
* `gemeente_codes`: 4-digit code of each municipality used during the analysis 
* `lods`: levels of detail used to extract 3D geometric features
* `distance_adjacent`: distance used to compute number of adjacent buildings 
* `distance_neighbours`: list of distances used to compute number of neighbouring buildings
* `ceiling_height`: average ceiling height used in height-based calculation of number of floors
* `voxel_scales`: number of voxels to fit the length of each mesh for each LOD (determines grid resolution)
* `class_threshold`: error metrics are computed separately for buildings above (>) and below (<=) this number of floors
* `plot_labels`: text to use for plot labels corresponding to each database column name