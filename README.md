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

<p float="left">
  <img src="./images/flowchart.jpg"/> 
</p>

## Dependencies 
The code was written in `Python (v3.9.7)`. The data used during the analysis was stored in a `PostgreSQL (v10.19)` database extended with `PostGIS (v3.0.1)`. The implementation depends on the following Python libraries: 

* [geopandas v0.7.0](https://pypi.org/project/geopandas/)
* [joblib v0.14.1](https://pypi.org/project/joblib/0.14.1/)

## Data

## Usage

### Data preparation 

* **Training data standardisation**: `get_train_data.py`
* **Data retrieval**: `retrieve_data.py`
* **Extract 2D features**: `extract_2d_features.py`
* **Extract 3D features**: `extract_3d_features.py`
* **Data cleaning**: `clean_data.py`
* **Extract reference model**: `get_ref_model.py`

All data preparation steps can be performed by running `main.py`. 

### Modelling and prediction 

* **Train models**: `train_models.py`
* **Select feature subsets**: `select_features.py`
* **Tune models**: `tune_models.py`
* **Evaluate models**: `test_models.py`

### Analysis 
* **Compute statistical measures**: `compute_stats.py`
* **Analysis of results**: `analyse_results.py`
* **Compare predictions to reference model**: `compare_to_ref.py`
* **Case study analysis**: `case_study.py`

## Parameters

The `params.json` file contains all parameters that can be set by the user. These parameters are defined as follows: 

* `models_to_train`: 
* `feature_selection_models`: 
* `tuned models`:
* `use_tuned_params`: 
* `best_estimator`:
* `features`: 
* `ml_algorithms`: 
* `best_estimator`: 
* `training_schema`: