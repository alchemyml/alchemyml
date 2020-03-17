# AlchemyML API Documentation
Version Date: 2020-03-17
<hr>

## Table of Contents
[TOC]

## Prerequisites
Sys, json, OS, requests.adapters.HTTPAdapter, urllib3.util.retry.Retry, pickle

#### Dependencies

* **Pandas:** _https://pandas.pydata.org/pandas-docs/stable/install.html_
    *    How to install **_Pandas_** via conda: `conda install pandas`

## Module Overview

### Description
AlchemyML API

### Flowchart


## List of scripts and their functions
* CRUD_classes
  * autentication()
    * get_api_token
  * dataset()
    * upload
    * view
    * update
    * delete
    * statistical_descriptors
  * experiment()
    * create
    * view
    * update
    * delete
    * statistical_descriptors
    * results
    * add_to_project
    * extract_from_project
    * send
  * project()
    * create
    * view
    * update
    * delete
* manual_ops
  * actions()
    * list_preprocessed_dataframes
    * download_dataframe
    * prepare_dataframe
    * encode_dataframe
    * drop_highly_correlated_components
    * impute_inconsistencies
    * drop_invalid_columns
    * target_column_analysis
    * balancing_dataframe
    * initial_exp_info
    * impute_missing_values
    * merge_cols_into_dt_index
    * detect_experiment_type
    * build_model
    * operational_info
    * detect_outliers
    * impute_outliers
    * download_properties_df

## CRUD_classes.py - Code explanations
Intro

### Prerequisites - Imports
* **Python** packages:
  * JSON: `import json`
  * OS: `import os`
  * Sys: `import sys`
* Functions from **request_handler**:
  * `from request_handler import retry_session, general_call`

### class _autentication_

### method _get_api_token_
```python
Code
```

#### Description
Description

#### I/O
* Parameters:
    * _**username**_ (_str_): Username
    * _**password**_ (_str_): Password

* Returns:
    * Status code (_int_)
    * Message (_str_) with the result
