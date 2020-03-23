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
* _CRUD_classes
  * authentication()
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
* _manual_ops
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

### class _authentication_
The class to be used to authenticate, through the _get_api_token_ method. 

#### method _get_api_token_
```python
def get_api_token(self, username, password):
    url = url_base + '/token/'
    data = json.dumps({'username':username, 'password':password})
    session = retry_session(retries = 10)
    r = session.post(url, data, verify = False)

    if r.status_code == 200:
        tokenJSON = json.loads(r.text)
        return r.status_code, tokenJSON['access']
    else:
        msgJSON = json.loads(r.text)
        msg = msgJSON['message']
        return r.status_code, msg
```

##### Description
This method returns the necessary token to be used from now on for the API requests. To be able to make use of the API before all it is necessary to sign-up.

##### I/O
* Parameters:
    * _**username**_ (_str_): Username
    * _**password**_ (_str_): Password

* Returns:
    * Status code (_int_)
    * Message (_str_) with the result

### class _dataset_
This class unifies and condenses all the operations related to datasets in their most general sense: uploading them to the workspace, listing them, removing them...

Each and every operation (request) needs the token that is obtained through the class _authentication_. 

#### method _upload_
```python
    def upload(self, *args, **kwargs):  
        str_meth_name = self.class_name + '.' + sys._getframe().f_code.co_name
        input_args = locals()['args']
        input_kwargs = locals()['kwargs']

        return general_call(str_meth_name, input_args, input_kwargs)
```

##### Description


##### I/O
* Parameters:
    * _**token**_ (_str_): API Token
    * _**file_path**_ (_str_): The path where the dataset file is located
    * _**dataset_name**_ (_str_): Custom name for the dataset file
    * _**description**_ (_str_, optional): Optional parameter to specify description if needed for the dataset. If no description is inputted, no description is added to the dataset. 

#### method _upload_
```python
    def upload(self, *args, **kwargs):  
        str_meth_name = self.class_name + '.' + sys._getframe().f_code.co_name
        input_args = locals()['args']
        input_kwargs = locals()['kwargs']

        return general_call(str_meth_name, input_args, input_kwargs)
```

##### Description


##### I/O
* Parameters:
    * _**token**_ (_str_): API Token
    * _**file_path**_ (_str_): The path where the dataset file is located
    * _**dataset_name**_ (_str_): Custom name for the dataset file
    * _**description**_ (_str_, optional): Optional parameter to specify description if needed for the dataset. If no description is inputted, no description is added to the dataset. 

#### method _upload_
```python
    def upload(self, *args, **kwargs):  
        str_meth_name = self.class_name + '.' + sys._getframe().f_code.co_name
        input_args = locals()['args']
        input_kwargs = locals()['kwargs']

        return general_call(str_meth_name, input_args, input_kwargs)
```

##### Description


##### I/O
* Parameters:
    * _**token**_ (_str_): API Token
    * _**file_path**_ (_str_): The path where the dataset file is located
    * _**dataset_name**_ (_str_): Custom name for the dataset file
    * _**description**_ (_str_, optional): Optional parameter to specify description if needed for the dataset. If no description is inputted, no description is added to the dataset. 

#### method _upload_
```python
    def upload(self, *args, **kwargs):  
        str_meth_name = self.class_name + '.' + sys._getframe().f_code.co_name
        input_args = locals()['args']
        input_kwargs = locals()['kwargs']

        return general_call(str_meth_name, input_args, input_kwargs)
```

##### Description


##### I/O
* Parameters:
    * _**token**_ (_str_): API Token
    * _**file_path**_ (_str_): The path where the dataset file is located
    * _**dataset_name**_ (_str_): Custom name for the dataset file
    * _**description**_ (_str_, optional): Optional parameter to specify description if needed for the dataset. If no description is inputted, no description is added to the dataset. 

#### method _upload_
```python
    def upload(self, *args, **kwargs):  
        str_meth_name = self.class_name + '.' + sys._getframe().f_code.co_name
        input_args = locals()['args']
        input_kwargs = locals()['kwargs']

        return general_call(str_meth_name, input_args, input_kwargs)
```

##### Description


##### I/O
* Parameters:
    * _**token**_ (_str_): API Token
    * _**file_path**_ (_str_): The path where the dataset file is located
    * _**dataset_name**_ (_str_): Custom name for the dataset file
    * _**description**_ (_str_, optional): Optional parameter to specify description if needed for the dataset. If no description is inputted, no description is added to the dataset. 

### class _experiment_
The class to be used to authenticate, through the _get_api_token_ method. 

#### method _get_api_token_
```python
def get_api_token(self, username, password):
    url = url_base + '/token/'
    data = json.dumps({'username':username, 'password':password})
    session = retry_session(retries = 10)
    r = session.post(url, data, verify = False)

    if r.status_code == 200:
        tokenJSON = json.loads(r.text)
        return r.status_code, tokenJSON['access']
    else:
        msgJSON = json.loads(r.text)
        msg = msgJSON['message']
        return r.status_code, msg
```

##### Description
This method returns the necessary token to be used from now on for the API requests. To be able to make use of the API before all it is necessary to sign-up.

##### I/O
* Parameters:
    * _**username**_ (_str_): Username
    * _**password**_ (_str_): Password

* Returns:
    * Status code (_int_)
    * Message (_str_) with the result

### class _project_
The class to be used to authenticate, through the _get_api_token_ method. 

#### method _get_api_token_
```python
def get_api_token(self, username, password):
    url = url_base + '/token/'
    data = json.dumps({'username':username, 'password':password})
    session = retry_session(retries = 10)
    r = session.post(url, data, verify = False)

    if r.status_code == 200:
        tokenJSON = json.loads(r.text)
        return r.status_code, tokenJSON['access']
    else:
        msgJSON = json.loads(r.text)
        msg = msgJSON['message']
        return r.status_code, msg
```

##### Description
This method returns the necessary token to be used from now on for the API requests. To be able to make use of the API before all it is necessary to sign-up.

##### I/O
* Parameters:
    * _**username**_ (_str_): Username
    * _**password**_ (_str_): Password

* Returns:
    * Status code (_int_)
    * Message (_str_) with the result