# AlchemyML API Documentation
Version Date: 2020-03-26
<hr>

## Prerequisites
* Python >= 3.6
* * requests >= 2.22.0
* * urllib3 >= 1.25.7

## Module Overview

### Description
AlchemyML is a multi-environment solution for data exploiation. 

To maximize customer convenience, there are three ways to run it: via the AlchemyML Platform, via the API, and via ad hoc solutions. The one documented below is the second tool, AlchemyML API. 

AlchemyML API is an easy way to use advanced data analysis techniques in Python, accelerating the work of the data scientist and optimizing her/his time and resources. 

AlchemyML API has operations at the dataset level (upload, list, delete...), at the experiment level (create, send, add to project, view metrics and logs...) and at the project level (create, update, delete...). Moreover, it also has specific actions so that the client can perform her/his own experiment manually: pre-process the dataset, remove highly correlated columns, detect outliers, impute missings...

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

## _CRUD_classes.py - Code explanations

### Prerequisites - Imports
* **Python** packages:
  * JSON: `import json`
  * OS: `import os`
  * Sys: `import sys`
* Functions from **_request_handler**:
  * `from ._request_handler import retry_session, general_call`

### class _authentication_
The class to be used to authenticate, through the _get_api_token_ method. 

#### method _get_api_token_
```python
def get_api_token(self, username, password):
    url = url_base + '/token/'
    data = json.dumps({'username':username, 'password':password})
    session = retry_session(retries = 10)
    r = session.post(url, data)

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
    * _**username**_ (_str_): Username. 
    * _**password**_ (_str_): Password. 

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
Through the call to this method, you will be able to upload a dataset.

We recommend you to consider the next points before uploading your dataset:
* The accepted reading formats are: .xlsx, .xls, .csv, .json, .xml, .sql.
* Files whose name contains two extensions will not be accepted. E.g.: 'Iris.xlsx.csv'.
* Your data set should contain at least 50 observations.
* The file must not exceed the size limit specified by the AlchemyML team.
* Make sure that your data are not empty. Otherwise, this file will be rejected.

##### I/O
* Parameters:
    * _**token**_ (_str_): API Token. 
    * _**file_path**_ (_str_): The path where the dataset file is located. 
    * _**dataset_name**_ (_str_): Custom name for the dataset file. 
    * _**description**_ (_str_, optional): Optional parameter to specify description if needed for the dataset. If no description is inputted, no description is added to the dataset. 

#### method _get_
```python
def get(self, *args, **kwargs):  
    str_meth_name = self.class_name + '.' + sys._getframe().f_code.co_name
    input_args = locals()['args']
    input_kwargs = locals()['kwargs']

    return general_call(str_meth_name, input_args, input_kwargs)
```

##### Description
This method lists the datasets available in the workspace.
* By setting the _detail_ parameter to **True** or **False**, you can control receiving the details of each uploaded dataset or simply a list with the names of the datasets.
* By setting the _dataset_name_ parameter, you can control for which datasets return the details.

##### I/O
* Parameters:
    * _**token**_ (_str_): API Token. 
    * _**dataset_name**_ (_str_/_list_, optional): Name or list of names of the dataset(s) for which details will be returned.
    * _**detail**_ (_bool_, optional): Optional boolean parameter to return the details for the specified dataset(s) (False/ True).

#### method _update_
```python
def update(self, *args, **kwargs):  
    str_meth_name = self.class_name + '.' + sys._getframe().f_code.co_name
    input_args = locals()['args']
    input_kwargs = locals()['kwargs']

    return general_call(str_meth_name, input_args, input_kwargs)
```

##### Description
This method gives the option to rename a dataset and/ or update the datasets description. At least one of the two previous options must be selected.

##### I/O
* Parameters:
    * _**token**_ (_str_): API Token. 
    * _**dataset_name**_ (_str_, optional): Name of the dataset to update.
    * _**new_dataset_name**_ (_str_, optional): New name of the specified dataset. If no name is inputted, the dataset won't be renamed.
    * _**new_description**_ (_str_/_list_, optional): New description for the specified dataset. If no description is inputted, the description is not going to be updated.

#### method _delete_
```python
def delete(self, *args, **kwargs):  
    str_meth_name = self.class_name + '.' + sys._getframe().f_code.co_name
    input_args = locals()['args']
    input_kwargs = locals()['kwargs']

    return general_call(str_meth_name, input_args, input_kwargs)
```

##### Description
Through the use of the Delete method you will be able to delete one, several or all uploaded datasets. Note that if a dataset consists of experiments associated with it, you must first remove the experiments that have been created.

AlchemyML is not responsible for any consequences that may be caused by removing one, several or all datasets.

##### I/O
* Parameters:
    * _**token**_ (_str_): API Token. 
    * _**dataset_name**_ (_str_/_list_): Name or list of names of the datasets to be removed from workspace. If _All_ used, then all datasets will be removed. Datasets will be removed only if were not used in any experiment.

#### method _statistical_descriptors_
```python
def statistical_descriptors(self, *args, **kwargs):  
    str_meth_name = self.class_name + '.' + sys._getframe().f_code.co_name
    input_args = locals()['args']
    input_kwargs = locals()['kwargs']

    return general_call(str_meth_name, input_args, input_kwargs)
```

##### Description
This method returns the most relevant statistical descriptors for each column of a dataset.

##### I/O
* Parameters:
    * _**token**_ (_str_): API Token
    * _**dataset_name**_ (_str_): Name of the dataset to return statistical descriptors.

### class _experiment_
This class unifies and condenses all the operations related to experiments in their most general sense: uploading them to the workspace, listing them, removing them... this class also contains the methods for adding experiments to projects, updating them, deleting them...

Each and every operation (request) needs the token that is obtained through the class _authentication_. 

#### method _create_
```python
def create(self, *args, **kwargs):
    str_meth_name = self.class_name + '.' + sys._getframe().f_code.co_name
    input_args = locals()['args']
    input_kwargs = locals()['kwargs']

    return general_call(str_meth_name, input_args, input_kwargs)
```

##### Description
By default, an automatic experiment will be created.

This option implies the execution of a sequence of steps that go from the dataset intake to the construction of the predictive model, including the pre-processing and cleaning of the data.

If the experiment procedure is set to manual, then the user has the possibility to control each phase of the experiment by running the available modules in the desired order. 

The possible operations that can be executed are those that appear in the manual operations section.

##### I/O
* Parameters:
    * _**token**_ (_str_): API Token. 
    * _**experiment_name**_ (_str_): Name used for the creation the experiment. This name is given by the user.
    * _**description**_ (_str_, optional): Optional parameter to specify description if needed for the experiment. If no description is inputted, no description is going to be added to the experiment.
    * _**dataset_name**_ (_str_): Name of the dataset used in the creation of experiment.
    * _**target_column**_ (_str_): Specifying the target column name.
    * _**clients_choice**_ (_str_): Type of experiment. Valid options: Regression, Classification, Time Series, Auto Detect. 
    * _**input_column_names**_ (_str_, optional): Column names of the dataset to use during the experiment. If no input column is introduced, all will be taken by default.
    * _**experiment_procedure**_ (_str_, optional): Valid options are: auto or manual.

#### method _get_
```python
def get(self, *args, **kwargs):
    str_meth_name = self.class_name + '.' + sys._getframe().f_code.co_name
    input_args = locals()['args']
    input_kwargs = locals()['kwargs']

    return general_call(str_meth_name, input_args, input_kwargs)
```

##### Description
Such as the datasets section, this method will let you know which experiments you have in your workspace.
* By setting the detail parameter to **True** or **False** you can control receiving details of each experiment or simply get a list with the names of the experiments.
* By setting the _experiment_name_ parameter, you control for which experiments return the details (one or some).

##### I/O
* Parameters:
    * _**token**_ (_str_): API Token. 
    * _**experiment_name**_ (_str_/_list_, optional): The name or list of experiment names to be listed.
    * _**detail**_ (_bool_, optional): Optional boolean parameter to return the details for the specified experiment(s) (False/ True).

#### method _update_
```python
def update(self, *args, **kwargs):
    str_meth_name = self.class_name + '.' + sys._getframe().f_code.co_name
    input_args = locals()['args']
    input_kwargs = locals()['kwargs']

    return general_call(str_meth_name, input_args, input_kwargs)
```

##### Description
This method gives the option to rename an experiment and/ or update the experiments description. At least one of the two previous options must be selected.

##### I/O
* Parameters:
    * _**token**_ (_str_): API Token. 
    * _**experiment_name**_ (_str_): Name of the experiment to update.
    * _**new_experiment_name**_ (_str_, optional): New name of the specified experiment. If no name is inputted, the experiment is not going to be renamed.
    * _**new_description**_ (_str_/_list_, optional): New description for the specified experiment. If no description is inputted, the description is not going to be updated.

#### method _delete_
```python
def delete(self, *args, **kwargs):
    str_meth_name = self.class_name + '.' + sys._getframe().f_code.co_name
    input_args = locals()['args']
    input_kwargs = locals()['kwargs']

    return general_call(str_meth_name, input_args, input_kwargs)
```

##### Description
Through the use of the endpoint Delete you will be able to delete one, several or all the experiments created.

AlchemyML is not responsible for any consequences that may be caused by removing one, several or all experiments.

##### I/O
* Parameters:
    * _**token**_ (_str_): API Token. 
    * _**experiment_name**_ (_str_/_list_): Name or list of experiment names to be deleted. If _All_ used, then all experiments will be removed.

#### method _statistical_descriptors_
```python
def statistical_descriptors(self, *args, **kwargs):
    str_meth_name = self.class_name + '.' + sys._getframe().f_code.co_name
    input_args = locals()['args']
    input_kwargs = locals()['kwargs']

    return general_call(str_meth_name, input_args, input_kwargs)
```

##### Description
This method returns the most relevant statistical descriptors for each column of the preprocessed dataset used in the experiment creation.

##### I/O
* Parameters:
    * _**token**_ (_str_): API Token. 
    * _**experiment_name**_ (_str_): Name of the experiment to return statistical descriptors.
    * _**dataset_name**_ (_str_): Name of the dataset used in the experiment creation.

#### method _results_
```python
def results(self, *args, **kwargs):
    str_meth_name = self.class_name + '.' + sys._getframe().f_code.co_name
    input_args = locals()['args']
    input_kwargs = locals()['kwargs']

    return general_call(str_meth_name, input_args, input_kwargs)
```

##### Description
The creation of an experiment (previous method CREATE) returns the results of that experiment. This method gives the option to retrieve the previous results whenever these are needed.

The results are delivered in a JSON structure consisting of two keys: **log**, **model_metrics**.
* **log** contains the information related to the decisions that AlchemyML has taken throughout the creation of the experiment, until finishing the construction of the predictive model.
* On the other hand, **model_metrics** will include the analytical information of these results: metrics obtained, relevant variables, type of experiment, etc.

##### I/O
* Parameters:
    * _**token**_ (_str_): API Token. 
    * _**experiment_name**_ (_str_): Name of the experiment to return the results.

#### method _add_to_project_
```python
def add_to_project(self, *args, **kwargs):
    str_meth_name = self.class_name + '.' + sys._getframe().f_code.co_name
    input_args = locals()['args']
    input_kwargs = locals()['kwargs']

    return general_call(str_meth_name, input_args, input_kwargs)
```

##### Description
This method gives the possibility to include an experiment or various into a specified project.

Projects are the way to order and group different experiments that are included within a general topic. For example, you could create a project under the theme of Smart Cities that includes experiments related to this topic.

##### I/O
* Parameters:
    * _**token**_ (_str_): API Token. 
    * _**associated_experiments**_ (_str_/_list_): Name or list of experiment names to be included into a specified project.
    * _**project_name**_ (_str_): The name of the project in which experiment(s) will be included.
    * _**description**_ (_str_, optional): Optional parameter to specify description if needed for the project. If no description is inputted, no description is going to be added to the experiment.

#### method _extract_from_project_
```python
def extract_from_project(self, *args, **kwargs):
    str_meth_name = self.class_name + '.' + sys._getframe().f_code.co_name
    input_args = locals()['args']
    input_kwargs = locals()['kwargs']

    return general_call(str_meth_name, input_args, input_kwargs)
```

##### Description
Given a project this method gives the possibility to extract specified experiments from it.

##### I/O
* Parameters:
    * _**token**_ (_str_): API Token. 
    * _**experiment_name**_ (_str_/_list_): Name or list of experiment names that are desired to be extracted from a given project.
    * _**project_name**_ (_str_): The project from which will be extracted the specified experiments.

#### method _send_
```python
def send(self, *args, **kwargs):
    str_meth_name = self.class_name + '.' + sys._getframe().f_code.co_name
    input_args = locals()['args']
    input_kwargs = locals()['kwargs']

    return general_call(str_meth_name, input_args, input_kwargs)
```

##### Description
This endpoint gives the possibility to send one or various experiments to another registered user.

If the user exists, a confirmation email will be sent. When the recipient confirms that he wants to receive an experiment from another user, an exact copy of the experiment will appear within his/her experiments section and will also be visible through the Workspace.

##### I/O
* Parameters:
    * _**token**_ (_str_): API Token. 
    * _**destination_email**_ (_str_): The receivers email address.
    * _**experiment_name**_ (_str_/_list_): The name or list of experiment names to be sent.

### class _project_
This class unifies and condenses all the operations related to projects in their most general sense: creating them, listing them, deleting them... 

Each and every operation (request) needs the token that is obtained through the class _authentication_. 

#### method _create_
```python
def create(self, *args, **kwargs):
    str_meth_name = self.class_name + '.' + sys._getframe().f_code.co_name
    input_args = locals()['args']
    input_kwargs = locals()['kwargs']

    return general_call(str_meth_name, input_args, input_kwargs) 
```

##### Description
This method creates a new project.

Projects are the way to order and group different experiments that are included within a general topic. For example, you could create a project under the theme of Smart Cities that includes experiments related to this topic.

##### I/O
* Parameters:
    * _**token**_ (_str_): API Token.  
    * _**project_name**_ (_str_): Name of the project.
    * _**description**_ (_str_, optional): Optional parameter to specify description if needed for the project. If no description is inputted, no description is going to be added to the project.
    * _**associated_experiments**_ (_str_/_list_, optional): Name or list of experiment names to be added to the project. If no experiments are inputted, an empty project is going to be created.

#### method _get_
```python
def get(self, *args, **kwargs):
    str_meth_name = self.class_name + '.' + sys._getframe().f_code.co_name
    input_args = locals()['args']
    input_kwargs = locals()['kwargs']

    return general_call(str_meth_name, input_args, input_kwargs) 
```

##### Description
Such as the datasets section, this method will let you know which projects you have in your workspace.
* By setting the detail parameter to **True** or **False** you can control receiving details of each project or simply get a list with the names of the projects.
* By setting the _project_name_ parameter, you control for which projects return the details (one or some).

##### I/O
* Parameters:
    * _**token**_ (_str_): API Token.  
    * _**project_name**_ (_str_/_list_, optional): Name or list of names of the project(s).
    * _**detail**_ (_bool_, optional): Optional boolean parameter to return the details for the specified project(s) (False/ True).

#### method _update_
```python
def update(self, *args, **kwargs):
    str_meth_name = self.class_name + '.' + sys._getframe().f_code.co_name
    input_args = locals()['args']
    input_kwargs = locals()['kwargs']

    return general_call(str_meth_name, input_args, input_kwargs) 
```

##### Description
This method gives the option to rename a project and/ or update the projects description. At least one of the two previous options must be selected.

##### I/O
* Parameters:
    * _**token**_ (_str_): API Token.  
    * _**project_name**_ (_str_): Name of the project to be updated.
    * _**new_project_name**_ (_str_, optional): New name of the specified project. If no name is inputted, the project is not going to be renamed.
    * _**new_description**_ (_str_/_list_, optional): New description for the specified project. If no description is inputted, the description is not going to be updated.

#### method _delete_
```python
def delete(self, *args, **kwargs):
    str_meth_name = self.class_name + '.' + sys._getframe().f_code.co_name
    input_args = locals()['args']
    input_kwargs = locals()['kwargs']

    return general_call(str_meth_name, input_args, input_kwargs) 
```

##### Description
Through the use of the method Delete you will be able to delete one, several or all the projects created.

AlchemyML is not responsible for any consequences that may be caused by removing one, several or all projects.

##### I/O
* Parameters:
    * _**token**_ (_str_): API Token.  
    * _**project_name**_ (_str_/_list_): Name or list of names of the projects to be deleted. If All used then, all projects will be removed.
 
## _manual_ops.py - Code explanations

### Prerequisites - Imports
* **Python** packages:
  * Sys: `import sys`
* Functions from **_request_handler**:
  * `from ._request_handler import general_call`

### class _actions_
Class that encompasses all the operations available in a manual experiment. 

#### method _list_preprocessed_dataframes_
```python
def list_preprocessed_dataframes(self, *args, **kwargs):
    str_meth_name = sys._getframe().f_code.co_name
    input_args = locals()['args']
    input_kwargs = locals()['kwargs']

    return general_call(str_meth_name, input_args, input_kwargs)    
```

##### Description
Method for listing the available processed dataframes for the given experiment.

##### I/O
* Parameters:
    * _**token**_ (_str_): API Token. 
    * _**experiment_name**_ (_str_): Experiment name to which processed dataframes will be returned.
    * _**download**_ (_bool_, optional): Optional boolean parameter to be set up if results needed to be downloaded.

#### method _download_dataframe_
```python
def download_dataframe(self, *args, **kwargs):
    str_meth_name = sys._getframe().f_code.co_name
    input_args = locals()['args']
    input_kwargs = locals()['kwargs']

    return general_call(str_meth_name, input_args, input_kwargs)    
```

##### Description
As the name of the endpoint suggests, this method gives the option to download the available processed dataframes for a given experiment.
* If keyword all in dataframe_name, all available dataframes will be downloaded.
* If unknown the available processed dataframes, call first List preprocessed dataframes.

##### I/O
* Parameters:
    * _**token**_ (_str_): API Token. 
    * _**experiment_name**_ (_str_): Name of experiment for which dataframe(s) needed to be download.
    * _**dataframe_name**_ (_str_): Dataframe name to be downloaded. Using the keyword all, all dataframes available for the experiment will be downloaded in a rar archive.

#### method _prepare_dataframe_
```python
def prepare_dataframe(self, *args, **kwargs):
    str_meth_name = sys._getframe().f_code.co_name
    input_args = locals()['args']
    input_kwargs = locals()['kwargs']

    return general_call(str_meth_name, input_args, input_kwargs)    
```

##### Description
This module is responsible for performing a first pre-processing of the dataset loaded by the user before the data goes through the AlchemyMLs next modules.

In general terms, it seeks to remove spaces to the left and right of a string, remove quotes from cells that are of type string, convert numerical data that comes in string format to numerical format, interpret and convert data that is of type date but comes in string format.

##### I/O
* Parameters:
    * _**token**_ (_str_): API Token. 
    * _**experiment_name**_ (_str_): Name of the experiment to be prepared.
    * _**download**_ (_bool_, optional): Optional boolean parameter to be set up if results needed to be downloaded.

#### method _encode_dataframe_
```python
def encode_dataframe(self, *args, **kwargs):
    str_meth_name = sys._getframe().f_code.co_name
    input_args = locals()['args']
    input_kwargs = locals()['kwargs']

    return general_call(str_meth_name, input_args, input_kwargs)    
```

##### Description
This is the sub-module in charge of coding the variables that indicate a category and are string type in numerical codes.

This operation is carried out because the automatic learning algorithms need to understand the nature of the data converted into numbers.

##### I/O
* Parameters:
    * _**token**_ (_str_): API Token. 
    * _**experiment_name**_ (_str_): Name of the experiment to be encoded.
    * _**download**_ (_bool_, optional): Optional boolean parameter to be set up if results needed to be downloaded.
    * _**target_col_name**_ (_str_, optional): Specifying the target column name.
    * _**prepare_dataset**_ (_bool_, optional): Optional boolean parameter that specifies if the dataset needs preparation or not.

#### method _drop_highly_correlated_components_
```python
def drop_highly_correlated_components(self, *args, **kwargs):
    str_meth_name = sys._getframe().f_code.co_name
    input_args = locals()['args']
    input_kwargs = locals()['kwargs']

    return general_call(str_meth_name, input_args, input_kwargs)    
```

##### Description
This is the method responsible for dropping highly correlated columns and duplicate rows.

The threshold to consider a column as highly correlated with another one is 0.9999.

Highly correlated columns can be both numerical and categorical columns.

##### I/O
* Parameters:
    * _**token**_ (_str_): API Token. 
    * _**experiment_name**_ (_str_): Name of the experiment on which process will take place.
    * _**download**_ (_bool_, optional): Optional boolean parameter to be set up if results needed to be downloaded.
    * _**target_col_name**_ (_str_, optional): Specifying the target column name.
    * _**prepare_dataset**_ (_bool_, optional): Optional boolean parameter that specifies if the dataset needs preparation or not.
    * _**component**_ (_str_, optional): Specifying whether you want to drop: "rows", "columns" or "both".

#### method _impute_inconsistencies_
```python
def impute_inconsistencies(self, *args, **kwargs):
    str_meth_name = sys._getframe().f_code.co_name
    input_args = locals()['args']
    input_kwargs = locals()['kwargs']

    return general_call(str_meth_name, input_args, input_kwargs)    
```

##### Description
This is the method responsible for iterating over each column of a dataset to find and correct inconsistencies. It is basically a submodule that searches for misspelled words, numbers or dates in an attempt to correct them.

You can choose between applying the operations to the entire dataset or just to the target column.

##### I/O
* Parameters:
    * _**token**_ (_str_): API Token. 
    * _**experiment_name**_ (_str_): Name of the experiment on which process will take place.
    * _**download**_ (_bool_, optional): Optional boolean parameter to be set up if results needed to be downloaded.
    * _**target_col_name**_ (_str_, optional): Specifying the target column name.
    * _**prepare_dataset**_ (_bool_, optional): Optional boolean parameter that specifies if the dataset needs preparation or not.
    * _**just_target**_ (_bool_, optional): Specifying whether you want to treat existing inconsistencies on the target or on the whole dataset (True/False). 

#### method _drop_invalid_columns_
```python
def drop_invalid_columns(self, *args, **kwargs):
    str_meth_name = sys._getframe().f_code.co_name
    input_args = locals()['args']
    input_kwargs = locals()['kwargs']

    return general_call(str_meth_name, input_args, input_kwargs)    
```

##### Description
Method to drop invalid columns in a experiment.

##### I/O
* Parameters:
    * _**token**_ (_str_): API Token. 
    * _**experiment_name**_ (_str_): Name of the experiment on which process will take place.
    * _**download**_ (_bool_, optional): Optional boolean parameter to be set up if results needed to be downloaded.
    * _**target_col_name**_ (_str_, optional): Specifying the target column name.
    * _**prepare_dataset**_ (_bool_, optional): Optional boolean parameter that specifies if the dataset needs preparation or not.

#### method _target_column_analysis_
```python
def target_column_analysis(self, *args, **kwargs):
    str_meth_name = sys._getframe().f_code.co_name
    input_args = locals()['args']
    input_kwargs = locals()['kwargs']

    return general_call(str_meth_name, input_args, input_kwargs)    
```

##### Description
This is the method responsible for telling the user wether the dataset is balanced or not by inspecting the target column.

##### I/O
* Parameters:
    * _**token**_ (_str_): API Token. 
    * _**experiment_name**_ (_str_): Name of the experiment on which process will take place.
    * _**target_col_name**_ (_str_, optional): Specifying the target column name.
    * _**prepare_dataset**_ (_bool_, optional): Optional boolean parameter that specifies if the dataset needs preparation or not.

#### method _balancing_dataframe_
```python
def balancing_dataframe(self, *args, **kwargs):
    str_meth_name = sys._getframe().f_code.co_name
    input_args = locals()['args']
    input_kwargs = locals()['kwargs']

    return general_call(str_meth_name, input_args, input_kwargs)    
```

##### Description
This is the method that deals with unbalanced classification datasets.

It detects unbalanced data, decides whether the data can be balanced (extreme cases are rejected), collects information on unbalance indicators and determines the method to be applied at the classification stage in order to adjust a balanced classifier.

##### I/O
* Parameters:
    * _**token**_ (_str_): API Token. 
    * _**experiment_name**_ (_str_): Name of the experiment on which process will take place.
    * _**download**_ (_bool_, optional): Optional boolean parameter to be set up if results needed to be downloaded.
    * _**target_col_name**_ (_str_, optional): Specifying the target column name.
    * _**prepare_dataset**_ (_bool_, optional): Optional boolean parameter that specifies if the dataset needs preparation or not.
    * _**auto_strategy**_ (_bool_, optional): Determines wether to force the generation of a balanced dataset or not. If auto_strategy is set to False, a balanced dataset will always be generated.

#### method _initial_exp_info_
```python
def initial_exp_info(self, *args, **kwargs):
    str_meth_name = sys._getframe().f_code.co_name
    input_args = locals()['args']
    input_kwargs = locals()['kwargs']

    return general_call(str_meth_name, input_args, input_kwargs)    
```

##### Description
This method returns initial information for the specified experiment.

##### I/O
* Parameters:
    * _**token**_ (_str_): API Token. 
    * _**experiment_name**_ (_str_): Name of the experiment on which process will take place.

#### method _impute_missing_values_
```python
def impute_missing_values(self, *args, **kwargs):
    str_meth_name = sys._getframe().f_code.co_name
    input_args = locals()['args']
    input_kwargs = locals()['kwargs']

    return general_call(str_meth_name, input_args, input_kwargs)    
```

##### Description
Method to use for missing values imputation.

##### I/O
* Parameters:
    * _**token**_ (_str_): API Token. 
    * _**experiment_name**_ (_str_): Name of the experiment on which process will take place.
    * _**download**_ (_bool_, optional): Optional boolean parameter to be set up if results needed to be downloaded.
    * _**target_col_name**_ (_str_, optional): Specifying the target column name.
    * _**prepare_dataset**_ (_bool_, optional): Optional boolean parameter that specifies if the dataset needs preparation or not.

#### method _merge_cols_into_dt_index_
```python
def merge_cols_into_dt_index(self, *args, **kwargs):
    str_meth_name = sys._getframe().f_code.co_name
    input_args = locals()['args']
    input_kwargs = locals()['kwargs']

    return general_call(str_meth_name, input_args, input_kwargs)    
```

##### Description
This is the method in charge of finding candidate columns with which to try to build a single datetime column.

##### I/O
* Parameters:
    * _**token**_ (_str_): API Token. 
    * _**experiment_name**_ (_str_): Name of the experiment on which process will take place.
    * _**download**_ (_bool_, optional): Optional boolean parameter to be set up if results needed to be downloaded.
    * _**target_col_name**_ (_str_, optional): Specifying the target column name.
    * _**prepare_dataset**_ (_bool_, optional): Optional boolean parameter that specifies if the dataset needs preparation or not.

#### method _detect_experiment_type_
```python
def detect_experiment_type(self, *args, **kwargs):
    str_meth_name = sys._getframe().f_code.co_name
    input_args = locals()['args']
    input_kwargs = locals()['kwargs']

    return general_call(str_meth_name, input_args, input_kwargs)    
```

##### Description
Method that gives the option to detect experiment type.

##### I/O
* Parameters:
    * _**token**_ (_str_): API Token. 
    * _**experiment_name**_ (_str_): Name of the experiment on which process will take place.
    * _**target_col_name**_ (_str_, optional): Specifying the target column name.
    * _**prepare_dataset**_ (_bool_, optional): Optional boolean parameter that specifies if the dataset needs preparation or not.
    * _**selected_option**_ (_str_, optional): For detect experiment type, the options available are: Regression, Classification, Time Series, Auto Detect.

#### method _build_model_
```python
def build_model(self, *args, **kwargs):
    str_meth_name = sys._getframe().f_code.co_name
    input_args = locals()['args']
    input_kwargs = locals()['kwargs']

    return general_call(str_meth_name, input_args, input_kwargs)    
```

##### Description
Method to build the model for a given experiment.

##### I/O
* Parameters:
    * _**token**_ (_str_): API Token. 
    * _**experiment_name**_ (_str_): Name of the experiment on which process will take place.
    * _**target_col_name**_ (_str_, optional): Specifying the target column name.
    * _**selected_option**_ (_str_, optional): For build the model the options available are: Regression, Classification, Time Series, Auto Detect.

#### method _operational_info_
```python
def operational_info(self, *args, **kwargs):
    str_meth_name = sys._getframe().f_code.co_name
    input_args = locals()['args']
    input_kwargs = locals()['kwargs']

    return general_call(str_meth_name, input_args, input_kwargs)    
```

##### Description
Through this method you can enter operational information related to each column: in this way you can specify what are the operating limits of a column and its tolerances. 

You can also indicate some values that you know and that occur within the values of the column so that the _impute_outliers_ module does not take them into account.

In addition, you can group the time-dependent columns by intervals (morning/evening/night) and you can detail whether the behavior of a column depends on the categories of another categorical column.

##### I/O
* Parameters:
    * _**token**_ (_str_): API Token. 
    * _**experiment_name**_ (_str_): Name of the experiment on which process will take place.
    * _**columns_info**_ (_str_/_list_/_dict_): Information on columns.

#### method _detect_outliers_
```python
def detect_outliers(self, *args, **kwargs):
    str_meth_name = sys._getframe().f_code.co_name
    input_args = locals()['args']
    input_kwargs = locals()['kwargs']

    return general_call(str_meth_name, input_args, input_kwargs)    
```

##### Description
This method gives the option of detect outliers. Different strategies are available, as univariate, bivariate, multivariate, complete.

##### I/O
* Parameters:
    * _**token**_ (_str_): API Token. 
    * _**experiment_name**_ (_str_): Name of the experiment to be used for outlier detection.
    * _**detection_strategy**_ (_str_, optional): Strategies available to employ for detection: univariate, bivariate, multivariate, complete.
    * _**columns**_ (_str_/_list_/_tuple_, optional): Defines to columns on which outliers detection is going to take place.
    * _**prepare_dataset**_ (_bool_, optional): Optional boolean parameter that specifies if the dataset needs preparation or not.

#### method _impute_outliers_
```python
def impute_outliers(self, *args, **kwargs):
    str_meth_name = sys._getframe().f_code.co_name
    input_args = locals()['args']
    input_kwargs = locals()['kwargs']

    return general_call(str_meth_name, input_args, input_kwargs)    
```

##### Description
Through this method outliers may be imputed using one of the available strategies.

##### I/O
* Parameters:
    * _**token**_ (_str_): API Token. 
    * _**experiment_name**_ (_str_): Experiment name on which outliers imputation is going to take place.
    * _**cols_to_impute**_ (_str_/_list_/_float_): Defines to columns on which outliers imputation is going to take place.
    * _**handling_strategy**_ (_str_/_dict_): Available options: 'auto', 'mean', 'median', 'mode', 'random_values', 'clipping', 'n_neighbors', 'quartile'.

