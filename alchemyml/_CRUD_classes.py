import json
import os
import sys

from ._request_handler import retry_session, general_call

class authentication():
    
    def get_api_token(self, username, password):
        '''
        This method returns the necessary token to be used from now on for the 
        API requests. To be able to make use of the API before all it is 
        necessary to sign-up.

        Parameters:

        username (str): Username. 
        password (str): Password. 
        '''
        from ._request_handler import retry_session
        from warnings import warn, filterwarnings

        filterwarnings('always')
        msg = 'authentication() is deprecated and it will be removed in \
            alchemyml 0.1.31; use method get_api_token directly from class \
            alchemyml instead'
        warn(msg, DeprecationWarning)

        url = 'https://alchemyml.com/api/token/'
        data = json.dumps({'username':username, 'password':password})
        session = retry_session(retries = 10)
        r = session.post(url, data)

        if r.status_code == 200:
            tokenJSON = json.loads(r.text)
            return tokenJSON['access']

        else:
            msgJSON = json.loads(r.text)
            msg = msgJSON['message']
            return msg

class dataset():
    '''
    Class to manage operations on datasets: uploading, updating, getting a 
    list of available datasets in the workspace, deleting and computing 
    statistical descriptors. 
    '''

    class_name = sys._getframe().f_code.co_name

    def upload(self, *args, **kwargs):  
        '''
        Through the call to this method, you will be able to upload a dataset.

        We recommend you to consider the next points before uploading your dataset:
        * The accepted reading formats are: .xlsx, .xls, .csv, .json, .xml, .sql.
        * Files whose name contains two extensions will not be accepted. E.g.: 'Iris.xlsx.csv'.
        * Your data set should contain at least 50 observations.
        * The file must not exceed the size limit specified by the AlchemyML team.
        * Make sure that your data are not empty. Otherwise, this file will be rejected.

        Parameters:

        token (str): API Token. 
        file_path (str): The path where the dataset file is located. 
        dataset_name (str): Custom name for the dataset file. 
        description (str, optional): Optional parameter to specify description 
            if needed for the dataset. If no description is inputted, no 
            description is added to the dataset. 
        '''
        str_meth_name = self.class_name + '.' + sys._getframe().f_code.co_name
        input_args = locals()['args']
        input_kwargs = locals()['kwargs']

        return general_call(self, str_meth_name, input_args, input_kwargs)

    def get(self, *args, **kwargs):
        '''
        This method lists the datasets available in the workspace.
        * By setting the detail parameter to True or False, you can 
            control receiving the details of each uploaded dataset or simply a 
            list with the names of the datasets.
        * By setting the dataset_name parameter, you can control for which 
            datasets return the details.

        Parameters:

        token (str): API Token. 
        dataset_name (str/list, optional): Name or list of names of the 
            dataset(s) for which details will be returned.
        detail (bool, optional): Optional boolean parameter to return the 
            details for the specified dataset(s) (False/ True).
        '''
        str_meth_name = self.class_name + '.' + sys._getframe().f_code.co_name
        input_args = locals()['args']
        input_kwargs = locals()['kwargs']

        return general_call(self, str_meth_name, input_args, input_kwargs)
    
    def update(self, *args, **kwargs):
        '''
        This method gives the option to rename a dataset and/ or update the 
            datasets description. At least one of the two previous options 
            must be selected.

        Parameters:

        token (str): API Token. 
        dataset_name (str, optional): Name of the dataset to update.
        new_dataset_name (str, optional): New name of the specified dataset. 
            If no name is inputted, the dataset won't be renamed.
        new_description (str/list, optional): New description for the specified 
            dataset. If no description is inputted, the description is not 
            going to be updated.
        '''
        str_meth_name = self.class_name + '.' + sys._getframe().f_code.co_name
        input_args = locals()['args']
        input_kwargs = locals()['kwargs']

        return general_call(self, str_meth_name, input_args, input_kwargs)

    def delete(self, *args, **kwargs):
        '''
        Through the use of the Delete method you will be able to delete one, 
            several or all uploaded datasets. Note that if a dataset consists 
            of experiments associated with it, you must first remove the 
            experiments that have been created.

        AlchemyML is not responsible for any consequences that may be caused by 
            removing one, several or all datasets.

        Parameters:

        token (str): API Token. 
        dataset_name (str/list): Name or list of names of the datasets to be 
            removed from workspace. If _All_ used, then all datasets will 
            be removed. Datasets will be removed only if were not used in 
            any experiment.
        '''
        str_meth_name = self.class_name + '.' + sys._getframe().f_code.co_name
        input_args = locals()['args']
        input_kwargs = locals()['kwargs']

        return general_call(self, str_meth_name, input_args, input_kwargs)

    def statistical_descriptors(self, *args, **kwargs):
        '''
        This method returns the most relevant statistical descriptors for 
            each column of a dataset.

        Parameters:

        token (str): API Token
        dataset_name (str): Name of the dataset to return statistical descriptors.
        '''
        str_meth_name = self.class_name + '.' + sys._getframe().f_code.co_name

        input_args = locals()['args']
        input_kwargs = locals()['kwargs']

        return general_call(self, str_meth_name, input_args, input_kwargs)

class experiment():
    '''
    Class to manage operations on experiments: creating, updating, getting a 
    list of available experiments in the workspace, deleting, computing 
    statistical descriptors of intermediate dataframes in the workflow of the 
    experiment, displaying experiment's results (model metrics, logs...) and 
    sending the experiment to another user of AlchemyML. Moreover, operations 
    related with projects are also available: adding an experiment to a project 
    and removing an experiment from a project. 
    '''

    class_name = sys._getframe().f_code.co_name

    def create(self, *args, **kwargs):
        '''
        By default, an automatic experiment will be created.

        This option implies the execution of a sequence of steps that go from 
            the dataset intake to the construction of the predictive model, 
            including the pre-processing and cleaning of the data.

        If the experiment procedure is set to manual, then the user has the 
            possibility to control each phase of the experiment by running the 
            available modules in the desired order. 

        The possible operations that can be executed are those that appear in 
            the manual operations section.

        Parameters:

        token (str): API Token. 
        experiment_name (str): Name used for the creation the experiment. 
            This name is given by the user.
        description (str, optional): Optional parameter to specify 
            description if needed for the experiment. If no description is 
            inputted, no description is going to be added to the experiment.
        dataset_name (str): Name of the dataset used in the creation of experiment.
        target_column (str): Specifying the target column name.
        clients_choice (str): Type of experiment. Valid options: Regression, 
            Classification, Time Series, Auto Detect. 
        input_column_names (str, optional): Column names of the dataset to 
            use during the experiment. If no input column is introduced, all 
            will be taken by default.
        experiment_procedure (str, optional): Valid options are: auto or manual.
        '''
        str_meth_name = self.class_name + '.' + sys._getframe().f_code.co_name
        input_args = locals()['args']
        input_kwargs = locals()['kwargs']

        return general_call(self, str_meth_name, input_args, input_kwargs)

    def get(self, *args, **kwargs):
        '''
        Such as the datasets section, this method will let you know which 
            experiments you have in your workspace.
        * By setting the detail parameter to True or False you can 
            control receiving details of each experiment or simply get a list 
            with the names of the experiments.
        * By setting the experiment_name parameter, you control for which 
            experiments return the details (one or some).
        
        Parameters:

        token (str): API Token. 
        experiment_name (str/list, optional): The name or list of experiment 
            names to be listed.
        detail (bool, optional): Optional boolean parameter to return the 
            details for the specified experiment(s) (False/ True).
        '''
        str_meth_name = self.class_name + '.' + sys._getframe().f_code.co_name
        input_args = locals()['args']
        input_kwargs = locals()['kwargs']

        return general_call(self, str_meth_name, input_args, input_kwargs)

    def update(self, *args, **kwargs):
        '''
        This method gives the option to rename an experiment and/ or update 
            the experiments description. At least one of the two previous 
            options must be selected.

        Parameters:

        token (str): API Token. 
        experiment_name (str): Name of the experiment to update.
        new_experiment_name (str, optional): New name of the specified 
            experiment. If no name is inputted, the experiment is not going 
            to be renamed.
        new_description (str/list, optional): New description for the 
            specified experiment. If no description is inputted, the 
            description is not going to be updated.
        '''
        str_meth_name = self.class_name + '.' + sys._getframe().f_code.co_name
        input_args = locals()['args']
        input_kwargs = locals()['kwargs']

        return general_call(self, str_meth_name, input_args, input_kwargs)

    def delete(self, *args, **kwargs):
        '''
        Through the use of the endpoint Delete you will be able to delete one, 
            several or all the experiments created.

        AlchemyML is not responsible for any consequences that may be caused 
            by removing one, several or all experiments.

        Parameters:

        token (str): API Token. 
        experiment_name (str/list): Name or list of experiment names to be 
            deleted. If All used, then all experiments will be removed.
        '''
        str_meth_name = self.class_name + '.' + sys._getframe().f_code.co_name
        input_args = locals()['args']
        input_kwargs = locals()['kwargs']

        return general_call(self, str_meth_name, input_args, input_kwargs)

    def statistical_descriptors(self, *args, **kwargs):
        '''
        This method returns the most relevant statistical descriptors for each 
            column of the preprocessed dataset used in the experiment creation.

        Parameters:

        token (str): API Token. 
        experiment_name (str): Name of the experiment to return statistical descriptors.
        dataset_name (str): Name of the dataset used in the experiment creation.
        '''
        str_meth_name = self.class_name + '.' + sys._getframe().f_code.co_name
        input_args = locals()['args']
        input_kwargs = locals()['kwargs']

        return general_call(self, str_meth_name, input_args, input_kwargs)

    def results(self, *args, **kwargs):
        '''
        The creation of an experiment (previous method CREATE) returns the 
            results of that experiment. This method gives the option to 
            retrieve the previous results whenever these are needed.

        The results are delivered in a JSON structure consisting of two keys: 
        * "log" contains the information related to the decisions that 
            AlchemyML has taken throughout the creation of the experiment, 
            until finishing the construction of the predictive model.
        * On the other hand, "model_metrics" will include the analytical 
            information of these results: metrics obtained, relevant variables, 
            type of experiment, etc.

        Parameters:

        token (str): API Token. 
        experiment_name (str): Name of the experiment to return the results.
        '''
        str_meth_name = self.class_name + '.' + sys._getframe().f_code.co_name
        input_args = locals()['args']
        input_kwargs = locals()['kwargs']

        return general_call(self, str_meth_name, input_args, input_kwargs)

    def add_to_project(self, *args, **kwargs):
        '''
        This method gives the possibility to include an experiment or various 
            into a specified project.

        Projects are the way to order and group different experiments that are 
            included within a general topic. For example, you could create a 
            project under the theme of Smart Cities that includes experiments 
            related to this topic.

        Parameters:

        token (str): API Token. 
        associated_experiments (str/list): Name or list of experiment names 
            to be included into a specified project.
        project_name (str): The name of the project in which experiment(s) 
            will be included.
        description (str, optional): Optional parameter to specify description 
            if needed for the project. If no description is inputted, 
            no description is going to be added to the experiment.
        '''
        str_meth_name = self.class_name + '.' + sys._getframe().f_code.co_name
        input_args = locals()['args']
        input_kwargs = locals()['kwargs']

        return general_call(self, str_meth_name, input_args, input_kwargs)

    def extract_from_project(self, *args, **kwargs):
        '''
        Given a project this method gives the possibility to extract specified 
            experiments from it.

        Parameters:

        token (str): API Token. 
        experiment_name (str/list): Name or list of experiment names that 
            are desired to be extracted from a given project.
        project_name (str): The project from which will be extracted the 
            specified experiments.
        '''
        str_meth_name = self.class_name + '.' + sys._getframe().f_code.co_name
        input_args = locals()['args']
        input_kwargs = locals()['kwargs']

        return general_call(self, str_meth_name, input_args, input_kwargs)

    def send(self, *args, **kwargs):
        '''
        This endpoint gives the possibility to send one or various experiments 
            to another registered user.

        If the user exists, a confirmation email will be sent. When the recipient 
            confirms that he wants to receive an experiment from another user, 
            an exact copy of the experiment will appear within his/her 
            experiments section and will also be visible through the Workspace.

        Parameters:

        token (str): API Token. 
        destination_email (str): The receivers email address.
        experiment_name (str/list): The name or list of experiment names to be sent.
        '''
        str_meth_name = self.class_name + '.' + sys._getframe().f_code.co_name
        input_args = locals()['args']
        input_kwargs = locals()['kwargs']

        return general_call(self, str_meth_name, input_args, input_kwargs)

class project():
    '''
    Class to manage operations on projects: creating, updating, getting a 
    list of available projects in the workspace and deleting. 
    '''

    class_name = sys._getframe().f_code.co_name

    def create(self, *args, **kwargs):
        '''
        This method creates a new project.

        Projects are the way to order and group different experiments that 
            are included within a general topic. For example, you could 
            create a project under the theme of Smart Cities that includes 
            experiments related to this topic.

        Parameters:

        token (str): API Token.  
        project_name (str): Name of the project.
        description (str, optional): Optional parameter to specify description 
            if needed for the project. If no description is inputted, no 
            description is going to be added to the project.
        associated_experiments (str/list, optional): Name or list of experiment 
            names to be added to the project. If no experiments are inputted, 
            an empty project is going to be created.
        '''
        str_meth_name = self.class_name + '.' + sys._getframe().f_code.co_name
        input_args = locals()['args']
        input_kwargs = locals()['kwargs']

        return general_call(self, str_meth_name, input_args, input_kwargs)    

    def get(self, *args, **kwargs):
        '''
        Such as the datasets section, this method will let you know which 
            projects you have in your workspace.
        * By setting the detail parameter to True or False you can control 
            receiving details of each project or simply get a list with the 
            names of the projects.
        * By setting the project_name parameter, you control for which projects 
            return the details (one or some).

        Parameters:

        token (str): API Token.  
        project_name (str/list, optional): Name or list of names of the project(s).
        detail (bool, optional): Optional boolean parameter to return the 
            details for the specified project(s) (False/ True).
        '''
        str_meth_name = self.class_name + '.' + sys._getframe().f_code.co_name
        input_args = locals()['args']
        input_kwargs = locals()['kwargs']

        return general_call(self, str_meth_name, input_args, input_kwargs) 

    def update(self, *args, **kwargs):
        '''
        This method gives the option to rename a project and/ or update the 
            projects description. At least one of the two previous options 
            must be selected.

        Parameters:

        token (str): API Token.  
        project_name (str): Name of the project to be updated.
        new_project_name (str, optional): New name of the specified project. 
            If no name is inputted, the project is not going to be renamed.
        new_description (str/list, optional): New description for the specified 
            project. If no description is inputted, the description is not 
            going to be updated.
        '''
        str_meth_name = self.class_name + '.' + sys._getframe().f_code.co_name
        input_args = locals()['args']
        input_kwargs = locals()['kwargs']

        return general_call(self, str_meth_name, input_args, input_kwargs)    

    def delete(self, *args, **kwargs):
        '''
        Through the use of the method Delete you will be able to delete one, 
            several or all the projects created.

        AlchemyML is not responsible for any consequences that may be caused 
            by removing one, several or all projects.

        Parameters:

        token (str): API Token.  
        project_name (str/list): Name or list of names of the projects to be 
            deleted. If All used then, all projects will be removed.
        '''
        str_meth_name = self.class_name + '.' + sys._getframe().f_code.co_name
        input_args = locals()['args']
        input_kwargs = locals()['kwargs']

        return general_call(self, str_meth_name, input_args, input_kwargs)    
     
