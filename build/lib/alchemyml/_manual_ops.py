import sys
from ._request_handler import general_call

class actions():

    def list_preprocessed_dataframes(self, *args, **kwargs):
        '''
        Method for listing available processed dataframes for the given experiment.

        Parameters:

        token (str): API Token. 
        experiment_name (str): Experiment name to which processed dataframes 
            will be returned.
        download (bool, optional): Optional boolean parameter to be set up 
            if results needed to be downloaded.
        '''
        str_meth_name = sys._getframe().f_code.co_name
        input_args = locals()['args']
        input_kwargs = locals()['kwargs']

        return general_call(str_meth_name, input_args, input_kwargs)    
    
    def download_dataframe(self, *args, **kwargs):
        '''
        As the name of the endpoint suggests, this method gives the option to 
            download the available processed dataframes for a given experiment.
        * If keyword all in dataframe_name, all available dataframes will be 
            downloaded.
        * If unknown the available processed dataframes, call first List 
            preprocessed dataframes.

        Parameters:

        token (str): API Token. 
        experiment_name (str): Name of experiment for which dataframe(s) 
            needed to be download.
        dataframe_name (str): Dataframe name to be downloaded. Using the 
            keyword all, all dataframes available for the experiment will be 
            downloaded in a rar archive.
        '''
        str_meth_name = sys._getframe().f_code.co_name
        input_args = locals()['args']
        input_kwargs = locals()['kwargs']

        return general_call(str_meth_name, input_args, input_kwargs)    
    
    def prepare_dataframe(self, *args, **kwargs):
        '''
        This module is responsible for performing a first pre-processing of 
            the dataset loaded by the user before the data goes through the 
            AlchemyMLs next modules.

        In general terms, it seeks to remove spaces to the left and right of 
            a string, remove quotes from cells that are of type string, 
            convert numerical data that comes in string format to numerical 
            format, interpret and convert data that is of type date but comes 
            in string format.

        Parameters:

        token (str): API Token. 
        experiment_name (str): Name of the experiment to be prepared.
        download (bool, optional): Optional boolean parameter to be set up 
            if results needed to be downloaded.
        '''
        str_meth_name = sys._getframe().f_code.co_name
        input_args = locals()['args']
        input_kwargs = locals()['kwargs']

        return general_call(str_meth_name, input_args, input_kwargs)    
    
    def encode_dataframe(self, *args, **kwargs):
        '''
        This is the sub-module in charge of coding the variables that indicate 
            a category and are string type in numerical codes.

        This operation is carried out because the automatic learning algorithms 
            need to understand the nature of the data converted into numbers.

        Parameters:

        token (str): API Token. 
        experiment_name (str): Name of the experiment to be encoded.
        download (bool, optional): Optional boolean parameter to be set up if 
            results needed to be downloaded.
        target_col_name (str, optional): Specifying the target column name.
        prepare_dataset (bool, optional): Optional boolean parameter that 
            specifies if the dataset needs preparation or not.
        '''
        str_meth_name = sys._getframe().f_code.co_name
        input_args = locals()['args']
        input_kwargs = locals()['kwargs']

        return general_call(str_meth_name, input_args, input_kwargs)    
    
    def drop_highly_correlated_components(self, *args, **kwargs):
        '''
        This is the method responsible for dropping highly correlated columns 
            and duplicate rows.

        The threshold to consider a column as highly correlated with another 
            one is 0.9999.

        Highly correlated columns can be both numerical and categorical columns.

        Parameters:

        token (str): API Token. 
        experiment_name (str): Name of the experiment on which process will take place.
        download (bool, optional): Optional boolean parameter to be set up if 
            results needed to be downloaded.
        target_col_name (str, optional): Specifying the target column name.
        prepare_dataset (bool, optional): Optional boolean parameter that 
            specifies if the dataset needs preparation or not.
        component (str, optional): Specifying whether you want to drop: 
            "rows", "columns" or "both".        
        '''
        str_meth_name = sys._getframe().f_code.co_name
        input_args = locals()['args']
        input_kwargs = locals()['kwargs']

        return general_call(str_meth_name, input_args, input_kwargs)    
    
    def impute_inconsistencies(self, *args, **kwargs):
        '''
        This is the method responsible for iterating over each column of 
            a dataset to find and correct inconsistencies. It is basically a 
            submodule that searches for misspelled words, numbers or dates in 
            an attempt to correct them.

        You can choose between applying the operations to the entire dataset 
            or just to the target column.

        Parameters:

        token (str): API Token. 
        experiment_name (str): Name of the experiment on which 
            process will take place.
        download (bool, optional): Optional boolean parameter to be set 
            up if results needed to be downloaded.
        target_col_name (str, optional): Specifying the target column name.
        prepare_dataset (bool, optional): Optional boolean parameter 
            that specifies if the dataset needs preparation or not.
        just_target (bool, optional): Specifying whether you want to treat 
            existing inconsistencies on the target or on the whole dataset (True/False). 
        '''
        str_meth_name = sys._getframe().f_code.co_name
        input_args = locals()['args']
        input_kwargs = locals()['kwargs']

        return general_call(str_meth_name, input_args, input_kwargs)    
        
    def drop_invalid_columns(self, *args, **kwargs):
        '''
        Method to drop invalid columns in a experiment.

        Parameters:

        token (str): API Token. 
        experiment_name (str): Name of the experiment on which process will 
            take place.
        download (bool, optional): Optional boolean parameter to be set up if 
            results needed to be downloaded.
        target_col_name (str, optional): Specifying the target column name.
        prepare_dataset (bool, optional): Optional boolean parameter that 
            specifies if the dataset needs preparation or not.
        '''
        str_meth_name = sys._getframe().f_code.co_name
        input_args = locals()['args']
        input_kwargs = locals()['kwargs']

        return general_call(str_meth_name, input_args, input_kwargs)       

    def target_column_analysis(self, *args, **kwargs):
        '''
        This is the method responsible for telling the user wether the 
            dataset is balanced or not by inspecting the target column.

        Parameters:

        token (str): API Token. 
        experiment_name (str): Name of the experiment on which process 
            will take place.
        target_col_name (str, optional): Specifying the target column name.
        prepare_dataset (bool, optional): Optional boolean parameter that 
            specifies if the dataset needs preparation or not.
        '''
        str_meth_name = sys._getframe().f_code.co_name
        input_args = locals()['args']
        input_kwargs = locals()['kwargs']

        return general_call(str_meth_name, input_args, input_kwargs)    

    def balancing_dataframe(self, *args, **kwargs):
        '''
        This is the method that deals with unbalanced classification datasets.

        It detects unbalanced data, decides whether the data can be balanced 
            (extreme cases are rejected), collects information on unbalance 
            indicators and determines the method to be applied at the 
            classification stage in order to adjust a balanced classifier.

        Parameters:

        token (str): API Token. 
        experiment_name (str): Name of the experiment on which process 
            will take place.
        download (bool, optional): Optional boolean parameter to be set up if 
            results needed to be downloaded.
        target_col_name (str, optional): Specifying the target column name.
        prepare_dataset (bool, optional): Optional boolean parameter that 
            specifies if the dataset needs preparation or not.
        auto_strategy (bool, optional): Determines wether to force the 
            generation of a balanced dataset or not. If auto_strategy is 
            set to False, a balanced dataset will always be generated.
        '''
        str_meth_name = sys._getframe().f_code.co_name
        input_args = locals()['args']
        input_kwargs = locals()['kwargs']

        return general_call(str_meth_name, input_args, input_kwargs)    
        
    def initial_exp_info(self, *args, **kwargs):
        '''
        This method returns initial information for the specified experiment.

        Parameters:

        token (str): API Token. 
        experiment_name (str): Name of the experiment on which process 
            will take place.
        '''
        str_meth_name = sys._getframe().f_code.co_name
        input_args = locals()['args']
        input_kwargs = locals()['kwargs']

        return general_call(str_meth_name, input_args, input_kwargs)    
    
    def impute_missing_values(self, *args, **kwargs):
        '''
        Method to use for missing values imputation.

        Parameters:

        token (str): API Token. 
        experiment_name (str): Name of the experiment on which process 
            will take place.
        download (bool, optional): Optional boolean parameter to be set up 
            if results needed to be downloaded.
        target_col_name (str, optional): Specifying the target column name.
        prepare_dataset (bool, optional): Optional boolean parameter that 
            specifies if the dataset needs preparation or not.
        '''
        str_meth_name = sys._getframe().f_code.co_name
        input_args = locals()['args']
        input_kwargs = locals()['kwargs']

        return general_call(str_meth_name, input_args, input_kwargs)    

    def merge_cols_into_dt_index(self, *args, **kwargs):
        '''
        This is the method in charge of finding candidate columns with which 
            to try to build a single datetime column.

        Parameters:

        token (str): API Token. 
        experiment_name (str): Name of the experiment on which process 
            will take place.
        download (bool, optional): Optional boolean parameter to be set up 
            if results needed to be downloaded.
        target_col_name (str, optional): Specifying the target column name.
        prepare_dataset (bool, optional): Optional boolean parameter that 
            specifies if the dataset needs preparation or not.
        '''
        str_meth_name = sys._getframe().f_code.co_name
        input_args = locals()['args']
        input_kwargs = locals()['kwargs']

        return general_call(str_meth_name, input_args, input_kwargs)    

    def detect_experiment_type(self, *args, **kwargs):
        '''
        Method that gives the option to detect experiment type.

        Parameters:

        token (str): API Token. 
        experiment_name (str): Name of the experiment on which process 
            will take place.
        target_col_name (str, optional): Specifying the target column name.
        prepare_dataset (bool, optional): Optional boolean parameter that 
            specifies if the dataset needs preparation or not.
        selected_option (str, optional): For detect experiment type, the 
            options available are: Regression, Classification, Time Series, Auto Detect.
        '''
        str_meth_name = sys._getframe().f_code.co_name
        input_args = locals()['args']
        input_kwargs = locals()['kwargs']

        return general_call(str_meth_name, input_args, input_kwargs)    
    
    def build_model(self, *args, **kwargs):
        '''
        Method to build the model for a given experiment.

        Parameters:

        token (str): API Token. 
        experiment_name (str): Name of the experiment on which process 
            will take place.
        target_col_name (str, optional): Specifying the target column name.
        selected_option (str, optional): For build the model the options 
            available are: Regression, Classification, Time Series, Auto Detect.
        '''
        str_meth_name = sys._getframe().f_code.co_name
        input_args = locals()['args']
        input_kwargs = locals()['kwargs']

        return general_call(str_meth_name, input_args, input_kwargs)    

    def operational_info(self, *args, **kwargs):
        '''
        Through this method you can enter operational information related to 
            each column: in this way you can specify what are the operating 
            limits of a column and its tolerances. 

        You can also indicate some values that you know and that occur within 
            the values of the column so that the impute_outliers module 
            does not take them into account.

        In addition, you can group the time-dependent columns by intervals 
            (morning/evening/night) and you can detail whether the behavior 
            of a column depends on the categories of another categorical column.

        Parameters:

        token (str): API Token. 
        experiment_name (str): Name of the experiment on which process 
            will take place.
        columns_info (str/list/dict): Information on columns.
        '''
        str_meth_name = sys._getframe().f_code.co_name
        input_args = locals()['args']
        input_kwargs = locals()['kwargs']

        return general_call(str_meth_name, input_args, input_kwargs)   
        
    def detect_outliers(self, *args, **kwargs):
        '''
        This method gives the option of detect outliers. Different strategies 
            are available, as univariate, bivariate, multivariate, complete.

        Parameters:

        token (str): API Token. 
        experiment_name (str): Name of the experiment to be used for 
            outlier detection.
        detection_strategy (str, optional): Strategies available to employ 
            for detection: univariate, bivariate, multivariate, complete.
        columns (str/list/tuple, optional): Defines to columns on which 
            outliers detection is going to take place.
        prepare_dataset (bool, optional): Optional boolean parameter that 
            specifies if the dataset needs preparation or not.
        '''
        str_meth_name = sys._getframe().f_code.co_name
        input_args = locals()['args']
        input_kwargs = locals()['kwargs']

        return general_call(str_meth_name, input_args, input_kwargs)    

    def impute_outliers(self, *args, **kwargs):
        '''
        Through this method outliers may be imputed using one of the available strategies.

        Parameters:

        token (str): API Token. 
        experiment_name (str): Experiment name on which outliers imputation 
            is going to take place.
        cols_to_impute (str/list/float): Defines to columns on which outliers 
            imputation is going to take place.
        handling_strategy (str/dict): Available options: 'auto', 'mean', 'median', 
            'mode', 'random_values', 'clipping', 'n_neighbors', 'quartile'.
        '''
        str_meth_name = sys._getframe().f_code.co_name
        input_args = locals()['args']
        input_kwargs = locals()['kwargs']

        return general_call(str_meth_name, input_args, input_kwargs) 

