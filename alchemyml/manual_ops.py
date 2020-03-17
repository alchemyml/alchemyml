import sys
from request_handler import general_call

class actions():

    def list_preprocessed_dataframes(self, *args, **kwargs):
        '''

        '''
        str_meth_name = sys._getframe().f_code.co_name
        input_args = locals()['args']
        input_kwargs = locals()['kwargs']

        return general_call(str_meth_name, input_args, input_kwargs)    
    
    def download_dataframe(self, *args, **kwargs):
        '''
        Parameters:
        ----------
        experiment_name : str
            Name of experiment for which dataframe(s) needed to be download

        dataframe_name : str
            Dataframe name to be downloaded. Using the keyword all, all 
            dataframes available for the experiment will be downloaded in a 
            rar archive.

        Returns:
        ----------
        

        '''
        str_meth_name = sys._getframe().f_code.co_name
        input_args = locals()['args']
        input_kwargs = locals()['kwargs']

        return general_call(str_meth_name, input_args, input_kwargs)    
    
    def prepare_dataframe(self, *args, **kwargs):
        '''
        
        '''
        str_meth_name = sys._getframe().f_code.co_name
        input_args = locals()['args']
        input_kwargs = locals()['kwargs']

        return general_call(str_meth_name, input_args, input_kwargs)    
    
    def encode_dataframe(self, *args, **kwargs):
        '''
        
        '''
        str_meth_name = sys._getframe().f_code.co_name
        input_args = locals()['args']
        input_kwargs = locals()['kwargs']

        return general_call(str_meth_name, input_args, input_kwargs)    
    
    def drop_highly_correlated_components(self, *args, **kwargs):
        '''
        
        '''
        str_meth_name = sys._getframe().f_code.co_name
        input_args = locals()['args']
        input_kwargs = locals()['kwargs']

        return general_call(str_meth_name, input_args, input_kwargs)    
    
    def impute_inconsistencies(self, *args, **kwargs):
        '''
        
        '''
        str_meth_name = sys._getframe().f_code.co_name
        input_args = locals()['args']
        input_kwargs = locals()['kwargs']

        return general_call(str_meth_name, input_args, input_kwargs)    
        
    def drop_invalid_columns(self, *args, **kwargs):
        '''
        
        '''
        str_meth_name = sys._getframe().f_code.co_name
        input_args = locals()['args']
        input_kwargs = locals()['kwargs']

        return general_call(str_meth_name, input_args, input_kwargs)       

    def target_column_analysis(self, *args, **kwargs):
        '''
        
        '''
        str_meth_name = sys._getframe().f_code.co_name
        input_args = locals()['args']
        input_kwargs = locals()['kwargs']

        return general_call(str_meth_name, input_args, input_kwargs)    

    def balancing_dataframe(self, *args, **kwargs):
        str_meth_name = sys._getframe().f_code.co_name
        input_args = locals()['args']
        input_kwargs = locals()['kwargs']

        return general_call(str_meth_name, input_args, input_kwargs)    
        
    def initial_exp_info(self, *args, **kwargs):
        str_meth_name = sys._getframe().f_code.co_name
        input_args = locals()['args']
        input_kwargs = locals()['kwargs']

        return general_call(str_meth_name, input_args, input_kwargs)    
    
    def impute_missing_values(self, *args, **kwargs):
        str_meth_name = sys._getframe().f_code.co_name
        input_args = locals()['args']
        input_kwargs = locals()['kwargs']

        return general_call(str_meth_name, input_args, input_kwargs)    

    def merge_cols_into_dt_index(self, *args, **kwargs):
        str_meth_name = sys._getframe().f_code.co_name
        input_args = locals()['args']
        input_kwargs = locals()['kwargs']

        return general_call(str_meth_name, input_args, input_kwargs)    

    def detect_experiment_type(self, *args, **kwargs):
        str_meth_name = sys._getframe().f_code.co_name
        input_args = locals()['args']
        input_kwargs = locals()['kwargs']

        return general_call(str_meth_name, input_args, input_kwargs)    
    
    def build_model(self, *args, **kwargs):
        str_meth_name = sys._getframe().f_code.co_name
        input_args = locals()['args']
        input_kwargs = locals()['kwargs']

        return general_call(str_meth_name, input_args, input_kwargs)    

    def operational_info(self, *args, **kwargs):
        str_meth_name = sys._getframe().f_code.co_name
        input_args = locals()['args']
        input_kwargs = locals()['kwargs']

        return general_call(str_meth_name, input_args, input_kwargs)   
        
    def detect_outliers(self, *args, **kwargs):
        str_meth_name = sys._getframe().f_code.co_name
        input_args = locals()['args']
        input_kwargs = locals()['kwargs']

        return general_call(str_meth_name, input_args, input_kwargs)    

    def impute_outliers(self, *args, **kwargs):
        str_meth_name = sys._getframe().f_code.co_name
        input_args = locals()['args']
        input_kwargs = locals()['kwargs']

        return general_call(str_meth_name, input_args, input_kwargs) 

    def download_properties_df(self, *args, **kwargs):
        str_meth_name = sys._getframe().f_code.co_name
        input_args = locals()['args']
        input_kwargs = locals()['kwargs']

        return general_call(str_meth_name, input_args, input_kwargs)  

