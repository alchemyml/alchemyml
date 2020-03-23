# vers anterior AML2310.py
import json
import os
import sys

from ._request_handler import retry_session, general_call

host = 'https://alchemyml.com'
url_base = host + '/api'

class authentication():
    def get_api_token(self, username, password):
        url = url_base + '/token/'
        data = json.dumps({'username':username, 'password':password})
        session = retry_session(retries = 10)
        r = session.post(url, data, verify = False)

        if r.status_code == 200:
            tokenJSON = json.loads(r.text)
            return tokenJSON['access']
        else:
            msgJSON = json.loads(r.text)
            msg = msgJSON['message']
            return msg

class dataset():

    class_name = sys._getframe().f_code.co_name

    def upload(self, *args, **kwargs):  
        str_meth_name = self.class_name + '.' + sys._getframe().f_code.co_name
        input_args = locals()['args']
        input_kwargs = locals()['kwargs']

        return general_call(str_meth_name, input_args, input_kwargs)

    def get(self, *args, **kwargs): # PALABRA RESERVADA!
        str_meth_name = self.class_name + '.' + sys._getframe().f_code.co_name
        input_args = locals()['args']
        input_kwargs = locals()['kwargs']

        return general_call(str_meth_name, input_args, input_kwargs)
    
    def update(self, *args, **kwargs):
        str_meth_name = self.class_name + '.' + sys._getframe().f_code.co_name
        input_args = locals()['args']
        input_kwargs = locals()['kwargs']

        return general_call(str_meth_name, input_args, input_kwargs)

    def delete(self, *args, **kwargs):
        str_meth_name = self.class_name + '.' + sys._getframe().f_code.co_name
        input_args = locals()['args']
        input_kwargs = locals()['kwargs']

        return general_call(str_meth_name, input_args, input_kwargs)

    def statistical_descriptors(self, *args, **kwargs):
        str_meth_name = self.class_name + '.' + sys._getframe().f_code.co_name

        input_args = locals()['args']
        input_kwargs = locals()['kwargs']

        return general_call(str_meth_name, input_args, input_kwargs)

class experiment():

    class_name = sys._getframe().f_code.co_name

    def create(self, *args, **kwargs):
        str_meth_name = self.class_name + '.' + sys._getframe().f_code.co_name
        input_args = locals()['args']
        input_kwargs = locals()['kwargs']

        return general_call(str_meth_name, input_args, input_kwargs)

    def get(self, *args, **kwargs):
        str_meth_name = self.class_name + '.' + sys._getframe().f_code.co_name
        input_args = locals()['args']
        input_kwargs = locals()['kwargs']

        return general_call(str_meth_name, input_args, input_kwargs)

    def update(self, *args, **kwargs):
        str_meth_name = self.class_name + '.' + sys._getframe().f_code.co_name
        input_args = locals()['args']
        input_kwargs = locals()['kwargs']

        return general_call(str_meth_name, input_args, input_kwargs)

    def delete(self, *args, **kwargs):
        str_meth_name = self.class_name + '.' + sys._getframe().f_code.co_name
        input_args = locals()['args']
        input_kwargs = locals()['kwargs']

        return general_call(str_meth_name, input_args, input_kwargs)

    def statistical_descriptors(self, *args, **kwargs):
        str_meth_name = self.class_name + '.' + sys._getframe().f_code.co_name
        input_args = locals()['args']
        input_kwargs = locals()['kwargs']

        return general_call(str_meth_name, input_args, input_kwargs)

    def results(self, *args, **kwargs):
        str_meth_name = self.class_name + '.' + sys._getframe().f_code.co_name
        input_args = locals()['args']
        input_kwargs = locals()['kwargs']

        return general_call(str_meth_name, input_args, input_kwargs)

    def add_to_project(self, *args, **kwargs):
        str_meth_name = self.class_name + '.' + sys._getframe().f_code.co_name
        input_args = locals()['args']
        input_kwargs = locals()['kwargs']

        return general_call(str_meth_name, input_args, input_kwargs)

    def extract_from_project(self, *args, **kwargs):
        str_meth_name = self.class_name + '.' + sys._getframe().f_code.co_name
        input_args = locals()['args']
        input_kwargs = locals()['kwargs']

        return general_call(str_meth_name, input_args, input_kwargs)

    def send(self, *args, **kwargs):
        str_meth_name = self.class_name + '.' + sys._getframe().f_code.co_name
        input_args = locals()['args']
        input_kwargs = locals()['kwargs']

        return general_call(str_meth_name, input_args, input_kwargs)

class project():

    class_name = sys._getframe().f_code.co_name

    def create(self, *args, **kwargs):
        str_meth_name = self.class_name + '.' + sys._getframe().f_code.co_name
        input_args = locals()['args']
        input_kwargs = locals()['kwargs']

        return general_call(str_meth_name, input_args, input_kwargs)    

    def get(self, *args, **kwargs):
        str_meth_name = self.class_name + '.' + sys._getframe().f_code.co_name
        input_args = locals()['args']
        input_kwargs = locals()['kwargs']

        return general_call(str_meth_name, input_args, input_kwargs) 

    def update(self, *args, **kwargs):
        str_meth_name = self.class_name + '.' + sys._getframe().f_code.co_name
        input_args = locals()['args']
        input_kwargs = locals()['kwargs']

        return general_call(str_meth_name, input_args, input_kwargs)    

    def delete(self, *args, **kwargs):
        str_meth_name = self.class_name + '.' + sys._getframe().f_code.co_name
        input_args = locals()['args']
        input_kwargs = locals()['kwargs']

        return general_call(str_meth_name, input_args, input_kwargs)    
     
