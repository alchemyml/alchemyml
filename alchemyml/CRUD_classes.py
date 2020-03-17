# vers anterior AML2310.py
import json
import os
import sys

from request_handler import retry_session, general_call

host = 'https://alchemyml.com' # 127.0.0.1:8011'
url_base = host + '/api'

# api_token = 'eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ0b2tlbl90eXBlIjoiYWNjZXNzIiwiZXhwIjoxNTgzOTk2Njg1LCJqdGkiOiJiZGUwNzgyOGU5OTI0OGY5YjA0MWU2NGE3OGU2NjU2NyIsInVzZXJfaWQiOjEyfQ.v3KJg0IEWRNUTNnOzaiFX7GZDQjNy1ja7GT0bRQljoQ'

class autentication():
    def get_api_token(self, username, password):
        global api_token # Prescindir de esto!
        url = url_base + '/token/'
        data = json.dumps({'username':username, 'password':password})
        session = retry_session(retries = 10)
        r = session.post(url, data, verify = False)

        if r.status_code == 200:
            tokenJSON = json.loads(r.text)
            api_token = tokenJSON['access']
            return r.status_code, tokenJSON['access']
        else:
            msgJSON = json.loads(r.text)
            msg = msgJSON['message']
            return r.status_code, msg

class dataset(): # datasets? 

    class_name = sys._getframe().f_code.co_name

    def upload(self, *args, **kwargs):  
        str_meth_name = self.class_name + '.' + sys._getframe().f_code.co_name
        input_args = locals()['args']
        input_kwargs = locals()['kwargs']

        return general_call(str_meth_name, input_args, input_kwargs)

    def view(self, *args, **kwargs): # PALABRA RESERVADA!
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

class experiment(): # experiments?

    class_name = sys._getframe().f_code.co_name

    def create(self, *args, **kwargs):
        str_meth_name = self.class_name + '.' + sys._getframe().f_code.co_name
        input_args = locals()['args']
        input_kwargs = locals()['kwargs']

        return general_call(str_meth_name, input_args, input_kwargs)

    def view(self, *args, **kwargs):
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

class project(): # projects?

    class_name = sys._getframe().f_code.co_name

    def create(self, *args, **kwargs):
        str_meth_name = self.class_name + '.' + sys._getframe().f_code.co_name
        input_args = locals()['args']
        input_kwargs = locals()['kwargs']

        return general_call(str_meth_name, input_args, input_kwargs)    

    def view(self, *args, **kwargs):
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
     
