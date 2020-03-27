import json 

from ._version import __version__

__doc__ = '''
alchemyml - Automation of data lifecycle
========================================

**alchemyml** is a multi-environment tool for automating the data lifecycle 
and accelerating advanced data analysis processes. From data ingest and 
preparation, through pre-processing and up to the construction of predictive 
models, **alchemyml** provides the data scientist with aldata science applied 
artificial intelligence tools to optimize their work, time and resources. 
'''

class alchemyml():

    from ._CRUD_classes import dataset, experiment, project
    from ._manual_ops import actions

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

        url = 'https://alchemyml.com/api/token/'
        data = json.dumps({'username':username, 'password':password})
        session = retry_session(retries = 10)
        r = session.post(url, data)

        if r.status_code == 200:
            tokenJSON = json.loads(r.text)
            self.token = tokenJSON['access']
            return self.token
        else:
            msgJSON = json.loads(r.text)
            msg = msgJSON['message']
            return msg

    dataset = dataset
    experiment = experiment
    project = project
    actions = actions


