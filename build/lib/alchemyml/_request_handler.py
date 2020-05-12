# vers anterior AML2310.py
import json
import os
import platform

from urllib3.util.retry import Retry
from urllib.request import urlretrieve
from requests import Session
from requests.adapters import HTTPAdapter
from pathlib import Path

from ._dict_urlData import dict_urlData

url_base = 'https://alchemyml.com/api'

def retry_session(retries, session = None, backoff_factor = 0.3, 
                  status_forcelist = (500, 502, 503, 504)):
    session = session or Session()
    retry = Retry(
        total = retries,
        read = retries,
        connect = retries,
        backoff_factor = backoff_factor,
        status_forcelist = status_forcelist,
    )
    adapter = HTTPAdapter(max_retries = retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    return session

def general_call(self, str_meth_name, input_args, input_kwargs):
    if str_meth_name in dict_urlData.keys():
        if hasattr(self, 'token'):
            api_token = self.token
        else:
            api_token = ''

        urlData = url_base + dict_urlData[str_meth_name]
        headers = {'Authorization': 'Bearer ' + api_token}
                
        if input_args:
            input_kwargs['args'] = input_args
            
        mi_data = input_kwargs
        
        if 'file_path' not in input_kwargs.keys():
            headers['Content-type'] = 'application/json'
            session = retry_session(retries = 10)
            api_request = session.post(urlData, headers = headers, json = mi_data)

        else:
            file_path = input_kwargs['file_path']
            del input_kwargs['file_path']
            
            if not os.path.exists(file_path):
                reply = {'success': False,
                          'status_code':'',
                          'message': 'Result Upload SCRIPT: File not found - NOT valid file path.',
                          'data': {'invalid input': 'File not found - NOT valid file path.'}}
                return reply

            if str_meth_name == 'dataset.upload':
                file_last_modif_date = int(round(os.stat(file_path).st_mtime))
                mi_data["last_modification_date"] = file_last_modif_date
                files = {'file_path': open(file_path, 'rb')}
                session = retry_session(retries=10)
                api_request = session.post(urlData, headers = headers, 
                                        files = files, data = mi_data)
            else:
                mi_data["file_path"] = file_path
                session = retry_session(retries=10)
                api_request = session.post(urlData, headers = headers, data = mi_data)
        
        session.close()
        res_json = json.loads(api_request.text)
        res_json_return = res_json.copy()
        res_json_return['status_code'] = api_request.status_code
        if api_request.status_code == 200:
            if 'data' in res_json:
                if isinstance(res_json['data'], dict ) and ('url' in res_json['data']):
                    f_name = str(res_json['data']['url']).split("/")[-1]
                    if str_meth_name == 'dataset.download':
                        if 'file_path' in locals():
                            file_path += '/' + f_name
                            urlretrieve(res_json['data']['url'], file_path)
                        else:
                            op_system = platform.system()
                            if op_system == 'Windows':
                                path_downloads = str(os.path.join(os.path.join(os.environ['USERPROFILE']), 'Downloads'))
                            else:
                                path_downloads = str(os.path.join(Path.home(), "Downloads"))
                            path_downloads += '/' + f_name
                            urlretrieve(res_json['data']['url'], path_downloads)

                    else:
                        session = retry_session(retries=10)
                        r = session.get(res_json['data']['url'])
                        session.close()
                        file = open(f_name, 'wb')
                        file.write(r.content)
                        file.close()

                    return 'File ' + f_name + ' successfully generated.'
                
                else:
                    return res_json_return

            else:
                return res_json_return

        elif api_request.status_code == 404:
            return '404: Not Found'

        elif api_request.status_code == 500:
            return '500: Internal Server Error'

        elif api_request.status_code == 503:
            return '503: Service Unavailable'        
    
        else:
            return res_json_return
    