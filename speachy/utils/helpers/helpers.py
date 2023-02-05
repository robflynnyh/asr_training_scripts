from typing import List, Dict, Any
from os import path
import subprocess, datetime, json, os, re

def isfalse(val:Any) -> bool:
    return val == False

def istrue(val:Any) -> bool:
    return val == True

def exists(val:Any) -> bool:
    return val is not None
    
def default(obj, default_val):
    return obj if exists(obj) else default_val

def save_json(obj:Dict, path:str):
    with open(path, 'w') as f:
        json.dump(obj, f)

def load_json(path:str) -> Dict:
    with open(path, 'r') as f:
        return json.load(f)

def run_cmd(cmd:str):
    print(f'Running {cmd}')
    subprocess.run(cmd, shell=True, check=True)

def get_date():
    return str(datetime.datetime.now()).replace(' ', '_').replace(':', '-').replace('.', '-')

def read_text(filename:str) -> List[str]:
    with open(filename, 'r') as f:
        txt = f.read().split('\n')
    return txt if txt[-1] != '' else txt[:-1]

def write_text(filename:str, text):
    if type(text) == str:
        with open(filename, 'w') as f:
            f.write(text)
    else:
        with open(filename, 'w') as f:
            for line in text:
                f.write(line)
                f.write('\n')

def load_envs(env_path='.env'):
    env_file = read_text(env_path)
    envs = {}
    for line in env_file:
        if line.startswith('#') or line.strip() == '':
            continue
        key, val = list(map(str.strip, line.split('='))) # remove whitespace
        envs[key] = val
    return envs

def request_env(env_name:str, env_path:str='.env'):
    envs = load_envs(env_path)
    assert env_name in envs, f'{env_name} not found in .env file'
    return envs[env_name]


def check_exists(path:str):
    assert os.path.exists(path), f'{path} does not exist'

def unpack_nested(nested:List[List[Any]]) -> List[Any]:
    return [item for sublist in nested for item in sublist]

def remove_multiple_spaces(text:str) -> str:
    return re.sub(' +', ' ', text)

def write_to_log(log_file, data):
    with open(log_file, 'a') as f:
        f.write(data)
        f.write('\n')

def check_exists(path:str):
    assert os.path.exists(path), f'{path} does not exist'