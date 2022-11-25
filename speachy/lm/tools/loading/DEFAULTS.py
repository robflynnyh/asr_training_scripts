from omegaconf.omegaconf import OmegaConf
from speachy.utils.general import load_config
import os

def get_model_defaults(field:str):
    dirname = os.path.dirname(__file__)
    default_config = load_config(os.path.join(dirname, './default_configs.yaml'))
    modeldefaults = default_config['model']
    if field in modeldefaults:
        return modeldefaults[field]
    else:
        return None



