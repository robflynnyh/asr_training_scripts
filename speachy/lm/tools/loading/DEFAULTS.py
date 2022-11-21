from omegaconf.omegaconf import OmegaConf
from speachy.utils.general import load_config
default_config = load_config('default_configs.yaml')

def get_model_defaults(field:str):
    modeldefaults = default_config['model']
    if field in modeldefaults:
        return modeldefaults[field]
    else:
        return None



