import torch
from omegaconf.omegaconf import OmegaConf
from einops import rearrange, repeat
from tqdm import tqdm
import torch.nn as nn
from . import DEFAULTS
from DEFAULTS import get_model_defaults


def load_qknorm_transformer(config:OmegaConf, tokenizer):
    from speachy.lm.models.qknorm_attention import transformer_lm
    transformer = transformer_lm(
        dim = config.get('dim', get_model_defaults('dim')),
        vocab_size = tokenizer.vocab_size,
        depth = config.get('depth', get_model_defaults('depth')),
        heads = config.get('n_heads', get_model_defaults('n_heads')),
        dim_head = config.get('dim_head', get_model_defaults('dim_head')),
        causal = config.get('causal', True),
        temperature= config.get('temperature', 15.5),
        dropout= config.get('dropout', 0.0),
        **config.get('kwargs', {})
    )


def autoload(config:OmegaConf, tokenizer):
    assert 'model' in config
    modelconfig = config['model']
    mtype = modelconfig.get('modeltype', 'qknorm')

    if mtype == 'qknorm':
        return load_qknorm_transformer(modelconfig, tokenizer)

