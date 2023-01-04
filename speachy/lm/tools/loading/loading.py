import torch
from omegaconf.omegaconf import OmegaConf
from einops import rearrange, repeat
from tqdm import tqdm
import torch.nn as nn
from .DEFAULTS import get_model_defaults


def load_qknorm_transformer(config:OmegaConf, tokenizer):
    from speachy.lm.models.qknorm_attention import transformer_lm
    transformer = transformer_lm(
        dim = config.get('d_model', 256),
        vocab_size = tokenizer.vocab_size,
        depth = config.get('n_layers', 12),
        heads = config.get('n_heads', 8),
        dim_head = config.get('dim_head', 32),
        causal = config.get('causal', True),
        temperature= config.get('temperature', 15.5),
        dropout= config.get('dropout', 0.1),
        **config.get('kwargs', {})
    )
    return transformer


def autoload(config:OmegaConf, tokenizer):
    assert 'model' in config
    modelconfig = config['model']
    mtype = modelconfig.get('modeltype', 'qknorm')

    if 'qknorm' in mtype:
        cfg = modelconfig[mtype]
        return load_qknorm_transformer(cfg, tokenizer)

