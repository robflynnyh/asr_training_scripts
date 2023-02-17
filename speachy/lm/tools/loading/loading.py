import torch
from omegaconf.omegaconf import OmegaConf
from einops import rearrange, repeat
from tqdm import tqdm
import torch.nn as nn
from .DEFAULTS import get_model_defaults
from speachy.lm import addons

def fetch_addons(model_config, tokenizer):
    add_on_modules = {}
    add_ons = model_config.get('add_ons', {})
    for add_on_module in add_ons.keys():
        if add_on_module == 'length_predictor':
            add_on_modules[add_on_module] = addons.LengthPredictor(dim=add_ons[add_on_module].get('dim', 256))
        if add_on_module == 'sep_token':
            sep_token = torch.randn(1, add_ons[add_on_module].get('dim', 256))
            torch.nn.init.normal_(sep_token, std=0.02)
            sep_token = nn.Parameter(sep_token)
            add_on_modules[add_on_module] = sep_token
        if add_on_module == 'next_sentence_pred':
            dim = add_ons[add_on_module].get('dim', 1500)
            dim = dim if dim != 'vocab' else tokenizer.vocab_size
            add_on_modules[add_on_module] = addons.NextSentenceTokenAdapter(dim=dim)
    return add_on_modules

def load_qknorm_transformer(config:OmegaConf, tokenizer, **kwargs):
    from speachy.lm.models.qknorm_attention import transformer_lm
    config_kwargs = config.get('kwargs', {})
    config_kwargs.update(kwargs)
    transformer = transformer_lm(
        dim = config.get('d_model', 256),
        vocab_size = tokenizer.vocab_size,
        depth = config.get('n_layers', 12),
        heads = config.get('n_heads', 8),
        dim_head = config.get('dim_head', 32),
        causal = config.get('causal', True),
        temperature = config.get('temperature', 15.5),
        intermediate_loss = config.get('intermediate_loss', True),
        self_conditioning = config.get('self_conditioning', False),
        dropout = config.get('dropout', 0.1),
        **config_kwargs
    )
    add_ons = fetch_addons(config, tokenizer)
    for add_on in add_ons.keys():
        setattr(transformer, add_on, add_ons[add_on]) # add the add-ons to the transformer

    return transformer

def autoload(config:OmegaConf, tokenizer, **kwargs):
    assert 'model' in config
    modelconfig = config['model']
    mtype = modelconfig.get('modeltype', 'qknorm')
    cfg = modelconfig[mtype]

    if 'qknorm' in mtype:
        return load_qknorm_transformer(cfg, tokenizer, **kwargs)

