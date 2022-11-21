import torch
from omegaconf.omegaconf import OmegaConf
from einops import rearrange, repeat
from tqdm import tqdm
import torch.nn as nn

from speachy.utils.helpers import (
    exists,
    istrue,
    isfalse
)

def loss_ce(logits, labels, ignore_index=-100):
    return torch.nn.functional.cross_entropy(rearrange(logits, 'b n c -> b c n'), labels, ignore_index=ignore_index)

def load_config(config:str):
    return OmegaConf.load(config)

def add_bos(tokens, bos_token_id):
    #bos = torch.ones_like(tokens[:, :1]) * bos_token_id # bug if 1st dim is 0
    bos = torch.ones(tokens.shape[0], 1, dtype=tokens.dtype, device=tokens.device) * bos_token_id
    return torch.cat([bos, tokens], dim=1)