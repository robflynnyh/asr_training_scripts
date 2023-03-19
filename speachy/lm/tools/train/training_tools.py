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

def loss_ce(logits, labels, ignore_index=-100, label_smoothing=0.0, reduction='mean'):
    return torch.nn.functional.cross_entropy(
            rearrange(logits, 'b n c -> b c n'), 
            labels, 
            ignore_index = ignore_index,
            label_smoothing = label_smoothing,
            reduction = reduction
        )


def add_bos(tokens, bos_token_id):
    #bos = torch.ones_like(tokens[:, :1]) * bos_token_id # bug if 1st dim is 0
    bos = torch.ones(tokens.shape[0], 1, dtype=tokens.dtype, device=tokens.device) * bos_token_id
    return torch.cat([bos, tokens], dim=1)


def add_eos(tokens, eos_id, token_lens):
    tokens[torch.arange(tokens.shape[0], device=tokens.device, dtype=torch.long), (token_lens - 1).to(torch.long)] = eos_id 
    return tokens

def mark_padding(targets, mask, pad_id):
    targets[~mask] = pad_id
    return targets

def token_lens_to_mask(token_lens, max_len=None):
    max_len = token_lens.max() if max_len is None else max_len
    mask = torch.arange(max_len, device=token_lens.device)[None, :] < token_lens[:, None]
    return mask

def batch_to_device(batch, device, return_all=False):
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.to(device)
    if isfalse(return_all):
        return batch['tokens'], batch['token_lens']
    else:
        return batch