import torch, torch.nn as nn, torch.nn.functional as F
from speachy.lm.tools.train import add_eos, token_lens_to_mask, mark_padding
import numpy as np
from einops import rearrange, repeat
from torch import einsum
from torch.utils.checkpoint import checkpoint # # gradient/activation checkpointing
from functools import partial
import string
from math import ceil
from einops import einsum as einsumops
from vector_quantize_pytorch import RandomProjectionQuantizer
from vector_quantize_pytorch.vector_quantize_pytorch_m import VectorQuantize
from typing import Optional, Tuple, List, Dict, Union, Callable
import torch_scatter
import math
from batchrenorm import BatchRenorm1d

def exists(val):
    return val is not None

# token shifting
# lucidrains implementation: https://github.com/lucidrains/x-transformers/blob/main/x_transformers/x_transformers.py
# BlinkDL idea from RWKV-LM https://github.com/BlinkDL/RWKV-LM
def shift(t, amount, mask = None):
    if amount == 0:
        return t
    else:
        amount = min(amount, t.shape[1])

    if exists(mask):
        t = t.masked_fill(~mask[..., None], 0.)

    return F.pad(t, (0, 0, amount, -amount), value = 0.)

class ShiftTokens(nn.Module):
    '''from Phil Wang's x-transformers library'''
    def __init__(self, shifts, fn):
        super().__init__()
        self.fn = fn
        self.shifts = tuple(shifts)

    def forward(self, x, **kwargs):
        mask = kwargs.get('mask', None)
        shifts = self.shifts
        segments = len(shifts)
        feats_per_shift = x.shape[-1] // segments
        splitted = x.split(feats_per_shift, dim = -1)
        segments_to_shift, rest = splitted[:segments], splitted[segments:]
        segments_to_shift = list(map(lambda args: shift(*args, mask = mask), zip(segments_to_shift, shifts)))
        x = torch.cat((*segments_to_shift, *rest), dim = -1)
        return self.fn(x, **kwargs)

def ff(dim, mult=4, dropout=0.1):
    return nn.Sequential(
        GLU(dim, dim * mult, nn.SiLU()),
        nn.Dropout(dropout),
        nn.Linear(dim * mult, dim)
    )

class DynamicPositionBias(nn.Module):
    '''Adapted from Phil Wang's x-transformers library'''
    def __init__(self, dim, *, heads, depth, log_distance = False, norm = False, activation=nn.ReLU):
        super().__init__()
        assert depth >= 1, 'depth for dynamic position bias MLP must be greater or equal to 1'
        self.log_distance = log_distance

        self.mlp = nn.ModuleList([])

        self.mlp.append(nn.Sequential(
            nn.Linear(1, dim),
            nn.LayerNorm(dim) if norm else nn.Identity(),
            activation()
        ))

        for _ in range(depth - 1):
            self.mlp.append(nn.Sequential(
                nn.Linear(dim, dim),
                nn.LayerNorm(dim) if norm else nn.Identity(),
                activation()
            ))

        self.mlp.append(nn.Linear(dim, heads))


    def forward(self, pos, indices, device, dtype):
        pos = pos.to(device=device, dtype=dtype)
        
        if self.log_distance:
            pos = torch.sign(pos) * torch.log(pos.abs() + 1)  # log of distance is sign(rel_pos) * log(abs(rel_pos) + 1)

        for layer in self.mlp:
            pos = layer(pos) 
      
        bias = pos[indices]
        #print(bias.shape)
        bias = rearrange(bias, 'b i j h -> b h i j')
        return bias

class ScaledSinuEmbedding(nn.Module):
    '''taken From Phil Wang's x-transformers library'''
    def __init__(self, dim):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1,))
        inv_freq = 1. / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, x):
        n, device = x.shape[1], x.device
        t = torch.arange(n, device = device).type_as(self.inv_freq)
        sinu = einsum('i , j -> i j', t, self.inv_freq)
        emb = torch.cat((sinu.sin(), sinu.cos()), dim = -1)
        return emb * self.scale

class ReLUSquared(nn.Module):
    def forward(self, x):
        return torch.pow(F.relu(x), 2)

def l2norm(t, groups = 1, dim = -1):
    if groups == 1:
        return F.normalize(t, p = 2, dim = dim)
    t = rearrange(t, '... (g d) -> ... g d', g = groups)
    t = F.normalize(t, p = 2, dim = dim)
    return rearrange(t, '... g d -> ... (g d)')



class Attention(nn.Module):
    def __init__(
        self,
        n_feats,
        head_dim,
        n_heads,
        dropout=0.1,
        bias=False,
        return_attention=False,
        causal=False,
        activation='softmax',
        **kwargs
    ):
        super().__init__()
        assert activation in ['relusq', 'softmax']
        self.shared_kv = kwargs.get('shared_kv', False)
        
        self.talking_heads = kwargs.get('talking_heads', 'none') # 'none', 'pre', 'both', 'post' 

        self.n_feats, self.head_dim, self.n_heads = n_feats, head_dim, n_heads
        self.dropout = nn.Dropout(dropout)
        self.bias = bias
        self.return_attention = return_attention
        self.causal = causal

        if self.talking_heads == 'pre' or self.talking_heads == 'both':
            self._head_proj = nn.Conv2d(n_heads, n_heads, (1, 1))
        if self.talking_heads == 'post' or self.talking_heads == 'both':
            self._head_proj_post = nn.Conv2d(n_heads, n_heads, (1, 1))
            

        self.activation = ReLUSquared() if activation == 'relusq' else nn.Softmax(dim=-1)

        if not self.shared_kv:
            self.qkv_proj = nn.Linear(n_feats, 3 * n_heads * head_dim, bias=bias)
            self.qkv = lambda x: rearrange(self.qkv_proj(x), "b n (h d qkv) -> qkv b h n d", qkv=3, h=n_heads, d=head_dim)
        else:
            self.q_proj, self.kv_proj = [nn.Linear(n_feats, el, bias=bias) for el in [n_heads * head_dim, 2 * head_dim]]
            map_q, map_kv = lambda q: rearrange(q, 'b n (h d) -> b h n d', h=n_heads), lambda kv: rearrange(kv, 'b n (kv d) -> kv b () n d', kv=2, d=head_dim)
            self.qkv = lambda x: (map_q(self.q_proj(x)), *map_kv(self.kv_proj(x)))

        self.out_proj = nn.Linear(n_heads * head_dim, n_feats, bias=bias)
    
    def head_proj(self, dots, mode='pre'):
        if mode == 'pre' and (self.talking_heads == 'pre' or self.talking_heads == 'both'):
            dots = self._head_proj(dots)
        if mode == 'post' and (self.talking_heads == 'post' or self.talking_heads == 'both'):
            dots = self._head_proj_post(dots)
        return dots      
  

    def attend(self, query, key, value, attn_mask, pos_bias):        
        dots = einsum('bhid,bhjd->bhij', query, key) * self.head_dim ** -0.5
        dots = self.head_proj(dots, mode='pre')

        dots += pos_bias
        dots.masked_fill_(attn_mask, -torch.finfo(dots.dtype).max)

        attn = self.activation(dots)
        attn = self.head_proj(attn, mode='post')
     
        attn = self.dropout(attn)
        return einsum("bhij,bhjd->bhid", attn, value)

    @staticmethod
    def attach_cache(kv, cache, cache_indices):
        kv = torch.stack(kv, dim=0)
        if cache is None:
            return kv
        zero_vector = torch.zeros_like(kv[:, :, :, :1, :])
        kv_w_cache = torch.cat([cache, kv, zero_vector], dim=-2)
        kv_w_cache = torch.gather(kv_w_cache, dim=-2, index=cache_indices) # we do this to remove unnecessary padding
        return kv_w_cache

    def forward(self, x, pos_bias, mask, cache=None, cache_indices=None):
        B, N, C, H, D = *x.shape, self.n_heads, self.head_dim
    
        q, k, v  = self.qkv(x)
        kv = self.attach_cache([k, v], cache, cache_indices)
        k, v = kv

        out = self.attend(q, k, v, mask, pos_bias)

        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.out_proj(out)
        return out, kv

class PreNorm(nn.Module):
    def __init__(self, dim, fn, elementwise_affine=True):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=elementwise_affine)
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(self.norm(x), *args, **kwargs)



class PreBatchReNorm(nn.Module):
    def __init__(self, dim, fn, affine=True):
        super().__init__()
        self.norm = BatchRenorm1d(dim, affine=affine)
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        x = rearrange(x, 'b n d -> b d n')
        x = self.norm(x)
        x = rearrange(x, 'b d n -> b n d')
        return self.fn(x, *args, **kwargs)


class NoGrad(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    @torch.no_grad()
    def forward(self, *args, **kwargs):
        return self.fn(*args, **kwargs)

class GLU(nn.Module):
    def __init__(self, dim_in, dim_out, activation):
        super().__init__()
        self.act = activation
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim = -1)
        return x * self.act(gate)

def orthogonal_loss_fn(t):
    # eq (2) from https://arxiv.org/abs/2112.00384
    h, n = t.shape[:2]
    normed_codes = l2norm(t)
    cosine_sim = einsum('h i d, h j d -> h i j', normed_codes, normed_codes)
    return (cosine_sim ** 2).sum() / (h * n ** 2) - (1 / n)



class orthoginal_loss(nn.Module): # same as above but as a module
    def __init__(self, weight=1.0):
        super().__init__()
        self.loss_fn = orthogonal_loss_fn_padded
        self.weight = weight

    def forward(self, t, mask=None):
        return self.loss_fn(t, mask) * self.weight

class Halfer(nn.Module): # uses conv instead of avg_pool1d
    def __init__(self, dim, exp_f=4):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim*exp_f*2, kernel_size=2, stride=2, padding=0, bias=True)
        self.act = nn.SiLU()
        self.empty_vec = torch.zeros(1,1,dim)
        self.ff = nn.Linear(dim*exp_f, dim)

    def forward(self, x, length):
        if x.shape[1] % 2 == 1:
            x = torch.cat([x, self.empty_vec.to(x.device).expand(x.shape[0],1,-1)], dim=1)
            length += 1
        x = self.conv(x.transpose(1,2)).transpose(1,2)
        a, b = x.chunk(2, dim=-1)
        x = a * self.act(b)
        x = self.ff(x)
        length = (length + 1).div(2).floor().long()
        return x, length

class InverseHalfer(nn.Module): # opposite of Halfer
    def __init__(self, dim, exp_f=2):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim*exp_f, kernel_size=2, stride=2, padding=0, bias=True)
        self.act = nn.SiLU()
        self.ff = nn.Linear(dim*exp_f, dim)

    def forward(self, x, length):
        x = self.conv(x.transpose(1,2)).transpose(1,2)
        x = self.ff(self.act(x))
        length = length.mul(2)
        return x, length

class HalferBlock(nn.Module):
    def __init__(self, dim, exp_f=2):
        super().__init__()
        self.halfer = PreNorm(dim=dim, fn=Halfer(dim, exp_f=exp_f))
        self.inverse_halfer = PreNorm(dim=dim, fn=InverseHalfer(dim, exp_f=exp_f))
        self.loss = nn.MSELoss(reduction='none')

    def forward(self, x, length, mask=None):
        halved_x, halved_length = self.halfer(x, length)
        def recon_loss_fn(quantized_x):
            restored_x, _ = self.inverse_halfer(quantized_x, halved_length)
            restored_x = restored_x[:, :x.shape[1], :] # trim to original length
            loss = torch.tensor([0.], device=x.device)
            if self.training:
                loss = self.loss(restored_x, x)
                if mask is not None: # mask out padding
                    loss.masked_fill_(mask[..., None], 0.)
                loss = loss.mean()
            return loss, restored_x
        return halved_x, halved_length, recon_loss_fn



class PredictionLayer(nn.Module):
    def __init__(self, dim, n_classes):
        super().__init__()
        self.proj = PreNorm(dim, nn.Linear(dim, n_classes))
    def forward(self, x):
        return self.proj(x)



def grab_last_token(x, length):
    return x[torch.arange(x.shape[0]), length-1][:,None,:]

class AttentionFF(nn.Module):
    def __init__(
            self, 
            dim, 
            n_heads, 
            head_dim, 
            dropout=0., 
            ff_mult=4,
            causal=True,
            **kwargs
        ):
        super().__init__()
        self.attention = Attention(dim, n_heads=n_heads, head_dim=head_dim, causal=causal,dropout=dropout,**kwargs)
        self.ff = ff(dim, mult=ff_mult, dropout=0.1)
        self.ff, self.attention = map(PreNorm, (dim, dim), (self.ff, self.attention))

    def forward(self, x, pos_bias, attn_mask, cache=None, cache_indices=None):
        x_out, kv = self.attention(x, pos_bias, attn_mask, cache, cache_indices)
        x = x_out + x
        x = self.ff(x) + x
        return x, kv


class AttentionFFstack(nn.Module):
    def __init__(
        self,
        dim=256, 
        total_depth=3,
        n_heads=8, 
        head_dim=32, 
        dropout=0., 
        ff_mult=4,
        causal=True,
        **kwargs
    ):   
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(total_depth):
            self.layers.append(AttentionFF(dim, n_heads, head_dim, dropout, ff_mult, causal, **kwargs))

    def forward(self, x, pos_bias, attn_mask, cache=None, cache_indices=None):
        cached_kvs = []
        for i, layer in enumerate(self.layers):
            x, kv = layer(x, pos_bias, attn_mask, cache[i] if cache is not None else None, cache_indices)
            cached_kvs.append(kv[None])
        return x, cached_kvs

def map_to_sequence(int_list):
    unique_ints = torch.unique(int_list)
    max_int = unique_ints.max()
    seq = torch.zeros(max_int + 1, dtype=torch.int64)
    seq[unique_ints] = torch.arange(len(unique_ints))
    return seq[int_list]


class EinopsFn(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x, y, pattern):
        return einsumops(x, y, pattern)

def stable_weighted_softmax(logits, weights):
    ## logits: B, H, N, V # weights: 1 H 1 V ##
    stable_logits = logits - logits.max(dim=-1, keepdim=True)[0] # subtract max to make logits more stable
    weights = weights.clamp(min=torch.finfo(weights.dtype).eps)
    numerators = torch.exp(stable_logits) * weights
    denominators = numerators.sum(dim=-1, keepdim=True) 
    return numerators / (denominators + torch.finfo(denominators.dtype).eps)
    
class StableWeightedSoftmax(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, logits, weights):
        return stable_weighted_softmax(logits, weights)

class transformer(nn.Module):
    def __init__(
            self, 
            dim, 
            depth, 
            heads, 
            dim_head, 
            causal=True,
            base_vocab_size=29,
            dropout = 0.1,
            **kwargs
        ):
        super().__init__()
    

        ff_mult = kwargs.get('ff_mult', 4)
        self.checkpoint_every_n = kwargs.get('checkpoint_every_n', 0)
        self.base_vocab = base_vocab_size
        self.max_vocab = kwargs.get('max_vocab', 10000)
        commitment_weight = kwargs.get('commitment_weight', 1.0)
        self.inner_depth = kwargs.get('inner_depth', 2)

        self.embedding = nn.Embedding(base_vocab_size, dim)

        self.causal = causal

        self.depth = depth
        self.positional_bias = DynamicPositionBias(
            dim = dim // 4,
            heads = heads,
            depth = 2,
            log_distance = False,
            norm = False
        )

        self.halfers = nn.ModuleList([])

        self.codebook_dim = 16
        self.codebook_heads = 12
        self.codebook_vocab = 1024
        
        self.vqs = nn.ModuleList([])
            
        self.prediction_layers = nn.ModuleList([])
        self.pred_scalers = torch.nn.Parameter(torch.ones((depth-1, 1))*5.0, requires_grad=True)

        self.attn_ffs = nn.ModuleList([])
        for lth in range(depth):
            self.attn_ffs.append(AttentionFFstack(
                total_depth=self.inner_depth,
                dim=dim,
                n_heads=heads,
                head_dim=dim_head,
                causal=causal,
                dropout=dropout,
                **kwargs
            ))

            if lth == 0:
                self.prediction_layers.append(PredictionLayer(dim, base_vocab_size))
            else:
                self.prediction_layers.append(PredictionLayer(dim, self.codebook_dim*self.codebook_heads*self.codebook_heads))

            if lth < depth - 1:
                self.vqs.append(
                    RandomProjectionQuantizer(
                        dim = dim,
                        codebook_dim = self.codebook_dim,
                        codebook_size = self.codebook_vocab,
                        num_codebooks = self.codebook_heads,  
                        norm = True  
                    )
                )
                #self.vqs[-1].requires_grad_(False)'''
                self.halfers.append(PreBatchReNorm(dim=dim, fn=Halfer(dim, exp_f=8), affine = True))
               

                self.halfers[-1].fn.requires_grad_(False)

        self.einopsfn = EinopsFn()
    


    @staticmethod
    def create_custom_forward(module):
        def custom_forward(*args, **kwargs):
            return module(*args, **kwargs)
        return custom_forward

    def checkpoint(self, layer, module, *args, **kwargs):
        condition = self.training and self.checkpoint_every_n != 0 and layer < self.depth - 1 and layer % self.checkpoint_every_n == 0
        return checkpoint(self.create_custom_forward(module), *args, **kwargs) if condition else module(*args, **kwargs)

    @staticmethod
    def get_cache(cache):
        if cache is None:
            return None
        return cache['cache'][0]

    @staticmethod
    def get_cache_indices(x_lens, cache_lens, cache_kv, x):  
        # used later w/ gather to remove padding when cache is concatenated with current input to remove padding
        max_new_len = (x_lens + cache_lens).max()
        # cache kv =  LAYERS, KEYS+VALUES (2), BATCH, HEADS, N, DIM
        B, H, N, D = x.shape[0], cache_kv.shape[-3], (x.shape[1] + cache_kv.shape[-2]), cache_kv.shape[-1]
        indices = []
        for i in range(B): # stinky for loop to sort out indices for gather 
            cache_indices = torch.arange(cache_lens[i], device='cpu')
            total_length = cache_lens[i] + x_lens[i] 
            diff_from_max_len = max_new_len - total_length
            x_indices = torch.arange(x_lens[i]+diff_from_max_len, device='cpu') + cache_kv.shape[-2]
            if diff_from_max_len > 0:
                x_indices[-diff_from_max_len:] = N  # last index will be used for padding
            new_indices = torch.cat([cache_indices, x_indices])
            indices.append(new_indices)

        indices = torch.stack(indices, dim=0)
        
        indices = rearrange(indices, 'b n -> () b () n ()').expand(2, B, H,-1, D) # 2 for key and value
        return indices.to(x.device)

    def create_masks_and_positions(self, x, length, cache): 
        x_len = length if length is not None else torch.tensor(x.shape[-2]).expand(x.shape[0])
        cache_len = cache['cache_lengths'] if exists(cache) else 0

        total_len = x_len + cache_len
        kv_mask = torch.arange(total_len.max(), device=x.device).expand(len(total_len), -1) >= total_len.unsqueeze(-1)
        q_mask = torch.arange(x_len.max(), device=x.device).expand(len(x_len), -1) >= x_len.unsqueeze(-1)
        attn_mask = ~(rearrange(~q_mask, "b n -> b () n ()") * rearrange(~kv_mask, "b n -> b () () n"))
        ##
        ##
        causal_mask = repeat(torch.arange(total_len.max(), device=x.device), 'i -> b r i', b=len(total_len), r=x_len.max())
        cache_offset = cache_len[:,None,None] if exists(cache) else cache_len
        diagonal_offset = torch.arange(x_len.max(), device=x.device)[None,:,None]
        ##
        ## positional stuff ##
        positional_grid = (causal_mask - cache_offset - diagonal_offset) * -1
        pos = torch.arange(positional_grid.min(), positional_grid.max()+1, device=x.device, dtype=x.dtype)[:,None]
        min_cache_len = 0 if cache_len.__class__ == int else cache_len.min()
        positional_indices = ((positional_grid) + (total_len.max() - min_cache_len - 1)) # shift so zero is the smallest number
        pos_bias = self.positional_bias(pos=pos, indices=positional_indices, dtype=x.dtype, device=x.device)
        ## positional stuff ##
        ##
        if self.causal:
            causal_mask = causal_mask >= (cache_offset + diagonal_offset + 1)
            attn_mask = torch.logical_or(attn_mask, causal_mask[:,None])
        ##
        return q_mask, attn_mask, total_len, x_len, cache_len, pos_bias


    def next_token_mse_loss(self, x, y, length):
        '''minimise MSE between each x_t and y_t+1'''
        B,N,D = x.shape
        y = y[:,1:].clone()
        x = x[:,:-1].clone()
        # mask that is shifted for next token prediction
        mask = torch.arange(N-1, device=x.device).expand(B,-1) >= length[:,None] -2 # -2 bcos -1 for N-1 arange and -1 for shifted mask
        x = l2norm(x, groups=8)
        y = l2norm(y, groups=8)
        loss = self.checkpoint(0, self.einopsfn, x, y, 'b n d, b n d -> b n')
        loss = (-1) * loss # maximise similarity
        loss = loss + 8 # shift so that loss is positive (number of groups)
        
        loss = loss.masked_fill(mask, 0)
        loss = loss.sum() / (~mask).sum()
        if torch.isnan(loss): # empty sequence
            loss = torch.tensor(0, device=x.device, dtype=x.dtype)
        return loss 



    def forward(self, x, length, cache=None, **kwargs):
        
        intermediate_logits = []
        next_token_preds = []
        
        intermediate_targets = [None] # for the first layer we use ground truth tokens as targets
        cache_lengths = []
        lengths = [length.clone()]
        cached_kvs = []
        x_outs = []

 
        curcache = cache[0] if exists(cache) else None
        mask, attn_mask, total_lens, x_len, cache_len, pos_bias = self.create_masks_and_positions(x, length, curcache)
        cache_indices = self.get_cache_indices(x_len, cache_len, curcache['cache'], x) if exists(curcache) else None

        mask, attn_mask, total_lens, x_len, cache_len, pos_bias = self.create_masks_and_positions(x, length, curcache)
        cache_indices = self.get_cache_indices(x_len, cache_len, curcache['cache'], x) if exists(curcache) else None

        #all_ema_data = []

        for i in range(self.depth):
            attn_ff, pred_layer = self.attn_ffs[i], self.prediction_layers[i]

            ## attention ff blocks ##
            
            attn_cache = [cache[ix]['cache'][0] if exists(cache) else None for ix in range(i*self.inner_depth, i*self.inner_depth+self.inner_depth)] 
            x, kvs = self.checkpoint(i, attn_ff, x, pos_bias, attn_mask, attn_cache, cache_indices)
    
            cached_kvs.extend(kvs)
            cache_lengths.extend([total_lens]*len(kvs))
            ## attention ff blocks ##
            if i!= 0:
                pred_scaler = self.pred_scalers[i-1]
                pred_emb = self.vqs[i-1].vq._codebook.embed
                pred_emb = self.checkpoint(i, l2norm, pred_emb)
                z = x
                zsim = self.checkpoint(i, pred_layer, z)
                zsim = rearrange(zsim, 'b n (h d) -> b h n d', h=self.codebook_heads)
                zsim = self.checkpoint(i, l2norm, zsim)
                zsim = self.checkpoint(i, self.einopsfn, zsim, pred_emb, 'b h n d, h v d -> b h n v') * pred_scaler
            else:
                zsim = self.checkpoint(i, pred_layer, x)
        
            clen = length.clone()
            
            if i!= 0:
                clen = repeat(clen, 'b -> (b h)', h=self.codebook_heads)
                zsim = rearrange(zsim, 'b h n v -> (b h) n v') 

            ntp = grab_last_token(zsim, clen)   
            if i!= 0:
                ntp = rearrange(ntp, '(b h) n v -> b h n v', h=self.codebook_heads)
                zsim = rearrange(zsim, '(b h) n v -> b h n v', h=self.codebook_heads)
                
            intermediate_logits.append(zsim)
            next_token_preds.append(ntp)
     

            if i != self.depth-1:
                halfer = self.halfers[i]
                x, length = self.checkpoint(i, halfer, x, length)
                x_outs.append(x)
                xbelow = x.clone()
                lengths.append(length.clone())
                
                
                curcache = cache[(i+1)*self.inner_depth] if exists(cache) else None
                mask, attn_mask, total_lens, x_len, cache_len, pos_bias = self.create_masks_and_positions(x, length, curcache)
                cache_indices = self.get_cache_indices(x_len, cache_len, curcache['cache'], x) if exists(curcache) else None


                B, N, D = x.shape
                vq = self.vqs[i]
                indices = self.checkpoint(i, vq, x)

                '''if self.training:
                    all_ema_data.append(ema_data)'''

                if len(indices.shape) == 2:
                    indices = rearrange(indices, 'b n -> b n ()') # add dummy head dimension

                intermediate_targets.append(indices)

        #orth_loss = orthogonal_loss_fn(pred_matrix)
        #commitment_loss =  orth_loss.sum() * 10.0 
        

        #print(len(cached_kvs), len(layer_below_next_token_preds), len(cache_lengths), len(next_token_preds))
        assert len(cached_kvs) == len(cache_lengths), 'something went wrong'
  
        cached_kvs = [{'cache': curcache, 'cache_lengths': curlen} for curcache, curlen in zip(cached_kvs, cache_lengths)]
        cached_kvs = {'layers': cached_kvs, 'next_sentence_pred': next_token_preds}
        '''# shift layers down list down by 1 
        print(len(cached_kvs['layers']))
        cached_kvs['layers'] = cached_kvs['layers'][self.inner_depth:] + [cached_kvs['layers'][-1]]*self.inner_depth
        print(len(cached_kvs['layers']))'''

        
        return {
            'logits': intermediate_logits,
            'targets': intermediate_targets,
            'cache': cached_kvs,
            'lengths': torch.stack(lengths),
            'x_outs': x_outs,
            #'all_ema_data': all_ema_data
        }


class transformer_lm(nn.Module):
    def __init__(
        self,
        dim,
        vocab_size,
        depth,
        heads,
        dim_head,
        causal=True,
        dropout=0.,
        use_abs_pos=False,
        **kwargs
    ):
        super().__init__()
    
        self.use_abs_pos = use_abs_pos
        if self.use_abs_pos:
            self.abs_pos_fn = ScaledSinuEmbedding(dim=dim)
        self.abs_pos = lambda x: x + self.abs_pos_fn(x) if self.use_abs_pos else x


        self.layers = transformer(
            dim = dim, 
            depth = depth, 
            heads = heads, 
            dim_head = dim_head, 
            causal = causal, 
            dropout = dropout,
            base_vocab_size = vocab_size,
            **kwargs
        )
        

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

   
    def calc_all_losses(self, tlm_out, prev_cache):
        eos_id = -100
        loss_fn = lambda l, t: F.cross_entropy(rearrange(l, 'b n c -> b c n'), t, ignore_index=-100, reduction='mean')
        losses = []

        def calc_token_loss(logits, targets, first_token_pred, length):
            if length.max() == 1 and not exists(first_token_pred):
                return None # no loss for single token sequences if no previous prediction is available
            heads = 1
            if len(logits.shape) == 4:
                heads = logits.shape[1]
                length = repeat(length, 'b -> (b h)', h=heads).clone().contiguous()
                logits = rearrange(logits, 'b h n v -> (b h) n v')
                targets = rearrange(targets, 'b n (h 1) -> (b h) n')
                first_token_pred = None if not exists(first_token_pred) else rearrange(first_token_pred, 'b h n v -> (b h) n v')

            if exists(first_token_pred): # concat zero vect to end of targets
                targets = rearrange(targets, '(b h) n -> b h n', h=heads)
                targets = torch.cat([targets, torch.zeros(targets.size(0), heads, 1, device=targets.device).long()], dim=2)
                targets = rearrange(targets, 'b h n -> (b h) n')
                logits = torch.cat([first_token_pred, logits], dim=1)
                length += 1
            else:
                if heads == 1:
                    targets[:,:-1] = targets.clone()[:,1:]
                else:
                    targets = rearrange(targets, '(b h) n -> b h n', h=heads)
                    targets[:,:,:-1] = targets.clone()[:,:,1:]
                    targets = rearrange(targets, 'b h n -> (b h) n')
            
            targets = add_eos(targets, eos_id=eos_id, token_lens=length)
            mask =  token_lens_to_mask(token_lens=length)
            
            targets = mark_padding(targets=targets, mask=mask, pad_id=eos_id)
            loss = loss_fn(logits, targets) #* heads # multiply by heads 
            
            return loss

    
        if exists(prev_cache):
            assert len(tlm_out['logits']) == len(prev_cache['next_sentence_pred']), 'something went wrong'

        for lth in range(len(tlm_out['logits'])):
            logits = tlm_out['logits'][lth]
            targets = tlm_out['targets'][lth]
            
            #print(targets.reshape(-1).unique().shape)
            first_token_pred = prev_cache['next_sentence_pred'][lth] if exists(prev_cache) else None
            
            lengths = tlm_out['lengths'][lth]
            loss = calc_token_loss(logits, targets.clone(), first_token_pred, lengths.clone())
            if exists(loss): # incase of single token sequences
                losses.append(loss)
              
        tlm_out['token_losses'] = torch.stack(losses) 

        return tlm_out

    def _forward(self, labels, length, cache:Dict=None, calc_loss=False, **kwargs):
        '''
        x: [B, N] (embedding indices)
        length: [B] (length of each sequence)
        cache: {cache_lengths: [B, N], cache: [L, KV, B, H, N, D]} KV: key and value (2)
        '''
        assert labels.shape[1] == length.max(), 'sequence length should be equal to the length of the longest sequence!'
        x = self.layers.embedding(labels)
        x = self.abs_pos(x) 
        
        outputs = self.layers(x, length, cache=cache['layers'] if exists(cache) else None, **kwargs)

        '''if len(outputs['all_ema_data']) > 0:
            for ix in range(len(outputs['all_ema_data'])):
                self.layers.vqs[ix]._codebook.update_codebook(
                    bins = outputs['all_ema_data'][ix]['bins'],
                    embed_sum = outputs['all_ema_data'][ix]['embed_sum'],
                )'''

        outputs['targets'][0] = labels.clone() # for the first layer we use ground truth tokens as targets
        if calc_loss:
            outputs = self.calc_all_losses(tlm_out=outputs, prev_cache=cache)
   
        return outputs

    def dev_forward(self, labels, length, cache:Dict=None, calc_loss=False, **kwargs):
        '''
        x: [B, N] (embedding indices)
        length: [B] (length of each sequence)
        cache: {cache_lengths: [B, N], cache: [L, KV, B, H, N, D]} KV: key and value (2)
        '''
        assert labels.shape[1] == length.max(), 'sequence length should be equal to the length of the longest sequence!'
        total_layer_lengths = torch.zeros(self.layers.depth, device=labels.device)
        total_layers_losses = torch.zeros(self.layers.depth, device=labels.device)

        x = self.layers.embedding(labels)
        x = self.abs_pos(x) 

        subsize = (2 ** (self.layers.depth - 1)) * 3
        B, N, D = x.shape
        substeps = math.ceil(N / subsize)
        remaining_length = length.clone()
        all_outputs = []
        prev_x, prev_labels = [x], [labels]
        
        prev_states = cache

        all_ema_data = []

        for i in range(substeps):
            

            if exists(prev_states):
                for ix in range(len(prev_states['layers'])):
                    inner_depth = self.layers.inner_depth 
                    #c_ix = ix + inner_depth if ix < len(prev_states['layers']) - inner_depth else len(prev_states['layers']) - 1
                    c_ix = ix
                    prev_states['layers'][ix]['cache'] = prev_states['layers'][c_ix]['cache'][:,:,unfinished_indices]
                    prev_states['layers'][ix]['cache_lengths'] = prev_states['layers'][c_ix]['cache_lengths'][unfinished_indices]

                for ix in range(len(prev_states['next_sentence_pred'])):
                    prev_states['next_sentence_pred'][ix] = prev_states['next_sentence_pred'][ix][unfinished_indices]

            #print(f'step {i} of {substeps} ({i/substeps*100:.2f}%)')
            curx = x[:, i*subsize:(i+1)*subsize].clone().contiguous()
            curlabels = labels[:, i*subsize:(i+1)*subsize].clone().contiguous()
            curlength = remaining_length.clone()
            curlength[curlength > subsize] = subsize
            remaining_length -= subsize
            
            outputs = self.layers(curx, curlength, cache=prev_states['layers'] if exists(prev_states) else None, **kwargs)

            '''if len(outputs['all_ema_data']) > 0:
                if len(all_ema_data) == 0:
                    all_ema_data = outputs['all_ema_data']
                else:
                    for ix in range(len(all_ema_data)):
                        all_ema_data[ix]['bins'] += outputs['all_ema_data'][ix]['bins']
                        all_ema_data[ix]['embed_sum'] += outputs['all_ema_data'][ix]['embed_sum'] '''

            #finished_indices = (curlength <= 0).nonzero().squeeze(1)
            # get length indices that are not less than/equal to zero (i.e. finished)
            unfinished_indices = (remaining_length > 0).nonzero().squeeze(1)
            finished_indices = (curlength <= 0).nonzero().squeeze(1)
            #print(f'finished: {len(finished_indices)}')
         
            x = x[unfinished_indices]
            labels = labels[unfinished_indices]
            remaining_length = remaining_length[unfinished_indices]
            length = length[unfinished_indices]

            outputs['targets'][0] = curlabels.clone() # for the first layer we use ground truth tokens as targets
            if calc_loss:
                outputs = self.calc_all_losses(tlm_out=outputs, prev_cache=prev_states)

                total_layer_lengths[:len(outputs['token_losses'])] += outputs['lengths'].sum(-1)[:len(outputs['token_losses'])]
                total_layers_losses[:len(outputs['token_losses'])] += outputs['token_losses'][:len(outputs['token_losses'])] * outputs['lengths'].sum(-1)[:len(outputs['token_losses'])]
                
            prev_states = outputs['cache']

        #print(total_layer_lengths, outputs['lengths'].sum(-1))
        total_layers_losses /= (total_layer_lengths + 1e-8)
        
        #total_layers_losses[:-1] = total_layers_losses[:-1] * 0.0 # only use the last layer for loss calculation (hehe :)
      
        '''if len(all_ema_data) > 0:
            for ix in range(len(all_ema_data)):
                self.layers.vqs[ix]._codebook.update_codebook(
                    bins = all_ema_data[ix]['bins'],
                    embed_sum = all_ema_data[ix]['embed_sum'],
                )'''

        return total_layers_losses

    def forward(self, labels, length, cache:Dict=None, calc_loss=False, ema_targets=None, **kwargs):
        '''
        x: [B, N] (embedding indices)
        length: [B] (length of each sequence)
        cache: {cache_lengths: [B, N], cache: [L, KV, B, H, N, D]} KV: key and value (2)
        '''
        assert labels.shape[1] == length.max(), 'sequence length should be equal to the length of the longest sequence!'
        total_layer_lengths = torch.zeros(self.layers.depth, device=labels.device)
        total_layers_losses = torch.zeros(self.layers.depth, device=labels.device)

        x = self.layers.embedding(labels)
        x = self.abs_pos(x) 

        subsize = (2 ** (self.layers.depth - 1)) * 3
        B, N, D = x.shape
        substeps = math.ceil(N / subsize)
        remaining_length = length.clone()
        all_outputs = []
        prev_x, prev_labels = [x], [labels]
        
        prev_states = cache

        iteration_targets = []


        for i in range(substeps):
            
            if exists(prev_states):
                for ix in range(len(prev_states['layers'])):
                    inner_depth = self.layers.inner_depth 
                    #c_ix = ix + inner_depth if ix < len(prev_states['layers']) - inner_depth else len(prev_states['layers']) - 1
                    c_ix = ix
                    prev_states['layers'][ix]['cache'] = prev_states['layers'][c_ix]['cache'][:,:,unfinished_indices]
                    prev_states['layers'][ix]['cache_lengths'] = prev_states['layers'][c_ix]['cache_lengths'][unfinished_indices]

                for ix in range(len(prev_states['next_sentence_pred'])):
                    prev_states['next_sentence_pred'][ix] = prev_states['next_sentence_pred'][ix][unfinished_indices]

            #print(f'step {i} of {substeps} ({i/substeps*100:.2f}%)')
            curx = x[:, i*subsize:(i+1)*subsize].clone().contiguous()
            curlabels = labels[:, i*subsize:(i+1)*subsize].clone().contiguous()
            curlength = remaining_length.clone()
            curlength[curlength > subsize] = subsize
            remaining_length -= subsize
            
            outputs = self.layers(curx, curlength, cache=prev_states['layers'] if exists(prev_states) else None, **kwargs)

            '''if len(outputs['all_ema_data']) > 0:
                if len(all_ema_data) == 0:
                    all_ema_data = outputs['all_ema_data']
                else:
                    for ix in range(len(all_ema_data)):
                        all_ema_data[ix]['bins'] += outputs['all_ema_data'][ix]['bins']
                        all_ema_data[ix]['embed_sum'] += outputs['all_ema_data'][ix]['embed_sum'] '''

            #finished_indices = (curlength <= 0).nonzero().squeeze(1)
            # get length indices that are not less than/equal to zero (i.e. finished)
            unfinished_indices = (remaining_length > 0).nonzero().squeeze(1)
            finished_indices = (curlength <= 0).nonzero().squeeze(1)
            #print(f'finished: {len(finished_indices)}')
         
            x = x[unfinished_indices]
            labels = labels[unfinished_indices]
            remaining_length = remaining_length[unfinished_indices]
            length = length[unfinished_indices]

            outputs['targets'][0] = curlabels.clone() # for the first layer we use ground truth tokens as targets

            if ema_targets is None:
                iteration_targets.append(outputs['targets'])

            if calc_loss and ema_targets is not None:
                outputs['targets'] = ema_targets[i]
                outputs = self.calc_all_losses(tlm_out=outputs, prev_cache=prev_states)

                total_layer_lengths[:len(outputs['token_losses'])] += outputs['lengths'].sum(-1)[:len(outputs['token_losses'])]
                total_layers_losses[:len(outputs['token_losses'])] += outputs['token_losses'][:len(outputs['token_losses'])] * outputs['lengths'].sum(-1)[:len(outputs['token_losses'])]
                
            prev_states = outputs['cache']

        #print(total_layer_lengths, outputs['lengths'].sum(-1))
        total_layers_losses /= (total_layer_lengths + 1e-8)
        
        #total_layers_losses[:-1] = total_layers_losses[:-1] * 0.0 # only use the last layer for loss calculation (hehe :)
      
        '''if len(all_ema_data) > 0:
            for ix in range(len(all_ema_data)):
                self.layers.vqs[ix]._codebook.update_codebook(
                    bins = all_ema_data[ix]['bins'],
                    embed_sum = all_ema_data[ix]['embed_sum'],
                )'''

        return total_layers_losses, iteration_targets
        