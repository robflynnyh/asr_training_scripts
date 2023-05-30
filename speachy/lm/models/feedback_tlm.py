import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
from einops import rearrange, repeat
import einops
from torch import einsum
from torch.utils.checkpoint import checkpoint # # gradient/activation checkpointing
from speachy.lm.tools.train import add_eos, token_lens_to_mask, mark_padding
from functools import partial
import string
from typing import Optional, Tuple, List, Dict, Union, Callable
import math
import random

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

class CosineAttention(nn.Module):
    def __init__(
        self,
        n_feats,
        head_dim,
        n_heads,
        dropout=0.1,
        bias=False,
        temperature=15.5,
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
  

    def attend(self, query, key, value):
        dots = einsum('bhid,bhjd->bhij', query, key) * self.head_dim ** -0.5
        dots = self.head_proj(dots, mode='pre')

        #dots.masked_fill_(attn_mask, -torch.finfo(dots.dtype).max)

        attn = self.activation(dots)
        attn = self.head_proj(attn, mode='post')

     
        attn = self.dropout(attn)
        return einsum("bhij,bhjd->bhid", attn, value)

    @staticmethod
    def attach_cache(kv, cache):
        kv = torch.stack(kv, dim=0)
        if cache is None:
            return kv
        kv_cache = torch.cat([cache, kv], dim=-2)
        return kv_cache

    def forward(self, x, cache=None):
        B, N, C, H, D = *x.shape, self.n_heads, self.head_dim
    
        q, k, v  = self.qkv(x)
        kv = self.attach_cache([k, v], cache)
        k, v = kv

        out = self.attend(q, k, v)

        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.out_proj(out)
        return out, kv

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(self.norm(x), *args, **kwargs)


class GLU(nn.Module):
    def __init__(self, dim_in, dim_out, activation):
        super().__init__()
        self.act = activation
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim = -1)
        return x * self.act(gate)

'''class CacheProjection(nn.Module):
    def __init__(self, depth):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(depth, depth))

    def forward(self, x_in):
        l, kv, b, h, n, d = x_in.shape
        weight = self.weight.softmax(dim=-1)
        x_in = rearrange(x_in, "l kv b h n d -> l () kv b h n d")
        weighted = einsum('lokbhnd,lz->lkbhnd', x_in, weight)
        return weighted'''

class CacheProjection(nn.Module):
    def __init__(self, dim, exp=2):
        super().__init__()
        self.linear = nn.Linear(dim, dim * exp * 2)
        self.act = nn.SiLU()
        self.out_proj = nn.Linear(dim * exp, dim)

    def forward(self, x_in):
        x = self.linear(x_in)
        a, b = x.chunk(2, dim=-1)
        x = self.act(a) * b
        x = self.out_proj(x)
        return x

class transformer(nn.Module):
    def __init__(
            self, 
            dim, 
            depth, 
            heads, 
            dim_head, 
            causal=True,
            temperature=15.5,
            shared_temperture=False,
            intermediate_loss=True,
            dropout = 0.1,
            **kwargs
        ):
        super().__init__()
        if depth == 1:
            intermediate_loss = False

        ff_mult = kwargs.get('ff_mult', 4)
        self.checkpoint_every_n = kwargs.get('checkpoint_every_n', 0)
        self.token_shift = kwargs.get('token_shift', False)

        self.causal = causal

        self.temperature = nn.Parameter(torch.tensor(temperature), requires_grad=True) if shared_temperture else temperature
        #L, KV, B, H, N, D
        

        #self.cache_projection = nn.Linear(dim_head * depth, dim_head * depth)
        self.cache_projection = PreNorm(dim_head*depth, CacheProjection(dim=dim_head*depth))

        self.intermediate_loss = intermediate_loss

        self.depth = depth
        self.shared_kv = kwargs.get('shared_kv', False)
        self.heads = heads

        self.token_shifter = lambda x: x
        if self.token_shift:
            self.token_shifter = ShiftTokens(range(0, 2), nn.Identity())
        self.token_shift = lambda x: self.token_shifter(x)

        self.layers = nn.ModuleList([])
        self.dim_head = dim_head
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, CosineAttention(
                    dim, 
                    n_heads=heads, 
                    head_dim=dim_head, 
                    causal=causal,
                    temperature=self.temperature,
                    dropout=dropout,
                    **kwargs
                )),
                PreNorm(dim, self.ff(dim, mult=ff_mult))
            ]))

    @staticmethod
    def ff(dim, mult=4, dropout=0.1):
        return nn.Sequential(
            GLU(dim, dim * mult, nn.SiLU()),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim)
        )

    @staticmethod
    def create_custom_forward(module):
        def custom_forward(*args, **kwargs):
            return module(*args, **kwargs)
        return custom_forward

    def checkpoint(self, layer, module, *args, **kwargs):
        condition = self.training and self.checkpoint_every_n != 0 and layer < self.depth - 1 and layer % self.checkpoint_every_n == 0
        return checkpoint(self.create_custom_forward(module), *args, **kwargs) if condition else module(*args, **kwargs)

    @staticmethod
    def get_cache(cache, i):
        if cache is None:
            return None
        return cache['cache'][i]




    def forward(self, x, length=None, self_condtioning=None, cache=None, **kwargs):
        B,N,D = x.shape
        intermediate_logits = []
        cached_kvs = []
  
        o_cache = cache['cache'].clone() if exists(cache) else None
        heads = self.heads if not self.shared_kv else 1

        kv_stack = []
        for i, (attn, ff) in enumerate(self.layers):

            x = self.token_shift(x)
            a_out, kv = self.checkpoint(i, attn, x, self.get_cache(cache, i))
            x = a_out + x

            kv_stack.append(kv[:,:,:,-1,None])
 
            x = self.checkpoint(i, ff, x) + x   

            if i < self.depth - 1 and self_condtioning is not None:
                x, logits = self_condtioning(x)
                intermediate_logits.append(logits)
        
    
        kv_stack = torch.cat(kv_stack, dim=-1)
        kv_stack = self.cache_projection(kv_stack)
        kv_stack = rearrange(kv_stack, "kv b h n (l d) -> l kv b h n d", l=self.depth)

        intermediate_logits = torch.stack(intermediate_logits, dim=0) if len(intermediate_logits) > 0 else None

        cached_kvs = torch.cat([o_cache, kv_stack], dim=-2) if exists(o_cache) else kv_stack
        cached_kvs = {'cache_lengths': cache['cache_lengths'] + 1 if exists(cache) else torch.ones(B).long().to(x.device), 'cache': cached_kvs}

        return x, intermediate_logits, cached_kvs

class shared_embedding_output_layer(nn.Module):
    '''Pass a embedding layer and then use this module as the output layer'''
    def __init__(self, embedding_layer, bias=False):
        super().__init__()
        self.embedding_layer = embedding_layer
        self.use_bias = bias
        if bias:
            self.bias = nn.Parameter(torch.zeros(embedding_layer.weight.shape[0]))#
            nn.init.xavier_uniform_(self.bias)

    def forward(self, x):
        return F.linear(x, weight=self.embedding_layer.weight, bias=self.bias if self.use_bias else None)


class transformer_lm(nn.Module):
    def __init__(
        self,
        dim,
        vocab_size,
        depth,
        heads,
        dim_head,
        causal=True,
        temperature=15.5,
        dropout=0.,
        shared_temperture=True,
        self_conditioning=False,
        intermediate_loss=True,
        use_abs_pos=False,
        **kwargs
    ):
        super().__init__()
        if depth == 1:
            self_conditioning == False

        self.self_conditioning = True if self_conditioning else None
        self.intermediate_loss = intermediate_loss

        self.use_abs_pos = use_abs_pos
        if self.use_abs_pos:
            self.abs_pos_fn = ScaledSinuEmbedding(dim=dim)
        self.abs_pos = lambda x: x + self.abs_pos_fn(x) if self.use_abs_pos else x

        if self_conditioning:
            self.reprojection_layer = nn.Linear(vocab_size, dim)

        self.layers = transformer(
            dim = dim, 
            depth = depth, 
            heads = heads, 
            dim_head = dim_head, 
            causal = causal, 
            dropout = dropout,
            temperature = temperature,
            shared_temperture = shared_temperture,
            intermediate_loss = intermediate_loss,
            **kwargs
        )

        self.tie_embedding = kwargs.get('tie_embedding', False)
        print('Tie embedding:', self.tie_embedding) if self.tie_embedding else None
 
        self.embedding = nn.Embedding(vocab_size, dim)
        self.to_logits = shared_embedding_output_layer(self.embedding) if self.tie_embedding else nn.Linear(dim, vocab_size)
        self.vocab_size = vocab_size

        self.post_norm = nn.LayerNorm(dim)


    def self_condition_fn(self):
        def self_condition(x):
            logits = self.to_logits(self.post_norm(x))
            if self.self_conditioning: # not effective for LMs (intermediate loss is tho)
                z = F.softmax(logits, dim=-1)
                z = self.reprojection_layer(z)
                x = z + x
            return x, logits
        return self_condition if (self.self_conditioning or self.intermediate_loss) and self.training else None

    def calc_all_losses(self, logits, targets, lengths):
        eos_id = -100
        loss_fn = lambda l, t: F.cross_entropy(rearrange(l, 'b n c -> b c n'), t, ignore_index=-100, reduction='mean')
        losses = []

        def calc_token_loss(logits, targets, length):
            if length.max() == 1:
                return None # no loss for single token sequences if no previous prediction is available
    
            targets[:,:-1] = targets.clone()[:,1:]        
            targets = add_eos(targets, eos_id=eos_id, token_lens=length)
            mask =  token_lens_to_mask(token_lens=length)
            
            targets = mark_padding(targets=targets, mask=mask, pad_id=eos_id)
            loss = loss_fn(logits, targets) #* heads # multiply by heads 
            
            return loss

        loss = calc_token_loss(logits, targets.clone(), lengths.clone())
        if exists(loss): # incase of single token sequences
            losses.append(loss)
  
        losses = torch.stack(losses) 
        return losses


    def forward(self, labels, length, cache:Dict=None, calc_loss=False, **kwargs):
        '''
        x: [B, N] (embedding indices)
        length: [B] (length of each sequence)
        cache: {cache_lengths: [B, N], cache: [L, KV, B, H, N, D]} KV: key and value (2)
        '''
        if labels.shape[1] != length.max():
            labels = labels[:, :length.max()]

        x = self.embedding(labels)
        x = self.abs_pos(x) 

        subsize = 1
        B, N, D = x.shape
        substeps = math.ceil(N / subsize)
        remaining_length = length.clone()
        untrimmed_remaining_length = length.clone()
        unfinished_untrimmed_indices = torch.arange(B, device=labels.device)
        unfinished_indices = torch.arange(B, device=labels.device)
        
        prev_states = cache

        logits_data = torch.zeros((B, N, self.vocab_size), device=x.device)

        for i in range(substeps):

            if exists(prev_states):
                prev_states['cache'] = prev_states['cache'][:, :,  unfinished_indices]
                prev_states['cache_lengths'] = prev_states['cache_lengths'][unfinished_indices]
  
            #print(f'step {i} of {substeps} ({i/substeps*100:.2f}%)') #if random.random() < 0.05 else None
            curx = x[:, i*subsize:(i+1)*subsize].clone().contiguous()
            curlength = remaining_length.clone()
            curlength[curlength > subsize] = subsize
            remaining_length -= subsize
            untrimmed_remaining_length -= subsize
            
            curx, _, cache = self.layers(curx, curlength, self_condtioning=self.self_condition_fn(), cache=prev_states, **kwargs)
            curx = self.post_norm(curx)
            logits = self.to_logits(curx)
            
            logits_data[unfinished_untrimmed_indices, i*subsize:(i+1)*subsize] = logits
       
            unfinished_indices = (remaining_length > 0).nonzero().squeeze(1)    
       
            unfinished_untrimmed_indices = (untrimmed_remaining_length > 0).nonzero().squeeze(1)
            
            x = x[unfinished_indices]
            
            remaining_length = remaining_length[unfinished_indices]
            prev_states = cache

        loss = None
        if calc_loss:
            loss = self.calc_all_losses(logits=logits_data, targets=labels, lengths=length)


        return logits_data, loss, prev_states