import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
from einops import rearrange, repeat
from torch import einsum
from torch.utils.checkpoint import checkpoint # # gradient/activation checkpointing
from functools import partial

from .state_space import S4

class DynamicPositionBias(nn.Module):
    '''taken From Phil Wang's x-transformers library'''
    def __init__(self, dim, *, heads, depth, log_distance = False, norm = False):
        super().__init__()
        assert depth >= 1, 'depth for dynamic position bias MLP must be greater or equal to 1'
        self.log_distance = log_distance

        self.mlp = nn.ModuleList([])

        self.mlp.append(nn.Sequential(
            nn.Linear(1, dim),
            nn.LayerNorm(dim) if norm else nn.Identity(),
            nn.ReLU()
        ))

        for _ in range(depth - 1):
            self.mlp.append(nn.Sequential(
                nn.Linear(dim, dim),
                nn.LayerNorm(dim) if norm else nn.Identity(),
                nn.ReLU()
            ))

        self.mlp.append(nn.Linear(dim, heads))

    def forward(self, n, device, dtype):

        # get the (n x n) matrix of distances
        seq_arange = torch.arange(n, device = device)
        context_arange = torch.arange(n, device = device)
        indices = rearrange(seq_arange, 'i -> i 1') - rearrange(context_arange, 'j -> 1 j')
        indices += (n - 1)
        
        # input to continuous positions MLP
        pos = torch.arange(-n + 1, n, device = device, dtype = dtype)
        pos = rearrange(pos, '... -> ... 1')

        if self.log_distance:
            pos = torch.sign(pos) * torch.log(pos.abs() + 1)  # log of distance is sign(rel_pos) * log(abs(rel_pos) + 1)

        for layer in self.mlp:
            pos = layer(pos)

        # get position biases        
        bias = pos[indices]
        bias = rearrange(bias, 'i j h -> h i j')
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

def pad_to_window_size(x, window_size, axis=3, mask=None):
    """
    Pad the input on two sides to be divisible by `window_size`
    """
    batch_size, sequence_length, hidden_size = x.shape
    if sequence_length % window_size == 0:
        return x, 0, mask
    padding_length = (window_size - sequence_length % window_size) % window_size
    padding = torch.zeros(batch_size, padding_length, hidden_size,
        device=x.device,
        dtype=x.dtype,
    )
    mask = F.pad(mask, (0, padding_length), value=True) 
    return torch.cat([x, padding], axis=axis), padding_length, mask

def unpad(x, padding_length):
    """
    Undo padding.
    """
    if padding_length > 0:
        return x[:, :-padding_length]
    return x

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

        self.n_feats, self.head_dim, self.n_heads = n_feats, head_dim, n_heads
        self.dropout = nn.Dropout(dropout)
        self.bias = bias
        self.return_attention = return_attention
        self.causal = causal

        self.temperature = torch.nn.Parameter(torch.tensor(temperature), requires_grad=True) if isinstance(temperature, float) else temperature

        self.activation = ReLUSquared() if activation == 'relusq' else nn.Softmax(dim=-1)

        if not self.shared_kv:
            self.qkv_proj = nn.Linear(n_feats, 3 * n_heads * head_dim, bias=bias)
            self.qkv = lambda x: rearrange(self.qkv_proj(x), "b w n (h d qkv) -> qkv b h w n d", qkv=3, h=n_heads, d=head_dim)
        else:
            self.q_proj, self.kv_proj = [nn.Linear(n_feats, el, bias=bias) for el in [n_heads * head_dim, 2 * head_dim]]
            map_q, map_kv = lambda q: rearrange(q, 'b n (h d) -> b h n d', h=n_heads), lambda kv: rearrange(kv, 'b n (kv d) -> kv b () n d', kv=2, d=head_dim)
            self.qkv = lambda x: (map_q(self.q_proj(x)), *map_kv(self.kv_proj(x)))

        self.out_proj = nn.Linear(n_heads * head_dim, n_feats, bias=bias)


    def attend(self, query, key, value, mask, pos_fn):
        
        query, key = map(l2norm, (query, key))

        dots = einsum('bhwid,bhwjd->bhwij', query, key) * self.temperature
        
        dots += pos_fn(dots.shape[-1], device=dots.device, dtype=dots.dtype).unsqueeze(1)
        qkmask = ~mask
        attn_mask = ~(rearrange(qkmask, "b w n -> b () w n ()") * rearrange(qkmask, "b w n -> b () w () n"))


        if self.causal: # create a regular causal mask    
            causal_mask = torch.ones(dots.shape[-2], dots.shape[-1], device=dots.device).triu(1).bool()
            attn_mask = torch.logical_or(attn_mask, causal_mask)
        
        dots.masked_fill_(attn_mask, -torch.finfo(dots.dtype).max)
    
        attn = self.activation(dots)   

        attn = self.dropout(attn)
 
        return einsum("bhwij,bhwjd->bhwid", attn, value)


    def forward(self, x, pos_fn, mask=None):
        assert pos_fn is not None, 'pls provide a position function'
        B, N, C, H, D = *x.shape, self.n_heads, self.head_dim
    
        if mask is None:
            mask = torch.zeros(B, N, device=x.device, dtype=torch.bool)

        WINDOW_SIZE = 25 if N > 45 else N
        
        x, pad_n, mask = pad_to_window_size(x, window_size=WINDOW_SIZE, axis=-2, mask=mask) # first pad so that sequence length is divisible by window size
        B, N, C = x.shape
        x = rearrange(x, 'b (w n) d -> b w n d', w=N// WINDOW_SIZE, n=WINDOW_SIZE) # group into windows
        mask = rearrange(mask, 'b (w n) -> b w n', w=N// WINDOW_SIZE, n=WINDOW_SIZE)

        q, k, v = self.qkv(x)
    
        out = self.attend(q, k, v, mask, pos_fn)

        out = rearrange(out, 'b h w n d -> b (w n) (h d)')

        out = self.out_proj(out)
        out = unpad(out, pad_n)
        return out

    '''
    def forward(self, x, pos_fn, mask=None):
        assert pos_fn is not None, 'pls provide a position function'
        B, N, C, H, D = *x.shape, self.n_heads, self.head_dim
        #print(x.shape, mask.shape)

        if mask is None:
            mask = torch.zeros(B, N, device=x.device, dtype=torch.bool)

        q, k, v = self.qkv(x)
    
        out = self.attend(q, k, v, mask, pos_fn)

        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.out_proj(out)
        return out'''

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

        self.temperature = nn.Parameter(torch.tensor(temperature), requires_grad=True) if shared_temperture else temperature

        self.intermediate_loss = intermediate_loss


        self.depth = depth
        self.positional_bias = DynamicPositionBias(
            dim = dim // 4,
            heads = heads,
            depth = 2,
            log_distance = False,
            norm = False
        )

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, S4(
                    d_model = dim,
                    dropout = 0.1,
                    tie_dropout = True,
                    transposed = False,
                )),
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

    def forward(self, x, mask=None, self_condtioning=None):
        intermediate_logits = []
        lengths = (~mask).sum(dim=-1)
        for i, (s4, attn, ff) in enumerate(self.layers):

            x = self.checkpoint(i, s4, x, None, 1.0, lengths)[0] + x
            x = self.checkpoint(i, attn, x, self.positional_bias, mask) + x
            x = self.checkpoint(i, ff, x) + x   

            if i < self.depth - 1 and self_condtioning is not None:
                x, logits = self_condtioning(x)
                intermediate_logits.append(logits)

        if len(intermediate_logits) > 0: # stack intermediate logits
            intermediate_logits = torch.stack(intermediate_logits, dim=0) # D x B x N x L
    
        return x, intermediate_logits

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


    def forward(self, x, mask=None):
        x = self.embedding(x)
        x = self.abs_pos(x)
        x, interim_logits = self.layers(x, mask=~mask if mask is not None else None, self_condtioning=self.self_condition_fn())
        x = self.post_norm(x)
        x = self.to_logits(x)

        return  { 'out': x, 'interim_logits': interim_logits } if self.training else x


        

