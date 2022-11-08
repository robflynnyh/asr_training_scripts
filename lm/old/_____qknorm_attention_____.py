import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
import numpy as np
from einops import rearrange, repeat
from torch import einsum
from torch.utils.checkpoint import checkpoint # # gradient/activation checkpointing

    


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

class ReLUSquared(nn.Module):
    def forward(self, x):
        return torch.pow(F.relu(x), 2)

def l2norm(x, dim = -1):
    return F.normalize(x, p = 2, dim = dim)

class CosineAttention(nn.Module):
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
        self.n_feats = n_feats
        self.head_dim = head_dim
        self.n_heads = n_heads
        self.dropout = nn.Dropout(dropout)
        self.bias = bias
        self.return_attention = return_attention

        self.causal = causal


        self.activation = ReLUSquared() if activation == 'relusq' else nn.Softmax(dim=-1)


        self.qkv_proj = nn.Linear(n_feats, 3 * n_heads * head_dim, bias=bias)
        self.out_proj = nn.Linear(n_heads * head_dim, n_feats, bias=bias)


    def attend(self, qkv, mask, pos_fn, temperature):
        query, key, value = qkv

       
        query, key = map(l2norm, (query, key))
        dots = torch.einsum('bhid,bhjd->bhij', query, key) * temperature
  
        dots += pos_fn(dots.shape[-1], device=dots.device, dtype=dots.dtype)
        qkmask = ~mask
        attn_mask = ~(rearrange(qkmask, "b n -> b () n ()") * rearrange(qkmask, "b n -> b () () n"))
    
        if self.causal: # create a regular causal mask
            causal_mask = torch.ones(dots.shape[-2], dots.shape[-1], device=dots.device).triu(1).bool()
            attn_mask = torch.logical_or(attn_mask, causal_mask)
        
        dots.masked_fill_(attn_mask, -torch.finfo(dots.dtype).max)
    
        attn = self.activation(dots)   
        attn = self.dropout(attn)
        return torch.einsum("bhij,bhjd->bhid", attn, value)


    def forward(self, x, pos_fn, mask=None, temperature=None):
        assert pos_fn is not None, 'pls provide a position function'
        assert temperature is not None, 'pls provide a temperature' # fix this later
    
        B, N, C, H, D = *x.shape, self.n_heads, self.head_dim
        #print(x.shape, mask.shape)

        if mask is None:
            mask = torch.zeros(B, N, device=x.device, dtype=torch.bool)

        qkv = rearrange(self.qkv_proj(x), "b n (h d qkv) -> qkv b h n d", qkv=3, h=H, d=D) # qkv projection
    
        out = self.attend(qkv, mask, pos_fn, temperature)

        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.out_proj(out)
        return out

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
            temperature=1.0,
            shared_temperture=True, # NOT USING ALWAYS SHARED
            dropout = 0.,
            checkpoint = False,
        ):
        super().__init__()

        self.temperature = nn.Parameter(torch.tensor(temperature), requires_grad=True)

        self.depth = depth
        self.positional_bias = DynamicPositionBias(
            dim = dim // 4,
            heads = heads,
            depth = 2,
            log_distance = False,
            norm = False
        )
        self.grad_checkpointing = checkpoint
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, CosineAttention(
                    dim, 
                    n_heads=heads, 
                    head_dim=dim_head, 
                    causal=causal,
                    temperature=self.temperature,
                    dropout=dropout
                )),
                PreNorm(dim, self.ff(dim))
            ]))

    @staticmethod
    def ff(dim, mult=4):
        return nn.Sequential(
            GLU(dim, dim * mult, nn.SiLU()),
            nn.Dropout(0.1),
            nn.Linear(dim * mult, dim)
        )

    @staticmethod
    def create_custom_forward(module):
        def custom_forward(*args, **kwargs):
            return module(*args, **kwargs)
        return custom_forward

    def checkpoint(self, layer, module, *args, **kwargs):
        condition = self.training and self.grad_checkpointing and layer < self.depth - 1
        return checkpoint(self.create_custom_forward(module), *args, **kwargs) if condition else module(*args, **kwargs)

    @staticmethod
    def get_sequence_length(mask):
        return (~mask).sum(dim=-1) if mask is not None else None

    def compute_temperature(self, seq_lens):
        temp = self.temperature * torch.log2(torch.pow(seq_lens, 2) - seq_lens)
        return rearrange(temp, 'b -> b () () ()') # broadcast to (b, h, n, n)

    @staticmethod
    def get_sequence_length(x, mask):
        if mask is None:
            return x.shape[1]
        else:
            return torch.sum(~mask, dim=-1)

    def forward(self, x, mask=None):
        for i, (attn, ff) in enumerate(self.layers):
            x = self.checkpoint(i, attn, x, self.positional_bias, mask, self.compute_temperature(self.get_sequence_length(x, mask))) + x
            x = self.checkpoint(i, ff, x) + x
        return x


class transformer_lm(nn.Module):
    def __init__(
        self,
        dim,
        vocab_size,
        depth,
        heads,
        dim_head,
        causal=True,
        temperature=15.0,
        dropout=0.,
        shared_temperture=True,
        **kwargs
    ):
        super().__init__()

        self.layers = transformer(
            dim = dim, 
            depth = depth, 
            heads = heads, 
            dim_head = dim_head, 
            causal = causal, 
            dropout = dropout,
            temperature = temperature,
            shared_temperture = shared_temperture
        )
 
        self.to_logits = nn.Linear(dim, vocab_size)
        self.embedding = nn.Embedding(vocab_size, dim)
        self.post_norm = nn.LayerNorm(dim)

    def forward(self, x, mask=None):
        x = self.embedding(x)
        x = self.layers(x, mask=~mask if mask is not None else None)
        x = self.post_norm(x)
        x = self.to_logits(x)
        return x
