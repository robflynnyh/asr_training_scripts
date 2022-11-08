import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
from einops import rearrange, repeat
from torch import einsum
from torch.utils.checkpoint import checkpoint # # gradient/activation checkpointing
from functools import partial


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


'''
uses code from Phil Wang's implementation. 
https://github.com/lucidrains/FLASH-pytorch/blob/main/flash_pytorch/flash_pytorch.py

Paper: https://arxiv.org/pdf/2202.10447.pdf

though this is slightly different, I remove the offset scale, as I am using l2 norm for q and k
and I keep softmax with the learnt temperature rather than relu squeared
I will test the proper implementation later but I think this will be better
'''

class CosineGatedAttentionUnit(nn.Module):
    def __init__(
        self,
        n_feats,
        head_dim,
        n_heads = 1,
        dropout=0.1,
        bias=False,
        temperature=15.5,
        return_attention=False,
        causal=False,
        activation='softmax',
        expansion_factor=2,
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

        self.expansion_factor = expansion_factor

        self.norm = nn.LayerNorm(n_feats)
        
        self.to_vgate = nn.Sequential(nn.Linear(n_feats, n_feats * expansion_factor * 2), nn.SiLU())
        self.to_query_key = nn.Sequential(nn.Linear(n_feats, head_dim * n_heads * 2), nn.SiLU())
        self.out_projection = nn.Linear(n_feats * expansion_factor, n_feats)

        self.causal = causal

        self.temperature = torch.nn.Parameter(torch.tensor(temperature), requires_grad=True) if isinstance(temperature, float) else temperature

        self.activation_type = activation
        self.activation = ReLUSquared() if activation == 'relusq' else nn.Softmax(dim=-1)

        self.to_out = nn.Sequential(nn.Linear(n_feats * expansion_factor, n_feats), nn.Dropout(0.1))

    


    def forward(self, x, pos_fn, mask=None):
        assert pos_fn is not None, 'pls provide a position function'
        B, N, C, H, D = *x.shape, self.n_heads, self.head_dim
        if mask is None:
            mask = torch.zeros(B, N, device=x.device, dtype=torch.bool)

        x = self.norm(x)

        v, gate = self.to_vgate(x).chunk(2, dim=-1)
     
        v = rearrange(v, 'b n (h d) -> b h n d', h=H)
     

        q, k = rearrange(self.to_query_key(x), 'b n (h d) -> b h n d', h=H).chunk(2, dim=-1)
        q, k = map(l2norm, (q, k)) # qk norm attention

        dots = einsum('bhid,bhjd->bhij', q, k) * self.temperature
        dots += pos_fn(dots.shape[-1], device=dots.device, dtype=dots.dtype)

        qkmask = ~mask
        attn_mask = ~(rearrange(qkmask, "b n -> b () n ()") * rearrange(qkmask, "b n -> b () () n"))

        if self.causal: # create a regular causal mask
            causal_mask = torch.ones(dots.shape[-2], dots.shape[-1], device=dots.device, dtype=torch.bool).triu(1)
            attn_mask = torch.logical_or(attn_mask, causal_mask)

        dots.masked_fill_(attn_mask, -torch.finfo(dots.dtype).max)

        attn = self.activation(dots) 

    
        out = einsum("bhij,bhjd->bhid", attn, v)
      
        out = rearrange(out, 'b h n d -> b n (h d)')

        out = out * gate

        out = self.to_out(out)
        
        return out



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
            checkpoint = False,
            **kwargs
        ):
        super().__init__()
        if depth == 1:
            intermediate_loss = False

     
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

        self.grad_checkpointing = checkpoint
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                CosineGatedAttentionUnit(
                    dim, 
                    n_heads=heads, 
                    head_dim=dim_head, 
                    causal=causal,
                    temperature=self.temperature,
                    dropout=dropout,
                    **kwargs
                ),
            )

    @staticmethod
    def create_custom_forward(module):
        def custom_forward(*args, **kwargs):
            return module(*args, **kwargs)
        return custom_forward

    def checkpoint(self, layer, module, *args, **kwargs):
        condition = self.training and self.grad_checkpointing and layer < self.depth - 1
        return checkpoint(self.create_custom_forward(module), *args, **kwargs) if condition else module(*args, **kwargs)

    def forward(self, x, mask=None, self_condtioning=None):
        intermediate_logits = []
        for i, attn in enumerate(self.layers):
            #x = attn(x, self.positional_bias, mask=mask) 
            x = self.checkpoint(i, attn, x, self.positional_bias, mask) + x
            
            if i < self.depth - 1 and self_condtioning is not None:
                x, logits = self_condtioning(x)
                intermediate_logits.append(logits)

        # stack intermediate logits
        if len(intermediate_logits) > 0:
            intermediate_logits = torch.stack(intermediate_logits, dim=0) # D x B x N x V

        return x, intermediate_logits


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
        **kwargs
    ):
        super().__init__()
        if depth == 1:
            self_conditioning == False

        self.self_conditioning = True if self_conditioning else None
        self.intermediate_loss = intermediate_loss


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
 
        self.to_logits = nn.Linear(dim, vocab_size)
        self.embedding = nn.Embedding(vocab_size, dim)
        self.post_norm = nn.LayerNorm(dim)

    def self_condition_fn(self):
        def self_condition(x):
            logits = self.to_logits(self.post_norm(x))
            if self.self_conditioning:
                z = F.softmax(logits, dim=-1)
                z = self.reprojection_layer(z)
                x = z + x
            return x, logits
        if (self.self_conditioning or self.intermediate_loss) and self.training:
            return self_condition
        else:
            return None


    def forward(self, x, mask=None):
        x = self.embedding(x)
        x, interim_logits = self.layers(x, mask=~mask if mask is not None else None, self_condtioning=self.self_condition_fn())
        x = self.post_norm(x)
        x = self.to_logits(x)

        return  { 'out': x, 'interim_logits': interim_logits } if self.training else x