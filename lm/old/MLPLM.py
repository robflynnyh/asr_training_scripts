import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
from einops import rearrange, repeat
from torch import einsum
from torch.utils.checkpoint import checkpoint # # gradient/activation checkpointing
from functools import partial





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
            checkpoint = True,
            **kwargs
        ):
        super().__init__()
        if depth == 1:
            intermediate_loss = False

        ff_mult = kwargs.get('ff_mult', 4)
     

        self.intermediate_loss = intermediate_loss

        self.depth = depth
 
        self.grad_checkpointing = checkpoint
        self.MLP_fnet = PreNorm(dim, MLPAttenion(
            dim, 
            n_heads=heads, 
            head_dim=dim_head, 
            causal=causal,
            dropout=dropout,
            **kwargs
        ))
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                self.MLP_fnet,
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
        condition = self.training and self.grad_checkpointing and layer < self.depth - 1
        return checkpoint(self.create_custom_forward(module), *args, **kwargs) if condition else module(*args, **kwargs)


    def forward(self, x, mask=None, self_condtioning=None):
        intermediate_logits = []
        for i, (attn, ff) in enumerate(self.layers):
            x = self.checkpoint(i, attn, x, mask) + x
            x = self.checkpoint(i, ff, x) + x

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
        return self_condition if (self.self_conditioning or self.intermediate_loss) and self.training else None


    def forward(self, x, mask=None):
        x = self.embedding(x)
        x = self.abs_pos(x)
        x, interim_logits = self.layers(x, mask=~mask if mask is not None else None, self_condtioning=self.self_condition_fn())
        x = self.post_norm(x)
        x = self.to_logits(x)

        return  { 'out': x, 'interim_logits': interim_logits } if self.training else x




def unpad(x, padding_length):
    """
    Undo padding.
    """
    if padding_length > 0:
        return x[:, :-padding_length]
    return x

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

class MLPAttenion(nn.Module):
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

        self.in_proj = nn.Linear(n_feats, n_heads * head_dim)

        window_sizes = [4, 8, 16, 32]
        self.RS_layers = nn.ModuleList(
            [recurrent_shift(dim_head=head_dim, window_size=ws, n_heads=n_heads, dropout=dropout) for ws in window_sizes]
        )
        self.out_proj = nn.Linear(n_heads * head_dim, n_feats)


    def forward(self, x, mask=None):
        B, N, C, H, D = *x.shape, self.n_heads, self.head_dim
       
        if mask is None:
            mask = torch.zeros(B, N, device=x.device, dtype=torch.bool)

        x = self.in_proj(x)
        x = rearrange(x, 'b n (h d) -> b h n d', h=H)       
    
        #mask = rearrange(mask, 'b n -> b () n ()')
        for rs in self.RS_layers:
            x = rs(x, mask=mask)

        out = rearrange(x, 'b h n d -> b n (h d)')

        out = self.out_proj(out)
      
       
        return out

def pad_to_window_size(x, window_size, axis=3, mask=None):
    """
    Pad the input on two sides to be divisible by `window_size`
    """
    batch_size, heads, sequence_length, hidden_size = x.shape
    if sequence_length % window_size == 0:
        return x, 0, mask
    padding_length = (window_size - sequence_length % window_size) % window_size
    padding = torch.zeros(batch_size, heads, padding_length, hidden_size,
        device=x.device,
        dtype=x.dtype,
    )
    mask = F.pad(mask, (0, padding_length), value=True) if mask is not None else None
    return torch.cat([x, padding], axis=axis), padding_length, mask

class OffsetScale(nn.Module):
    def __init__(self, dim, heads = 8):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(1, heads, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, heads, 1, 1, dim))
        nn.init.normal_(self.gamma, std = 0.02)

    def forward(self, x):
        return x * self.gamma + self.beta

class FNetBlock(nn.Module):
  def __init__(self):
    super().__init__()

  def forward(self, x):
    x = torch.fft.fft(torch.fft.fft(x, dim=-1), dim=-2).real
    return x



class recurrent_shift(nn.Module):
    def __init__(self, dim_head, n_heads, window_size, dropout=0.1, bias=True):
        super().__init__()
        self.dim = dim_head
        self.dropout = nn.Dropout(dropout)
        self.bias = bias
        self.WINDOW_SIZE = window_size
     

        self.project_out = nn.Linear(self.dim * 2, self.dim, bias=bias)

        self.fnet = FNetBlock()
        self.zero_vector = torch.zeros([1, n_heads, 1, 1, self.dim])

        self.v_offset = OffsetScale(dim_head, n_heads)

 

    def forward(self, x, mask):
        WINDOW_SIZE = self.WINDOW_SIZE
      
        x, pad_n, new_mask = pad_to_window_size(x, window_size=WINDOW_SIZE, axis=-2, mask=mask) 

       
        x = rearrange(x, 'b h (w n) d -> b h w n d', w=x.shape[-2]// WINDOW_SIZE, n=WINDOW_SIZE) # group into windows
        v = self.v_offset(x)
        v_mask = rearrange(new_mask, 'b (w n) -> b 1 w n', w=new_mask.shape[-1]// WINDOW_SIZE, n=WINDOW_SIZE) # group into windows
        v = self.fnet(v)
        v.masked_fill_(v_mask.unsqueeze(-1), 0)
     
       
        zero_vector = self.zero_vector.to(v.device).expand(v.shape[0], -1, -1, v.shape[-2], -1)
    
        v = torch.cat([zero_vector, v], dim=-3)[:,:,:-1] # shift by one
      
        x = torch.cat([x, v], dim=-1)
       
        x = self.project_out(x)
        x = rearrange(x, 'b h w n d -> b h (w n) d')
        
        if pad_n > 0:
            x = x[:,:,:-pad_n]

        x.masked_fill_(rearrange(mask, 'b n -> b () n ()'), 0)
        return x
        