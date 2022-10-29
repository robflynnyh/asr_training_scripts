import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
import numpy as np
from einops import rearrange, repeat
from torch import einsum

class DynamicPositionBias(nn.Module):
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

class MyopicAttention(nn.Module):
    def __init__(
        self,
        n_feats,
        head_dim,
        n_heads,
        dropout=0.0,
        max_keep_keys=50,
        chunk_window=3,
        bias=True,
        return_attention=False,
        causal=False,
    ):
        super().__init__()
        self.n_feats = n_feats
        self.head_dim = head_dim
        self.n_heads = n_heads
        self.dropout = dropout
        self.bias = bias
        self.return_attention = return_attention

        self.causal = causal

        self.scale = head_dim ** -0.5

        self.max_keep_keys = max_keep_keys
        self.W = chunk_window


        self.qkv_proj = nn.Linear(n_feats, 3 * n_heads * head_dim, bias=bias)
        self.out_proj = nn.Linear(n_heads * head_dim, n_feats, bias=bias)

    def pad_to_window_size(self, x, window_size, axis=3, mask=None):
        """
        Pad the input on two sides to be divisible by `window_size`
        """
        QKV, batch_size, heads, sequence_length, hidden_size = x.shape
        padding_length = (window_size - sequence_length % window_size) % window_size
        padding = torch.zeros(QKV, batch_size, heads, padding_length, hidden_size,
            device=x.device,
            dtype=x.dtype,
        )
        mask = F.pad(mask, (0, padding_length), value=True) 
        return torch.cat([x, padding], axis=axis), padding_length, mask

    def unpad(self, x, padding_length):
        """
        Undo padding.
        """
        if padding_length > 0:
            return x[:, :-padding_length]
        return x

    def ChunkGrid(self, Total_Size, Block_Size):
        Psize = Total_Size // Block_Size
        chunk_grid = (torch.arange(0, Psize).repeat(Psize,1) - torch.arange(0, Psize).repeat(Psize,1).T ).repeat_interleave(Block_Size, dim=1).abs()
        #chunk_grid = 1 - (chunk_grid / chunk_grid.max(dim=-1)[0].unsqueeze(-1)) # don't normalize cus it'll stretch the distribution by sequence length
        return chunk_grid    

    def causal_windowed_mask(self, window_number, window_size, device):
        return torch.ones(window_number, window_number, device=device).triu(1).bool().repeat_interleave(window_size, dim=1)

    def forward(self, x, mask, pos_fn, return_attention=False):
        assert mask is not None, 'pls wear a mask'
        assert pos_fn is not None, 'pls provide a position function'
        B, N, C, H, D = *x.shape, self.n_heads, self.head_dim

        tokeep = min(self.max_keep_keys, N) if self.max_keep_keys != -1 else N # number of keys to keep
        W = min(self.W, N) if self.W != -1 else N # window size

        qkv = rearrange(self.qkv_proj(x), "b n (h d qkv) -> qkv b h n d", qkv=3, h=H, d=D) # qkv projection

        qkv, pad_n, mask = self.pad_to_window_size(qkv, W, axis=3, mask=mask) # add padding so it's divisible by W
        q, kv = qkv[0], qkv[1:] # separate q and kv, we keep kv together for now as we apply the same operations to both
        
        q = rearrange(q, "b h (n w) d -> b h n w d", w=W)# split q into windows/chunks of size W
      
        q_mask = repeat(rearrange(mask, "b (n w) -> b n w", w=W), "b n w -> b h n w", h=H) # do the same for the mask
            
        kv = repeat(kv, "kv b h n d -> kv b h nw n d", nw=q.shape[2]) # duplicate k and v for total number of windows
        #print(q.shape, kv.shape)
        KV, B, H, NW, N, D = kv.shape

        chunkgrid = self.ChunkGrid(Total_Size=N, Block_Size=W).to(q.device)
        chunkgrid = repeat(chunkgrid, "w n -> b h w n", b=B, h=H).contiguous()

        SCALE = torch.tensor(3.0, device=q.device, dtype=q.dtype)
        ALPHA = torch.tensor(2.0, device=q.device, dtype=q.dtype)
        pareto_dist = torch.distributions.pareto.Pareto(SCALE, ALPHA).sample(chunkgrid.shape).to(q.device)
        chunkgrid = chunkgrid - pareto_dist

        chunkgrid = repeat(chunkgrid, "b h w n -> kv b h w n", kv=2)

        cmask = repeat(mask, 'b n -> kv b h nw n', kv=2, h=H, nw=NW)

        if self.causal:
            causal_mask = self.causal_windowed_mask(window_number=NW, window_size=W, device=q.device)
            cmask = cmask & causal_mask
        
        chunkgrid = chunkgrid.masked_fill(cmask, torch.finfo(q.dtype).max) # max cus we topk in reverse order 

        keep_indices = chunkgrid.topk(k=tokeep, dim=-1, sorted=False, largest=False).indices.sort(dim=-1).values
        KV, B, H, NW, N, D = kv.shape 
        kv = kv.gather(-2, repeat(keep_indices, "kv b h w n -> kv b h w n d", d=D))

        kv_mask = repeat(mask, "b n -> b h nw n", h=H, nw=NW)
        if self.causal:
            kv_mask = kv_mask & causal_mask 
        kv_mask = kv_mask.gather(-1, keep_indices[0])

        k, v = kv
        # nw (number of windows) = p (in the einsum below)
        dots = einsum("b h n p d, b h n z d -> b h n p z ", q, k) * self.scale # Z is number of chunks in Q, N is max sequence length after dropping

        ## positional stuff
        pos_bias = pos_fn(N, device=dots.device, dtype=dots.dtype)
        pos_bias = repeat(pos_bias, 'h i j -> b h i j', b = B)
        pos_bias = rearrange(pos_bias, 'b h (n w) j -> b h n w j', w = W)

        pos_bias = pos_bias.gather(-1, repeat(keep_indices, "kv b h nw n -> kv b h nw w n", w=W)[0])
        
        dots = dots + pos_bias

  
        mask_val = -torch.finfo(dots.dtype).max
        qk_mask = rearrange(q_mask, "b h n w -> b h n w ()") * rearrange(kv_mask, "b h w n -> b h w () n")
        dots.masked_fill_(qk_mask, mask_val)

        attn = dots.softmax(dim=-1)
  
        out = einsum("b h n w z, b h n z d -> b h n w d", attn, v)
        out = rearrange(out, "b h n w d -> b (n w) (h d)")
        out = self.unpad(out, pad_n)
        out = self.out_proj(out)
        return out if not return_attention else (out, attn)


class macaron_transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, causal=True, max_keep_keys=-1, W=-1):
        super().__init__()

        self.positional_bias = DynamicPositionBias(
            dim = dim,
            heads = heads,
            depth = 2,
            log_distance = False,
            norm = False
        )

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                self.ff(dim),
                MyopicAttention(dim, n_heads=heads, head_dim=dim_head, causal=causal, max_keep_keys=max_keep_keys, chunk_window=W),
                self.ff(dim),
                nn.LayerNorm(dim)
            ]))

    @staticmethod
    def ff(dim, mult=4):
        return nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * mult),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(dim * mult, dim),
            nn.Dropout(0.1)
        )

    def forward(self, x, mask=None):
        for pre, attn, post, ln_out in self.layers:
            x = x + pre(x) * 0.5
            x = x + attn(x, mask=mask, pos_fn=self.positional_bias) 
            x = x + post(x) * 0.5
            x = ln_out(x)
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
        max_keep_keys=-1,
        W=-1,
        **kwargs
    ):
        super().__init__()
        self.layers = macaron_transformer(dim, depth, heads, dim_head, causal=causal, max_keep_keys=max_keep_keys, W=W)
        self.to_logits = nn.Linear(dim, vocab_size)
        self.embedding = nn.Embedding(vocab_size, dim)

    def forward(self, x, mask=None):
        x = self.embedding(x)
        x = self.layers(x, mask=mask)
        x = self.to_logits(x)
        return x