import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
import numpy as np
from einops import rearrange, repeat
from torch import einsum
from torch.utils.checkpoint import checkpoint # # gradient/activation checkpointing

from torch_scatter import scatter # can remove this dependency if you want (used by learnable_myopic_OLD)

class GumbelSigmoid(nn.Module):
    """
    adapted : https://github.com/yandexdataschool/gumbel_lstm/blob/master/gumbel_sigmoid.py

    A gumbel-sigmoid nonlinearity with gumbel(0,1) noize
    In short, it's a function that mimics #[a>0] indicator where a is the logit
    
    Explaination and motivation: https://arxiv.org/abs/1611.01144
    
    Math:
    Sigmoid is a softmax of two logits: a and 0
    e^a / (e^a + e^0) = 1 / (1 + e^(0 - a)) = sigm(a)
    
    Gumbel-sigmoid is a gumbel-softmax for same logits:
    gumbel_sigm(a) = e^([a+gumbel1]/t) / [ e^([a+gumbel1]/t) + e^(gumbel2/t)]
    where t is temperature, gumbel1 and gumbel2 are two samples from gumbel noize: -log(-log(uniform(0,1)))
    gumbel_sigm(a) = 1 / ( 1 +  e^(gumbel2/t - [a+gumbel1]/t) = 1 / ( 1+ e^(-[a + gumbel1 - gumbel2]/t)
    gumbel_sigm(a) = sigm([a+gumbel1-gumbel2]/t)
    
    For computation reasons:
    gumbel1-gumbel2 = -log(-log(uniform1(0,1)) +log(-log(uniform2(0,1)) = -log( log(uniform2(0,1)) / log(uniform1(0,1)) )
    gumbel_sigm(a) = sigm([a-log(log(uniform2(0,1))/log(uniform1(0,1))]/t)
    
    
    :param t: temperature of sampling. Lower means more spike-like sampling. Can be symbolic.
    :param eps: a small number used for numerical stability
    :returns: a callable that can (and should) be used as a nonlinearity
    
    """
    def __init__(
            self, 
            init_temperature=1.0,
            final_temperature=0.1,
            total_steps=10000, 
            eps=1e-20,
            
        ):
        super().__init__()
        self.temperature=init_temperature
        self.final_temperature=final_temperature
        self.total_steps=total_steps
        self.decay = (final_temperature/init_temperature)**(1/total_steps)
        self.eps=eps

    def step(self):
        if self.temperature > self.final_temperature:
            self.temperature *= self.decay
         
    def __call__(self,logits):
        """computes a gumbel sigmoid sample"""
        temperature = self.temperature if self.training else self.final_temperature
        
        #sample from Gumbel(0, 1)
        uniform1 = torch.rand_like(logits)
        uniform2 = torch.rand_like(logits)
        
        noise = -torch.log(torch.log(uniform2 + self.eps)/torch.log(uniform1 + self.eps) +self.eps)
        #draw a sample from the Gumbel-Sigmoid distribution
        gumbel = ((logits + noise) / temperature).sigmoid()
        if self.training:
            self.step()
        return gumbel
        



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


class SequenceMaskingAttention(nn.Module):
    def __init__(
        self,
        n_feats,
        head_dim,
        n_heads,
        dropout=0.0,
        sequence_dropout=0.1,
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

        self.scale = head_dim ** -0.5

        self.sequence_dropout = sequence_dropout

        self.activation = ReLUSquared() if activation == 'relusq' else nn.Softmax(dim=-1)


        self.qkv_proj = nn.Linear(n_feats, 3 * n_heads * head_dim, bias=bias)
        self.out_proj = nn.Linear(n_heads * head_dim, n_feats, bias=bias)

    def standard_forward(self, qkv, mask, pos_fn): 
        query, key, value = qkv
        dots = torch.einsum('bhid,bhjd->bhij', query, key) * self.scale
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
        

    def forward(self, x, pos_fn, mask=None, return_attention=False, standard_attention=False):
        assert pos_fn is not None, 'pls provide a position function'
        B, N, C, H, D = *x.shape, self.n_heads, self.head_dim
        #print(x.shape, mask.shape)

        standard_attention = standard_attention if self.training else True # we don't want to use dropout during inference

        if mask is None:
            mask = torch.zeros(B, N, device=x.device, dtype=torch.bool)

        qkv = rearrange(self.qkv_proj(x), "b n (h d qkv) -> qkv b h n d", qkv=3, h=H, d=D) # qkv projection
        if not standard_attention:
            q, k, v = qkv
            # get a random sequence of zero or ones that is a quarter of the sequence length
            seq_mask = torch.randint(0, 100, (B, H, max(N // 4,1)), device=x.device, dtype=torch.int8) < (self.sequence_dropout * 100)
            seq_mask = seq_mask.repeat_interleave(5, dim=-1)[..., :N] # repeat the mask to be the same length as the sequence
            
            if seq_mask.shape[-1] != N:
                diff = N - seq_mask.shape[-1]
                pad_seq = torch.randint(0, 100, (B, H, diff), device=x.device, dtype=torch.int8) < (self.sequence_dropout * 100)
                seq_mask = torch.cat((seq_mask, pad_seq), dim=-1)


            dots = einsum("b h i d, b h j d -> b h i j", q, k) * self.scale # dot product between the queries and keys
            
            # positional stuff
            pos_bias = pos_fn(N, device=dots.device, dtype=dots.dtype)
            dots = dots + pos_bias

    
            qkmask = ~mask
            attn_mask = ~(rearrange(qkmask, "b n -> b () n ()") * rearrange(qkmask, "b n -> b () () n"))
            
            if self.causal:
                causal_mask = torch.ones(dots.shape[-2], dots.shape[-1], device=dots.device).triu(1).bool()
                attn_mask = torch.logical_or(attn_mask, causal_mask)
            mask_val = -torch.finfo(dots.dtype).max

            dots.masked_fill_(attn_mask, mask_val)

            attn = self.activation(dots)
            
            attn = attn.masked_fill(rearrange(seq_mask, "b h n -> b h () n"), 0) # apply the sequence mask

            attn = self.dropout(attn)
            out = einsum("bhij,bhjd->bhid", attn, v)
        else:
            out = self.standard_forward(qkv, mask, pos_fn)


        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.out_proj(out)
        return out if not return_attention else (out, attn)

class SequenceDropoutAttention(nn.Module):
    def __init__(
        self,
        n_feats,
        head_dim,
        n_heads,
        dropout=0.0,
        sequence_dropout=0.4,
        bias=False,
        return_attention=False,
        causal=False,
        activation='softmax',
        just_mask=True, # if true only mask the keys don't remove them
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

        self.scale = head_dim ** -0.5

        self.sequence_dropout = sequence_dropout

        self.activation = ReLUSquared() if activation == 'relusq' else nn.Softmax(dim=-1)

        self.just_mask = just_mask

        self.qkv_proj = nn.Linear(n_feats, 3 * n_heads * head_dim, bias=bias)
        self.out_proj = nn.Linear(n_heads * head_dim, n_feats, bias=bias)

    def standard_forward(self, qkv, mask, pos_fn): 
        query, key, value = qkv
        dots = torch.einsum('bhid,bhjd->bhij', query, key) * self.scale
        dots += pos_fn(dots.shape[-1], device=dots.device, dtype=dots.dtype)
        attn_mask = rearrange(mask, "b n -> b () n ()") * rearrange(mask, "b n -> b () () n")
    
        if self.causal: # create a regular causal mask
            causal_mask = torch.ones(dots.shape[-2], dots.shape[-1], device=dots.device).triu(1).bool()
            attn_mask = torch.logical_or(attn_mask, causal_mask)
        
        dots.masked_fill_(attn_mask, -torch.finfo(dots.dtype).max)
    
        attn = self.activation(dots)   
        attn = self.dropout(attn)
        return torch.einsum("bhij,bhjd->bhid", attn, value)
        

    def forward(self, x, pos_fn, mask=None, return_attention=False, standard_attention=False):
        assert pos_fn is not None, 'pls provide a position function'
        B, N, C, H, D = *x.shape, self.n_heads, self.head_dim

        standard_attention = standard_attention if self.training else True # we don't want to use dropout during inference

        if mask is None:
            mask = torch.zeros(B, N, device=x.device, dtype=torch.bool)

        qkv = rearrange(self.qkv_proj(x), "b n (h d qkv) -> qkv b h n d", qkv=3, h=H, d=D) # qkv projection
        if not standard_attention:
            q, kv = qkv[0], qkv[1:] # separate q and kv, we keep kv together for now as we apply the same operations to both
            # get a random sequence of zero or ones that is a quarter of the sequence length
            seq_mask = torch.randint(0, 10, (B, H, max(N // 4,1)), device=x.device, dtype=torch.int8) 
            seq_mask = seq_mask.repeat_interleave(5, dim=-1)[..., :N] # repeat the mask to be the same length as the sequence
            if seq_mask.shape[-1] != N:
                seq_mask = F.pad(seq_mask, (0, N - seq_mask.shape[-1]), value=4) # do this better
            k = int(N * (1 - self.sequence_dropout)) # get the number of tokens to keep
            # mask out seq_mask to prioritize keeping non-masked values
            seq_mask = seq_mask.masked_fill(mask.unsqueeze(1), -1)
            keep_indices = torch.topk(seq_mask, k, dim=-1, largest=True, sorted=False).indices.sort(dim=-1).values # get the indices to keep
            _, kv_B, kv_H, kv_N, kv_D = kv.shape
            # get the keys and values to keep
            kv = kv.gather(-2, repeat(keep_indices, "b h n -> kv b h n d", kv=2, d=D)) # get the keys and values to keep
        
            k_mask = repeat(mask, "b n -> b h n", h=H).gather(-1, keep_indices) # get the mask for the keys
            k, v = kv # separate the keys and values

            dots = einsum("b h i d, b h j d -> b h i j", q, k) * self.scale # dot product between the queries and keys
            
            # positional stuff
            pos_bias = pos_fn(N, device=dots.device, dtype=dots.dtype)
            pos_bias = repeat(pos_bias, 'h i j -> b h i j', b = B)
            keep_indices = repeat(keep_indices, "b h n -> b h i n", i=N)
            pos_bias = pos_bias.gather(-1, keep_indices)
            dots = dots + pos_bias
        
            qk_mask = rearrange(mask, "b n -> b () n ()") * rearrange(k_mask, "b h n -> b h () n")
            
            if self.causal:
                causal_mask = keep_indices > rearrange(torch.arange(0, N, device=q.device), "n -> n ()", n=N)
                qk_mask = torch.logical_or(qk_mask, causal_mask)
            
            dots.masked_fill_(qk_mask, -torch.finfo(dots.dtype).max)
            attn = self.activation(dots)
            attn = self.dropout(attn)
            out = einsum("bhij,bhjd->bhid", attn, v)
        else:
            out = self.standard_forward(qkv, mask, pos_fn)


        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.out_proj(out)
        return out if not return_attention else (out, attn)




class MyopicAttention(nn.Module):
    def __init__(
        self,
        n_feats,
        head_dim,
        n_heads,
        dropout=0.0,
        max_keep_keys=256,
        chunk_window=48,
        bias=False,
        return_attention=False,
        causal=False,
    ):
        super().__init__()
        self.n_feats = n_feats
        self.head_dim = head_dim
        self.n_heads = n_heads
        self.dropout = nn.Dropout(dropout)
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

    @staticmethod
    def causal_windowed_mask(window_number, window_size, device):
        return torch.ones(window_number, window_number, device=device).triu(1).bool().repeat_interleave(window_size, dim=1)


    def standard_forward(self, qkv, mask, pos_fn): 
        query, key, value = qkv
        dots = torch.einsum('bhid,bhjd->bhij', query, key) * self.scale
        dots += pos_fn(dots.shape[-1], device=dots.device, dtype=dots.dtype)
        attn_mask = rearrange(mask, "b n -> b () n ()") * rearrange(mask, "b n -> b () () n")
    
        if self.causal: # create a regular causal mask
            causal_mask = torch.ones(dots.shape[-2], dots.shape[-1], device=dots.device).triu(1).bool()
            attn_mask = torch.logical_or(attn_mask, causal_mask)
        
        dots.masked_fill_(attn_mask, -torch.finfo(dots.dtype).max)
    
        attn = dots.softmax(dim=-1)
        
        attn = self.dropout(attn)
        

        out = torch.einsum('bhij,bhjd->bhid', attn, value)
        return out

    def forward(self, x, pos_fn, mask=None, return_attention=False, standard_attention=False):
        assert pos_fn is not None, 'pls provide a position function'
        B, N, C, H, D = *x.shape, self.n_heads, self.head_dim
        if mask is None:
            mask = torch.zeros(B, N, device=x.device, dtype=torch.bool)

        tokeep = min(self.max_keep_keys, N) if self.max_keep_keys != -1 else N # number of keys to keep
        W = min(self.W, N) if self.W != -1 else N # window size
        standard_attention = standard_attention if (W != N and tokeep != N) else True # this is equivalent to standard attention if W&tokeep == N so we can just use standard attention to save unnecessary computation

        qkv = rearrange(self.qkv_proj(x), "b n (h d qkv) -> qkv b h n d", qkv=3, h=H, d=D) # qkv projection

        if standard_attention == False:
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
                cmask = torch.logical_or(cmask, causal_mask)
            
            chunkgrid = chunkgrid.masked_fill(cmask, torch.finfo(q.dtype).max) # max cus we topk in reverse order 

            keep_indices = chunkgrid.topk(k=tokeep, dim=-1, sorted=False, largest=False).indices.sort(dim=-1).values
            KV, B, H, NW, N, D = kv.shape 
            kv = kv.gather(-2, repeat(keep_indices, "kv b h w n -> kv b h w n d", d=D))

            kv_mask = repeat(mask, "b n -> b h nw n", h=H, nw=NW)
        
            kv_mask = kv_mask.gather(-1, keep_indices[0])

            k, v = kv
            # nw (number of windows) = p (in the einsum below)
            dots = einsum("b h n p d, b h n z d -> b h n p z ", q, k) * self.scale # Z is number of chunks in Q, N is max sequence length after dropping

            ## positional stuff
            pos_bias = pos_fn(N, device=dots.device, dtype=dots.dtype)
            pos_bias = repeat(pos_bias, 'h i j -> b h i j', b = B)
            pos_bias = rearrange(pos_bias, 'b h (n w) j -> b h n w j', w = W)

            keep_indices = repeat(keep_indices, "kv b h nw n -> kv b h nw w n", w=W)[0] 
            pos_bias = pos_bias.gather(-1, keep_indices)
            
            dots = dots + pos_bias

            qk_mask = rearrange(q_mask, "b h n w -> b h n w ()") * rearrange(kv_mask, "b h w n -> b h w () n")

            if self.causal:
                causal_mask = keep_indices > rearrange(torch.arange(0, N, device=q.device), "(nw w) -> nw w ()", w=W, nw=NW)
                qk_mask = torch.logical_or(qk_mask, causal_mask)
        
            dots.masked_fill_(qk_mask, -torch.finfo(dots.dtype).max)

            attn = dots.softmax(dim=-1)
            attn = self.dropout(attn)
    
            out = einsum("b h n w z, b h n z d -> b h n w d", attn, v)
            out = rearrange(out, "b h n w d -> b (n w) (h d)")
            out = self.unpad(out, pad_n)
           
        else:
            out = self.standard_forward(qkv=qkv, mask=mask, pos_fn=pos_fn)
            out = rearrange(out, "b h n d -> b n (h d)")
        
        out = self.out_proj(out)
        return out if not return_attention else (out, attn)



class LearnableMyopicAttentionOLD(nn.Module):
    def __init__(
        self,
        n_feats,
        head_dim,
        n_heads,
        dropout=0.0,
        max_keep_keys=256,
        chunk_window=48,
        bias=False,
        return_attention=False,
        causal=False,
        **kwargs
    ):
        super().__init__()
        self.n_feats = n_feats
        self.head_dim = head_dim
        self.n_heads = n_heads
        self.dropout = nn.Dropout(dropout)
        self.bias = bias
        self.return_attention = return_attention

        self.causal = causal

        self.scale = head_dim ** -0.5

        self.max_keep_keys = max_keep_keys
        self.W = chunk_window


        self.half_precision_mode = kwargs.get("half_precision_mode", 'float32') # 'float32' or 'float16'

        self.rate = nn.Parameter(torch.tensor(kwargs.get('rate', 0.1), requires_grad=True))
     
        distance_multiplier_prior = kwargs.get('distance_multiplier_prior', 1.0)
        self.distance_multiplier = nn.Parameter(torch.tensor(distance_multiplier_prior), requires_grad=True)


        self.qkv_proj = nn.Linear(n_feats, 3 * n_heads * head_dim, bias=bias)
        self.out_proj = nn.Linear(n_heads * head_dim, n_feats, bias=bias)


    def pad_to_window_size(self, x, window_size, axis=3, mask=None):
        """
        Pad the input on two sides to be divisible by `window_size`
        """
        QKV, batch_size, heads, sequence_length, hidden_size = x.shape
        if sequence_length % window_size == 0:
            return x, 0, mask
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
        return chunk_grid 

    @staticmethod
    def causal_windowed_mask(window_number, window_size, device):
        return torch.ones(window_number, window_number, device=device).triu(1).bool().repeat_interleave(window_size, dim=1)


    def standard_forward(self, qkv, mask, pos_fn):
        query, key, value = qkv
        dots = torch.einsum('bhid,bhjd->bhij', query, key) * self.scale
        dots += pos_fn(dots.shape[-1], device=dots.device, dtype=dots.dtype)
        attn_mask = rearrange(mask, "b n -> b () n ()") * rearrange(mask, "b n -> b () () n")
    
        if self.causal: # create a regular causal mask
            causal_mask = torch.ones(dots.shape[-2], dots.shape[-1], device=dots.device).triu(1).bool()
            attn_mask = torch.logical_or(attn_mask, causal_mask)
        
        dots.masked_fill_(attn_mask, -torch.finfo(dots.dtype).max)
    
        attn = dots.softmax(dim=-1)
        
        attn = self.dropout(attn)

        out = torch.einsum('bhij,bhjd->bhid', attn, value)
        return out

    def half_precision_if_on_cuda(self, x, is_cuda):
        if not is_cuda:
            return x
        elif self.half_precision_mode == 'float16':
            return x.half()
        else:
            return x

    def forward(self, x, pos_fn, mask=None, return_attention=False, standard_attention=False):
        assert pos_fn is not None, 'pls provide a position function'
        B, N, C, H, D = *x.shape, self.n_heads, self.head_dim
        if mask is None:
            mask = torch.zeros(B, N, device=x.device, dtype=torch.bool)

        tokeep = min(self.max_keep_keys, N) if self.max_keep_keys != -1 else N # number of keys to keep
        W = min(self.W, N) if self.W != -1 else N # window size

        standard_attention = standard_attention if (W != N and tokeep != N) else True # this is equivalent to standard attention if W&tokeep == N so we can just use standard attention to save unnecessary computation

        qkv = rearrange(self.qkv_proj(x), "b n (h d qkv) -> qkv b h n d", qkv=3, h=H, d=D) # qkv projection

        if standard_attention == False:
            qkv, pad_n, mask = self.pad_to_window_size(qkv, W, axis=3, mask=mask) # add padding so it's divisible by W

            q, kv = qkv[0], qkv[1:] # separate q and kv, we keep kv together for now as we apply the same operations to both
            
            q = rearrange(q, "b h (n w) d -> b h n w d", w=W)# split q into windows/chunks of size W
        
            q_mask = repeat(rearrange(mask, "b (n w) -> b n w", w=W), "b n w -> b h n w", h=H) # do the same for the mask
                
            kv = repeat(kv, "kv b h n d -> kv b h nw n d", nw=q.shape[2]) # duplicate k and v for total number of windows
            #print(q.shape, kv.shape)
            KV, B, H, NW, N, D = kv.shape

            chunkgrid = self.ChunkGrid(Total_Size=N, Block_Size=W).to(q.device)
            chunkgrid = self.half_precision_if_on_cuda(chunkgrid, q.is_cuda)
         
            chunkgrid = repeat(chunkgrid, "w n -> b h w n", b=B, h=H).contiguous() # duplicate chunkgrid for batch and heads
            distance_multiplier = self.half_precision_if_on_cuda(self.distance_multiplier, q.is_cuda).to(chunkgrid.device).abs()
            chunkgrid = chunkgrid * distance_multiplier # apply distance multiplier
            
            RATE = self.rate.to(chunkgrid.device).abs() # can only be positive
            exp_dist = torch.distributions.exponential.Exponential(RATE).rsample(chunkgrid.shape).to(chunkgrid.dtype).to(chunkgrid.device)
            chunkgrid = chunkgrid - exp_dist
            
            chunkgrid = repeat(chunkgrid, "b h w n -> kv b h w n", kv=2)

            cmask = repeat(mask, 'b n -> kv b h nw n', kv=2, h=H, nw=NW)

            if self.causal:
                causal_mask = self.causal_windowed_mask(window_number=NW, window_size=W, device=q.device)
                cmask = torch.logical_or(cmask, causal_mask)
           
            chunkgrid = chunkgrid.masked_fill(cmask, torch.finfo(chunkgrid.dtype).max) # max cus we topk in reverse order 
        
         
            keep_indices = chunkgrid.topk(k=tokeep, dim=-1, sorted=False, largest=False)
            if self.training: # only do this during training
                '''
                we want to take a half of the keep indices (the ones with the largest vals) and apply a softmax to the values
                then scatter the values with a multiply reduction to the k tensor
                this allows the model to learn which keys to keep using the parametized pareto distribution (bcos sequence level dropout is not differentiable)
                so kinda works like a relu (but not really)
                '''
                sorted_vals_with_indices = keep_indices.values.sort(-1)
                num_to_scatter = tokeep // 3  # number of keys to scatter to kv
                if num_to_scatter > 0:
                    scatter_indices = sorted_vals_with_indices.indices[..., -num_to_scatter:].long() # indices of the keys to scatter
                    scatter_indices = repeat(scatter_indices[0], 'b h w n -> b h w n d', d=D)
                    scatter_vals = sorted_vals_with_indices.values[..., -num_to_scatter:] # values of the keys to scatter
                    scatter_vals = scatter_vals[0].softmax(-1) * -1 + 1 # softmax but we want the smallest values to be the largest
                    scatter_vals = repeat(scatter_vals, 'b h w n -> b h w n d', d=D)
               
                    kv = kv.contiguous() 
                    kv[0] = scatter(
                        src = scatter_vals,
                        index = scatter_indices,
                        dim = -2,
                        out = self.half_precision_if_on_cuda(kv[0], kv.is_cuda),
                        reduce = 'mul'
                    )
                    del scatter_indices, scatter_vals 

            keep_indices = keep_indices.indices.sort(dim=-1).values
            
            KV, B, H, NW, N, D = kv.shape 
            kv = kv.gather(-2, repeat(keep_indices, "kv b h w n -> kv b h w n d", d=D))

            kv_mask = repeat(mask, "b n -> b h nw n", h=H, nw=NW)
        
            kv_mask = kv_mask.gather(-1, keep_indices[0])

            k, v = kv
            # nw (number of windows) = p (in the einsum below)
            dots = einsum("b h n p d, b h n z d -> b h n p z ", q, k) * self.scale # Z is number of chunks in Q, N is max sequence length after dropping

            ## positional stuff
            pos_bias = pos_fn(N, device=dots.device, dtype=dots.dtype)
            pos_bias = repeat(pos_bias, 'h i j -> b h i j', b = B)
            pos_bias = rearrange(pos_bias, 'b h (n w) j -> b h n w j', w = W)

            keep_indices = repeat(keep_indices, "kv b h nw n -> kv b h nw w n", w=W)[0] 
            pos_bias = pos_bias.gather(-1, keep_indices)
            
            dots = dots + pos_bias

            qk_mask = rearrange(q_mask, "b h n w -> b h n w ()") * rearrange(kv_mask, "b h w n -> b h w () n")

            if self.causal:
                causal_mask = keep_indices > rearrange(torch.arange(0, N, device=q.device), "(nw w) -> nw w ()", w=W, nw=NW)
                qk_mask = torch.logical_or(qk_mask, causal_mask)
        
            dots.masked_fill_(qk_mask, -torch.finfo(dots.dtype).max)

            attn = dots.softmax(dim=-1)
            attn = self.dropout(attn)
    
            out = einsum("b h n w z, b h n z d -> b h n w d", attn, v)
            out = rearrange(out, "b h n w d -> b (n w) (h d)")
            out = self.unpad(out, pad_n)
        else:
            out = self.standard_forward(qkv=qkv, mask=mask, pos_fn=pos_fn)
            out = rearrange(out, "b h n d -> b n (h d)")
        
        out = self.out_proj(out)
        return out if not return_attention else (out, attn)

class LearnableMyopicAttention(nn.Module):
    def __init__(
        self,
        n_feats,
        head_dim,
        n_heads,
        dropout=0.0,
        max_keep_keys=256,
        chunk_window=48,
        bias=False,
        return_attention=False,
        causal=False,
    ):
        super().__init__()
        self.n_feats = n_feats
        self.head_dim = head_dim
        self.n_heads = n_heads
        self.dropout = nn.Dropout(dropout)
        self.bias = bias
        self.return_attention = return_attention

        self.causal = causal

        self.scale = head_dim ** -0.5

        self.max_keep_keys = max_keep_keys
        self.W = chunk_window

        self.grid_pos_projection = nn.Linear(1, head_dim*n_heads)
        #self.grid_k_projection = nn.Linear(head_dim, head_dim)
        self.grid_k_bias = nn.Parameter(torch.zeros(n_heads, 1, 1, head_dim))
        nn.init.xavier_uniform_(self.grid_k_bias)
        self.grid_activation = nn.SiLU()
        self.grid_scaler_projection = nn.Linear(head_dim, 1)
        

        anneal_steps = 10000
        initial_temp = 1.0
        final_temp = 0.1

        self.gumbel_sigmoid = GumbelSigmoid(
            init_temperature = initial_temp,
            final_temperature = final_temp,
            total_steps = anneal_steps
        ) 

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

    def valuegrid(self, total_size, block_size, k):
        
        n = total_size // block_size
        device, dtype = k.device, k.dtype
  
        indices = (rearrange(torch.arange(n, device = device), 'i -> i 1') - rearrange(torch.arange(n, device = device), 'j -> 1 j')) + (n - 1)
        pos = torch.arange(-n + 1, n, device = device, dtype = torch.float32)
        pos = rearrange(pos, '... -> ... 1')
        pos = self.grid_pos_projection(pos)[indices]
    
        pos = pos.repeat_interleave(block_size, dim=1)
        pos = rearrange(pos, "p n (h d) -> () h p n d", h = self.n_heads, d = self.head_dim)
    
        k_voting = k + self.grid_k_bias
        k_voting = k_voting + pos
        k_voting = self.grid_activation(k_voting)
        k_voting = self.grid_scaler_projection(k_voting).squeeze(-1)
        k_voting = k_voting / k_voting.sum(dim=-1, keepdim=True)
      
        #k_voting = torch.randn((k.shape[0], k.shape[1], k.shape[2], k.shape[3]), device=k.device, dtype=k.dtype)
        #k_voting = k_voting / k_voting.sum(dim=-1, keepdim=True)
        k_voting = self.gumbel_sigmoid(k_voting)
        #print((k_voting[0,0,0] < 0.1).sum() / k_voting.shape[-1])
        return k_voting    
   

    @staticmethod
    def causal_windowed_mask(window_number, window_size, device):
        return torch.ones(window_number, window_number, device=device).triu(1).bool().repeat_interleave(window_size, dim=1)


    def standard_forward(self, qkv, mask, pos_fn): # USE GUMBEL SOFTMAX
        query, key, value = qkv
        dots = torch.einsum('bhid,bhjd->bhij', query, key) * self.scale
        dots += pos_fn(dots.shape[-1], device=dots.device, dtype=dots.dtype)
        attn_mask = rearrange(mask, "b n -> b () n ()") * rearrange(mask, "b n -> b () () n")
    
        if self.causal: # create a regular causal mask
            causal_mask = torch.ones(dots.shape[-2], dots.shape[-1], device=dots.device).triu(1).bool()
            attn_mask = torch.logical_or(attn_mask, causal_mask)
        
        dots.masked_fill_(attn_mask, -torch.finfo(dots.dtype).max)
    
        attn = dots.softmax(dim=-1)
        
        attn = self.dropout(attn)

        out = torch.einsum('bhij,bhjd->bhid', attn, value)
        return out

    def forward(self, x, pos_fn, mask=None, return_attention=False, standard_attention=False):
        assert pos_fn is not None, 'pls provide a position function'
        B, N, C, H, D = *x.shape, self.n_heads, self.head_dim
        if mask is None:
            mask = torch.zeros(B, N, device=x.device, dtype=torch.bool)

        tokeep = min(self.max_keep_keys, N) if self.max_keep_keys != -1 else N # number of keys to keep
        W = min(self.W, N) if self.W != -1 else N # window size
        standard_attention = standard_attention if (W != N and tokeep != N) else True # this is equivalent to standard attention if W&tokeep == N so we can just use standard attention to save unnecessary computation

        qkv = rearrange(self.qkv_proj(x), "b n (h d qkv) -> qkv b h n d", qkv=3, h=H, d=D) # qkv projection

        if standard_attention == False:
            qkv, pad_n, mask = self.pad_to_window_size(qkv, W, axis=3, mask=mask) # add padding so it's divisible by W

            q, kv = qkv[0], qkv[1:] # separate q and kv, we keep kv together for now as we apply the same operations to both
            
            q = rearrange(q, "b h (n w) d -> b h n w d", w=W)# split q into windows/chunks of size W
        
            q_mask = repeat(rearrange(mask, "b (n w) -> b n w", w=W), "b n w -> b h n w", h=H) # do the same for the mask
                
            kv = repeat(kv, "kv b h n d -> kv b h nw n d", nw=q.shape[2]) # duplicate k and v for total number of windows
            #print(q.shape, kv.shape)
            KV, B, H, NW, N, D = kv.shape

            valuegrid = self.valuegrid(total_size=N, block_size=W, k=kv[0]).to(kv.device)
            #print((valuegrid[0,0,0] < 0.1).sum() / valuegrid.shape[-1])
            #valuegrid = repeat(valuegrid, "b h w n -> kv b h w n", kv=2)
         
            if self.training:
                kv = torch.cat((kv[0].unsqueeze(0) * valuegrid.unsqueeze(-1), kv[1:]))
            

            cmask = repeat(mask, 'b n -> kv b h nw n', kv=2, h=H, nw=NW)

            if self.causal:
                causal_mask = self.causal_windowed_mask(window_number=NW, window_size=W, device=q.device)
                cmask = torch.logical_or(cmask, causal_mask)
            
            valuegrid = valuegrid.masked_fill(cmask, -torch.finfo(q.dtype).max) 

            keep_indices = valuegrid.topk(k=tokeep, dim=-1, sorted=False, largest=True).indices.sort(dim=-1).values
            KV, B, H, NW, N, D = kv.shape 

            kv = kv.gather(-2, repeat(keep_indices, "kv b h w n -> kv b h w n d", d=D))

            kv_mask = repeat(mask, "b n -> b h nw n", h=H, nw=NW)
        
            kv_mask = kv_mask.gather(-1, keep_indices[0])

            k, v = kv
            # nw (number of windows) = p (in the einsum below)
            dots = einsum("b h n p d, b h n z d -> b h n p z ", q, k) * self.scale # Z is number of chunks in Q, N is max sequence length after dropping

            ## positional stuff
            pos_bias = pos_fn(N, device=dots.device, dtype=dots.dtype)
            pos_bias = repeat(pos_bias, 'h i j -> b h i j', b = B)
            pos_bias = rearrange(pos_bias, 'b h (n w) j -> b h n w j', w = W)

            keep_indices = repeat(keep_indices, "kv b h nw n -> kv b h nw w n", w=W)[0] 
            pos_bias = pos_bias.gather(-1, keep_indices)
            
            dots = dots + pos_bias

            qk_mask = rearrange(q_mask, "b h n w -> b h n w ()") * rearrange(kv_mask, "b h w n -> b h w () n")

            if self.causal:
                causal_mask = keep_indices > rearrange(torch.arange(0, N, device=q.device), "(nw w) -> nw w ()", w=W, nw=NW)
                qk_mask = torch.logical_or(qk_mask, causal_mask)
        
            dots.masked_fill_(qk_mask, -torch.finfo(dots.dtype).max)

            attn = dots.softmax(dim=-1)
            attn = self.dropout(attn)
    
            out = einsum("b h n w z, b h n z d -> b h n w d", attn, v)
            out = rearrange(out, "b h n w d -> b (n w) (h d)")
            out = self.unpad(out, pad_n)
        else:
            out = self.standard_forward(qkv=qkv, mask=mask, pos_fn=pos_fn)
            out = rearrange(out, "b h n d -> b n (h d)")
        
        out = self.out_proj(out)
        return out if not return_attention else (out, attn)


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
            max_keep_keys = -1, 
            W = -1,
            dropout = 0.,
            checkpoint = False,
        ):
        super().__init__()

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
                PreNorm(dim, SequenceMaskingAttention(
                    dim, 
                    n_heads=heads, 
                    head_dim=dim_head, 
                    causal=causal, 
                    max_keep_keys=max_keep_keys, 
                    chunk_window=W,
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

    def forward(self, x, mask=None):
        for i, (attn, ff) in enumerate(self.layers):
            x = self.checkpoint(i, attn, x, self.positional_bias, mask) + x
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
        max_keep_keys=-1,
        W=-1,
        dropout=0.,
        **kwargs
    ):
        super().__init__()
        self.layers = transformer(dim, depth, heads, dim_head, causal=causal, max_keep_keys=max_keep_keys, W=W, dropout=dropout)
 
        self.to_logits = nn.Linear(dim, vocab_size)
        self.embedding = nn.Embedding(vocab_size, dim)
        self.post_norm = nn.LayerNorm(dim)

    def forward(self, x, mask=None):
        x = self.embedding(x)
        x = self.layers(x, mask=~mask if mask is not None else None)
        x = self.post_norm(x)
        x = self.to_logits(x)
        return x
