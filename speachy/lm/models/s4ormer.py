import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
from einops import rearrange, repeat
from torch import einsum
from torch.utils.checkpoint import checkpoint # # gradient/activation checkpointing
from functools import partial
import string
from typing import Optional, Tuple, List, Dict, Union, Callable
from speachy.lm.tools.train import add_eos, token_lens_to_mask, mark_padding
from .state_space import S4

def exists(val):
    return val is not None

class S4Block(nn.Module):
    def __init__(self, dim, d_state=64, s4_depth=2, dropout=0.1):
        super().__init__()
        self.dim, self.d_state, self.s4_depth = dim, d_state, s4_depth
        self.layers = nn.ModuleList([])
        for _ in range(s4_depth):
            self.layers.append(S4(dim, d_state, dropout=dropout, transposed=False))

    def forward(self, u, lengths=None):
        for layer in self.layers:
            u, _ = layer(u, lengths=lengths)
        return u


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



class s4ormer(nn.Module):
    def __init__(
            self, 
            dim, 
            depth, 
            intermediate_loss=True,
            dropout = 0.1,
            dropout_residual = 0.1,
            **kwargs
        ):
        super().__init__()
        if depth == 1:
            intermediate_loss = False

        ff_mult = kwargs.get('ff_mult', 4)
        self.checkpoint_every_n = kwargs.get('checkpoint_every_n', 0)

        self.dropout_residual = nn.Dropout(dropout_residual)
        self.intermediate_loss = intermediate_loss

        self.depth = depth

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, S4Block(dim=dim, d_state=64, s4_depth=2, dropout=dropout)),
                PreNorm(dim, self.ff(dim, mult=ff_mult, dropout=dropout))
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


    def forward(self, x, length=None, self_condtioning=None):
        intermediate_logits = []
        rd = self.dropout_residual 
        
        for i, (s4, ff) in enumerate(self.layers):
        
            x = self.checkpoint(i, s4, x, length) + rd(x)
            x = self.checkpoint(i, ff, x) + rd(x)   

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


class s4ormer_lm(nn.Module):
    def __init__(
        self,
        dim,
        vocab_size,
        depth,
        dropout=0.,
        self_conditioning=False,
        intermediate_loss=False,
        **kwargs
    ):
        super().__init__()
        if depth == 1:
            self_conditioning == False

        self.self_conditioning = True if self_conditioning else None
        self.intermediate_loss = intermediate_loss

   
        if self_conditioning:
            self.reprojection_layer = nn.Linear(vocab_size, dim)

        self.layers = s4ormer(
            dim = dim, 
            depth = depth, 
            dropout = dropout,
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

    def loss_fn(self, logits, interim_logits, targets, length):
        eos_id = -100
        loss_fn = lambda l, t: F.cross_entropy(rearrange(l, 'b n c -> b c n'), t, ignore_index=-100, reduction='mean')
        interim_losses = []

        def calc_token_loss(logits, targets, length):
            if length.max() == 1:
                return None # no loss for single token sequences if no previous prediction is available
    
            targets[:,:-1] = targets.clone()[:,1:]        
            targets = add_eos(targets, eos_id=eos_id, token_lens=length)
            mask =  token_lens_to_mask(token_lens=length)
            
            targets = mark_padding(targets=targets, mask=mask, pad_id=eos_id)
            loss = loss_fn(logits, targets) #* heads # multiply by heads             
            return loss

        main_loss = calc_token_loss(logits, targets, length)
        if interim_logits is not None:
            for interim_logits_ in interim_logits:
                interim_losses.append(calc_token_loss(interim_logits_, targets, length))
            main_loss = main_loss * 0.5 + sum(interim_losses) * 0.5
        
        return main_loss


    def forward(self, x, length, calc_loss = False, **kwargs):
        '''
        x: [B, N] (embedding indices)
        length: [B] (length of each sequence)
        '''
        if x.shape[1] > length.max():
            x = x[:, :length.max()] # trim x to max length
            
        targets = x.clone() if calc_loss else None
        x = self.embedding(x)
        x, interim_logits = self.layers(x, length, self_condtioning=self.self_condition_fn())
        x = self.post_norm(x)
        x = self.to_logits(x)

        if calc_loss:
            return self.loss_fn(x, interim_logits, targets, length)
        else:
            return  x, interim_logits



class CharacterTokenizer(): # only for testing!
    def __init__(self):
        self.vocab = ['#', '/'] + list(string.ascii_lowercase) + [' '] # bos/eos -> /, pad -> #
        self.vocab_size = len(self.vocab)
        self.token_to_id = {token: i for i, token in enumerate(self.vocab)}
        self.id_to_token = {i: token for i, token in enumerate(self.vocab)}
    
    def __call__(self, text):
        return self.tokenize(text)

    def tokenize(self, text):
        return [self.token_to_id[token] for token in text]

def collate_fn(tensors:List[torch.Tensor], pad_token:int): # only for testing!
    max_len = max([t.shape[0] for t in tensors])
    lengths = torch.tensor([t.shape[0] for t in tensors])
    padded_tensors = [torch.cat([t, torch.full((max_len - t.shape[0],), pad_token, dtype=t.dtype)], dim=0) for t in tensors]
    return torch.stack(padded_tensors, dim=0), lengths


@torch.no_grad()
def test_fn():
    tokenizer = CharacterTokenizer()
    model = s4ormer_lm(
        dim=256,
        depth=6,
        dropout=0.1,
        intermediate_loss=True,
        vocab_size=tokenizer.vocab_size,
    )
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    f_1, f_2, f_3 = torch.tensor(tokenizer('/hi there how u/')), torch.tensor(tokenizer('/buenos dias captain hook/')), torch.tensor(tokenizer('/whats up donkey man/'))
    fb, fb_lengths = collate_fn([f_1, f_2, f_3], pad_token=tokenizer.token_to_id['#'])
    fb, fb_lengths = fb.to(device), fb_lengths.to(device)
    logits_s1, interim_logits = model(fb, fb_lengths)
    print(logits_s1.shape)
    print('WE PASSED THE TEST')

if __name__ == '__main__':
    test_fn()
  