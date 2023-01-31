import torch

from omegaconf.omegaconf import OmegaConf
import tools
from einops import rearrange, repeat
from tqdm import tqdm
import torch.nn as nn

from tools import (
    exists,
    isfalse
)

def do_sample(distribution, temperature=1.0):
    if temperature == 0.0:
        return torch.argmax(distribution, dim=-1)

    else:
        return torch.multinomial(distribution, num_samples=1).squeeze(-1)

@torch.no_grad()
def greedy_generate(model, tokenizer, input_txt, max_len, force_cpu=False, temperature=0.0):
    model.eval()
    device = 'cuda' if torch.cuda.is_available() and isfalse(force_cpu) else 'cpu'
    model.to(device)
    input_ids = [0] + tokenizer.text_to_ids(input_txt)
    input_ids = torch.tensor(input_ids, device=device).unsqueeze(0)
    
    output_tokens = input_ids.squeeze().tolist()
    while len(output_tokens) < max_len:
        logits = model(input_ids)
       # print(input_ids)
        logits = logits[:, -1, :]
        logits = logits[:, 1:] # remove <pad>
        probs = torch.softmax(logits, dim=-1)
        next_token = do_sample(probs, temperature=temperature) + 1 # add <pad>
    
        output_tokens.append(next_token.item())
        input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)
    #print(output_tokens)
    return f'{tokenizer.ids_to_text(output_tokens)}'



@torch.no_grad()
def eval_perplexity(model, tokens, token_lens, return_ppl=True):
    model.eval()
    token_lens += 1
    tokens = add_bos(tokens, bos_token_id=0)
    targets = tokens.clone()
    targets[:, :-1] = tokens[:, 1:]
    targets = add_eos(targets, eos_id=0, token_lens=token_lens)
    mask = token_lens_to_mask(token_lens)
    
    targets = mark_padding(targets, mask, pad_id=-100)

    model_args = {'x': tokens, 'mask': mask} if isfalse(callable(getattr(model, 'get_args', False))) \
        else model.get_args(tokens=tokens, mask=mask, lengths=token_lens)
    
    logits = model(**model_args)
    loss = torch.nn.functional.cross_entropy(rearrange(logits, 'b n c -> b c n'), targets, ignore_index=-100, reduction='none')
    loss = loss.sum(dim=1) / token_lens # per-token loss
    ppl = torch.exp(loss) # per-token perplexity

    return torch.exp(loss).cpu() if return_ppl else loss.cpu()

@torch.no_grad()
def eval_corpus_perplexity(model, dataloader, device, word_level=False):
    model.eval()
    model.to(device)
    losses = []
    pbar = tqdm(dataloader)
    all_token_lens = []
    text_lens = []
    for batch in pbar:
        b_text = batch['text']
        #print(b_text)
        b_text_lens = [len(t.split()) + 1  for t in b_text] # + 1 for <eos> ! split() ignores whitespaces which is what we want
        text_lens.extend(b_text_lens)

        tokens, token_lens = batch_to_device(batch, device)

        cur_loss = eval_perplexity(model, tokens, token_lens, return_ppl=False)
        losses.append(cur_loss)
        all_token_lens.append(token_lens)
        pbar.set_description(f'loss: {cur_loss.mean().item():.2f}')

    all_token_lens = torch.cat(all_token_lens).cpu()
    avg_token_lens = all_token_lens.reshape(-1).float().mean()
 
    losses = torch.cat(losses)
    avg_loss = losses.reshape(-1).float().mean()
    text_lens = torch.tensor(text_lens)

    if not word_level:
        ppl = torch.exp(avg_loss) 
    else:
        ppl = (losses * (all_token_lens+1)) / text_lens # +1 for <eos> !
        ppl = torch.exp(ppl.mean())        

    return ppl, avg_token_lens.item()


def model_pipeline(model, tokenizer, eval=True, device='cpu'):
    raise NotImplementedError
    model.eval() if eval else model.train()
    device = torch.device(device)
    model.to(device)

    def _model_pipeline(batch:List[str]) -> dict:
        '''returns loss'''
        tokens = [tokenizer.text_to_ids(txt) for txt in batch]
        token_lens = torch.tensor([len(token) for token in tokens], device=device)
        # pad tokens
        max_len = token_lens.max().item()
        tokens = [token + [0] * (max_len - len(token)) for token in tokens]
        tokens = torch.tensor(tokens, device=device)
        tokens = add_bos(tokens, bos_token_id=0)
        targets = tokens.clone()
        targets[:, :-1] = tokens[:, 1:] # shift left
        targets = add_eos(targets, eos_id=0, token_lens=token_lens)
        mask = token_lens_to_mask(token_lens)
        targets = mark_padding(targets, mask, pad_id=-100)

        model_args = {'x': tokens, 'mask': mask} if isfalse(callable(getattr(model, 'get_args', False))) \
            else model.get_args(tokens=tokens, mask=mask, lengths=token_lens)


def loss_ce(logits, labels, ignore_index=-100, label_smoothing=0.0):
    return torch.nn.functional.cross_entropy(
            rearrange(logits, 'b n c -> b c n'), 
            labels, 
            ignore_index = ignore_index,
            label_smoothing = label_smoothing
        )

def load_config(config:str):
    return OmegaConf.load(config)

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

def token_lens_to_mask(token_lens):
    max_len = token_lens.max()
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

def get_max_length(dataloader):
    max_len = 0
    for batch in dataloader:
        max_len = max(max_len, batch['tokens'].shape[1])
    return max_len

class PerceiverARadapter(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def process_labels(self, labels):
        return labels[:, self.model.cross_attn_seq_len:] 

    def forward(self, x, mask=None):
        return self.model(x=x, prefix_mask=mask, labels=None)


class S4adapter(nn.Module):
    '''
    Creates a network using state space model as layers in place of something like attention
    '''
    def __init__(
            self, 
            vocab_size,
            s4config,
            n_layers = 4,
        ):
        super().__init__()
        raise NotImplementedError
        from lm.s4 import S4
        
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, s4config['d_model'])
        self.predict = nn.Linear(s4config['d_model'], vocab_size)
        self.softmax = nn.Softmax(dim=-1)
        self.s4config = s4config

    def get_args(self, tokens, mask, lengths):
        return {'tokens': tokens,'mask': mask,'lengths': lengths,}

    def greedy_generate(self, text, tokenizer, num_steps=10, device='cpu', temperature=0.0):
        raise NotImplementedError  

    def _forward(self, u, lengths, return_states=False, states=None):
        pass

    def forward(self, tokens, mask=None, lengths=None):
        x = self.embedding(tokens)
        x = x.masked_fill(~mask.unsqueeze(-1), 0) if exists(mask) else x
        logits = self._forward(x, lengths, return_states=False)
        return logits
    


def load_model(config:OmegaConf, tokenizer, max_len:int=None):
    assert 'model' in config
    modelconfig = config['model']
    mtype = modelconfig.get('modeltype', 'transformer')

    if 'transformer' in mtype:
        import x_transformers
        assert mtype in modelconfig
        modelconfig = modelconfig[mtype]
        model = x_transformers.TransformerWrapper(
            num_tokens=tokenizer.vocab_size,
            max_seq_len=max_len if exists(max_len) else modelconfig.get('max_seq_len', 1024),
            attn_layers= x_transformers.Decoder(
                dim = modelconfig.get('d_model', 256),
                depth = modelconfig.get('n_layers', 12),
                heads = modelconfig.get('n_heads', 8),
                rotary_pos_emb = modelconfig.get('rotary_pos_emb', False),
                dynamic_pos_bias = modelconfig.get('dynamic_pos_bias', True),
            )
        )
    elif 'myopic' in mtype:
        from lm.myopic_attention import transformer_lm
        assert mtype in modelconfig
        modelconfig = modelconfig[mtype]
        model = transformer_lm(
            dim = modelconfig.get('d_model', 256),
            vocab_size=tokenizer.vocab_size,
            depth = modelconfig.get('n_layers', 12),
            heads = modelconfig.get('n_heads', 8),
            max_keep_keys=modelconfig.get('max_keep_keys', 128),
            W = modelconfig.get('W', 48),
            dim_head = modelconfig.get('dim_head', 32),
            causal=True,
            dropout= modelconfig.get('dropout', 0.0),
        )
    elif 'mlp' in mtype:
        from lm.MLPLM import transformer_lm
        modelconfig = modelconfig[mtype]
        
        model = transformer_lm( # some of this is redundant 
            dim = modelconfig.get('d_model', 256),
            vocab_size=tokenizer.vocab_size,
            depth = modelconfig.get('n_layers', 12),
            heads = modelconfig.get('n_heads', 8),
            dim_head = modelconfig.get('dim_head', 32),
            causal=True,
            dropout= modelconfig.get('dropout', 0.0),
            temperature= modelconfig.get('temperature', 15.5),
            **modelconfig.get('kwargs', {})
        )
        
    elif 'qknorm' in mtype:
        if 'gau' in mtype:
            from lm.gau_qknorm_attention import transformer_lm
        elif 'hierarchy' in mtype:
            from lm.qknorm_attention_hierarchy import transformer_lm
        else:
            from lm.qknorm_attention import transformer_lm
        assert mtype in modelconfig
        modelconfig = modelconfig[mtype]
        model = transformer_lm(
            dim = modelconfig.get('d_model', 256),
            vocab_size=tokenizer.vocab_size,
            depth = modelconfig.get('n_layers', 12),
            heads = modelconfig.get('n_heads', 8),
            dim_head = modelconfig.get('dim_head', 32),
            causal=True,
            dropout= modelconfig.get('dropout', 0.0),
            temperature= modelconfig.get('temperature', 15.5),
            **modelconfig.get('kwargs', {})
        )

    elif mtype == 'perceiverAR':
        from perceiver_ar_pytorch import PerceiverAR
        assert 'perceiverAR' in modelconfig
        modelconfig = modelconfig['perceiverAR']
        model = PerceiverAR(
            num_tokens = tokenizer.vocab_size,
            dim = modelconfig.get('d_model', 256),
            depth = modelconfig.get('depth', 12),
            heads = modelconfig.get('n_heads', 8),
            dim_head = modelconfig.get('dim_head', 32),
            cross_attn_seq_len = modelconfig.get('cross_attn_seq_len', 256),
            cross_attn_dropout = modelconfig.get('cross_attn_dropout', 0.4),
            max_seq_len = max_len if exists(max_len) else modelconfig.get('max_seq_len', 1024),
        )
        model = PerceiverARadapter(model)

    elif mtype == 'S4':
        #from lm.s4 import S4
        assert 'S4' in modelconfig
        modelconfig = modelconfig['S4']
        model = S4adapter(
            s4config = {
                'measure': modelconfig.get('measure', 'legs'),
                'mode': modelconfig.get('mode', 'nplr'),
                'transposed': modelconfig.get('transposed', False),
                'd_model': modelconfig.get('d_model', 2048),
                'd_state': modelconfig.get('d_state', 256),
            },
            vocab_size = tokenizer.vocab_size,
            n_layers = modelconfig.get('n_layers', 6),
        )
    else:
        model = None
        raise NotImplementedError(f'Unknown model type {mtype}')
    model.vocab_size = tokenizer.vocab_size
    return model