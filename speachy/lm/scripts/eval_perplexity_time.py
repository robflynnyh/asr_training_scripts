import argparse
from ast import parse
import tools
from importlib import reload as rl
import non_iid_dataloader as niiddl, lhotse
from tqdm import tqdm
import torch
import torch
from omegaconf.omegaconf import OmegaConf
import model_utils
from tools import isfalse, istrue, exists
from speachy.asr.dataloading import non_iid_dataloader as niiddl
from speachy.lm.tools.loading import autoload
import os 
from einops import repeat
import time

from speachy.utils.general import (
    load_config,
    load_checkpoint,
    load_tokenizer
)

from speachy.lm.tools.train import (
    loss_ce,
    batch_to_device,
    token_lens_to_mask,
    add_bos,
    add_eos,
    mark_padding,
)



class argsclass:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def trim_cache(kv_cache, max_len):
    if max_len == 0:
        return None
    if kv_cache is None:
        return None

    if max_len == -1:
        return kv_cache
    #print(kv_cache['cache'].shape)
    
    if kv_cache['cache'].shape[-2] > max_len:
        print(kv_cache['cache_lengths'].shape, kv_cache['cache_lengths']) 
        bos = kv_cache['cache'][:, :, :, :, 0, :].unsqueeze(-2).clone()
        kv_cache['cache'] = kv_cache['cache'][:, :, :, :, -max_len:, :]
        kv_cache['cache'] = torch.cat([bos, kv_cache['cache']], dim=-2)
        kv_cache['cache_lengths'] = torch.tensor([kv_cache['cache'].shape[-2]]).to(kv_cache['cache_lengths'].device)
        kv_cache['cache_lengths'] = repeat(kv_cache['cache_lengths'], '() -> b', b=25)
        print(kv_cache['cache_lengths'].shape, kv_cache['cache_lengths'])
    return kv_cache


@torch.no_grad()
def main(args):
    device = torch.device(args.device)

   
    config = load_config(args.config)
    tokenizer_path = os.path.join(config['model']['tokenizer']['dir'], 'tokenizer.model')
    tokenizer = load_tokenizer(tokenizer_path)

    dataset = config['dataset']
    corpus = tools.load_corpus(
        target_folder = tools.request_env(dataset+'_PATH'),
        prefix_path = tools.request_env(dataset+'_BASE'),
        file_name = tools.request_env(dataset+'_NAME')
    ) 

    wordlevel = False if args.token_level else True

    model = autoload(config, tokenizer)
    epoch, val_loss  = load_checkpoint(args=argsclass(**{'checkpoint': args.checkpoint}), model=model, force_cpu=True)
    modeltype = config['model']['modeltype']
    model.to(device)
    model.eval()


    print('Model Loaded')
    print(f'Model type: {modeltype}. Epoch: {epoch}. Val loss: {val_loss}')


    device = torch.device(args.device)
    model.to(device)
    torch.cuda.empty_cache() 

    dl = niiddl.get_eval_dataloader(
        corpus[args.split], 
        batch_size=1, # only implement batch size 1 for now ):
        shuffle=False,
        max_duration=0,
        max_allowed_utterance_gap=args.max_allowed_utterance_gap,
        concat_samples=True,
        return_meta_data=True,
    )

    cache = None
    prev_end = 0
    prev_recording_id = None
    total_loss = 0
    total_words = 0
    last_logit = None
    tokens_covered = 0
    TIME_LIMIT = 10 # minutes
    TIME_LIMIT = TIME_LIMIT * 60 # seconds
    pbar = tqdm(dl)
    stime = time.time()
    for i, batch in enumerate(pbar):
        if time.time() - stime > TIME_LIMIT:
            print(f'Covered {tokens_covered} tokens in {TIME_LIMIT} seconds')
            exit()

        text, metadata = batch['text'][0][0], batch['metadata'][0][0]
        recording_id = metadata['recording_id']
        start_t, end_t = metadata['timings'][0].values()
        #print(metadata['timings'], text)

        #
        if recording_id != prev_recording_id or (start_t - prev_end) > args.max_allowed_utterance_gap:
            cache = None # don't propagate cache through utterances from different recordings or with large gaps
            last_logit = None
            print('resetting cache', recording_id != prev_recording_id, start_t - prev_end, '\n\n\n\n')

        cache = trim_cache(cache, args.max_cache)
        last_logit = None if not exists(cache) else last_logit

        if len(text) == 0:
            prev_end = end_t
            prev_recording_id = recording_id
            continue

        tokens = tokenizer.text_to_ids(text)
        tokens = torch.tensor(tokens).unsqueeze(0)

        tokens = tokens.to(device)

        tokens = add_bos(tokens, bos_token_id=0) if not exists(cache) else tokens
        token_lens = torch.tensor([tokens.shape[1]]).to(device)

        
        #print(tokens.shape)
        # incremantal decoding
        for t_idx in range(tokens.shape[1]):
            #print(tokens[:,None, t_idx].shape, torch.tensor([[1]]).shape)
            input_c = tokens[:, None, t_idx].clone()
            input_c = repeat(input_c, '() () -> b ()', b=25)
            tlens = torch.tensor([1]).to(device)
            tlens = repeat(tlens, '() -> b', b=25)
            logits, _, cached_states = model(x=input_c, length=tlens, cache=cache)
            tokens_covered += 1
            cache = cached_states
     

    
        prev_end = end_t
        prev_recording_id = recording_id
        total_words += len(text.split(' '))

        

        
# 25047




if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', type=str, default='./experiment_configs/lm/decoder_pg19_sep_token_ted_am.yaml')
    parser.add_argument('--max_cache', type=int, default=1000)

    parser.add_argument('--no_sep', action='store_true')

    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--max_gpu_duration', type=float, default=-1)
    parser.add_argument('-mgap','--max_allowed_utterance_gap', type=float, default=10.0, help='max allowed gap between utterances in seconds')
    parser.add_argument('--token_level', action='store_true')
    parser.add_argument('--split', type=str, default='test')
    
    parser.add_argument('--checkpoint', type=str, default='./checkpoints/open_sub_ft_ted/ft_ted_checkpoint_1259_id_36.pt')
    args = parser.parse_args()

    if args.device == 'auto':
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if args.checkpoint == '':
        print('No checkpoint specified...')
        ckpt = input('Please specify a checkpoint to evaluate: ')
        args.checkpoint = ckpt

    if 'cuda' in args.device and isfalse(torch.cuda.is_available()):
        print('CUDA not available, defaulting to CPU')
        args.device = 'cpu'

    if args.max_gpu_duration == -1:
        args.max_gpu_duration = float('inf')

    main(args)

