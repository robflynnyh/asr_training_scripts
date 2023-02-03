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

durations = [
    0.0,
]

class argsclass:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


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
    last_token = None
    for i, batch in enumerate(tqdm(dl)):
        text, metadata = batch['text'][0][0], batch['metadata'][0][0]
        recording_id = metadata['recording_id']
        start_t, end_t = metadata['timings'][0].values()
        if recording_id != prev_recording_id or (start_t - prev_end) > args.max_allowed_utterance_gap:
            cache = None # don't propagate cache through utterances from different recordings or with large gaps
        if len(text) == 0:
            continue
        tokens = tokenizer.text_to_ids(text)
        tokens = torch.tensor(tokens).unsqueeze(0)
        tokens = tokens.to(device)
        if cache is None:
            tokens = add_bos(tokens, bos_token_id=0)
        targets = tokens.clone()



if __name__ == '__main__':
    raise NotImplementedError('burrr')
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', type=str, default='./experiment_configs/lm/decoder_pg19.yaml')
    
    parser.add_argument('--max_len', type=int, default=1862)

    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--max_gpu_duration', type=float, default=-1)
    parser.add_argument('-mgap','--max_allowed_utterance_gap', type=float, default=10.0, help='max allowed gap between utterances in seconds')
    parser.add_argument('--token_level', action='store_true')
    parser.add_argument('--split', type=str, default='test')
    
    parser.add_argument('--checkpoint', type=str, default='./checkpoints/1kw_pg19checkpoints_nths/finetuned_tedlium/pg_19_ft_71_id_76_60s_BEST.pt')
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

