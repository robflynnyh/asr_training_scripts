import argparse
from ast import parse
import tools
from importlib import reload as rl
import non_iid_dataloader as niiddl, lhotse
from tqdm import tqdm
from torch.nn import TransformerDecoder, TransformerDecoderLayer, Transformer
import torch
import x_transformers, torch
from omegaconf.omegaconf import OmegaConf
import lm_utils
import model_utils
from tools import isfalse, istrue, exists
import non_iid_dataloader as niiddl
import lm_utils
import os 

durations = [
    0.0,
    15.0,
    30.0,
    50.0,
    60.0,
    75.0,
    100.0,
    120.0,
    140.0,
    180.0,
    200.0,
    250.0,
    300.0,
]

class argsclass:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def main(args):
    device = torch.device(args.device)
    tk = tools.load_tokenizer()
    tokenizer = tk
    corpus = tools.load_corpus()
    partition = niiddl.prepare_partition(corpus['train'])
    config = lm_utils.load_config(args.config)
    tokenizer_path = os.path.join(config['model']['tokenizer']['dir'], 'tokenizer.model')
    tokenizer = tools.load_tokenizer(tokenizer_path)

    model = lm_utils.load_model(config, tokenizer, max_len=args.max_len)
    epoch, val_loss  = model_utils.load_checkpoint(args=argsclass(**{'checkpoint': args.checkpoint}), model=model, force_cpu=True)
    modeltype = config['model']['modeltype']
    model.to(device)
    model.eval()

    '''
    # enable dropout
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()
    '''
    
    print('Model Loaded')
    print(f'Model type: {modeltype}. Epoch: {epoch}. Val loss: {val_loss}')

    for duration in durations:
        if duration > args.max_gpu_duration:
            device = torch.device('cpu')
            model.to(device)
            torch.cuda.empty_cache() 

        dl = niiddl.get_data_loader(
            corpus['test'], 
            tokenizer=tokenizer, 
            batch_size=args.batch_size, 
            shuffle=False,
            max_duration=duration,
            text_only=True,
            max_allowed_utterance_gap=args.max_allowed_utterance_gap
        )
        ppl, avg_len = lm_utils.eval_corpus_perplexity(model, dl, device=device)
        print(f"Duration: {duration}, PPL: {ppl}, Avg Len: {avg_len}")



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./lm/decoder_test.yaml')
    
    parser.add_argument('--max_len', type=int, default=1862)
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--max_gpu_duration', type=float, default=-1)
    parser.add_argument('-mgap','--max_allowed_utterance_gap', type=float, default=3.0, help='max allowed gap between utterances in seconds')

    parser.add_argument('--checkpoint', type=str, default='')
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

