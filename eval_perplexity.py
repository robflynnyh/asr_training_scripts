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
    400.0,
    500.0,
    600.0,
    700.0,
    800.0,
    1000.0,
    1250.0,
    1500.0
]

class argsclass:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def main(args):
    device = torch.device(args.device)

   
    config = lm_utils.load_config(args.config)
    tokenizer_path = os.path.join(config['model']['tokenizer']['dir'], 'tokenizer.model')
    tokenizer = tools.load_tokenizer(tokenizer_path)

    dataset = config['dataset']
    corpus = tools.load_corpus(
        target_folder = tools.request_env(dataset+'_PATH'),
        prefix_path = tools.request_env(dataset+'_BASE'),
        file_name = tools.request_env(dataset+'_NAME')
    ) 

    wordlevel = False if args.token_level else True

    model = lm_utils.load_model(config, tokenizer, max_len=args.max_len)
    epoch, val_loss  = model_utils.load_checkpoint(args=argsclass(**{'checkpoint': args.checkpoint}), model=model, force_cpu=True)
    modeltype = config['model']['modeltype']
    model.to(device)
    model.eval()


    print('Model Loaded')
    print(f'Model type: {modeltype}. Epoch: {epoch}. Val loss: {val_loss}')

    for duration in durations:
        if duration > args.max_gpu_duration:
            device = torch.device('cpu')
            model.to(device)
            torch.cuda.empty_cache() 

        dl = niiddl.get_data_loader(
            corpus[args.split],
            tokenizer=tokenizer, 
            batch_size=args.batch_size, 
            shuffle=False,
            max_duration=duration,
            text_only=True,
            max_allowed_utterance_gap=args.max_allowed_utterance_gap,
            split_speakers=True
        )
        ppl, avg_len = lm_utils.eval_corpus_perplexity(model, dl, device=device, word_level=wordlevel)
        print(f"Duration: {duration}, PPL: {ppl}, Avg Len: {avg_len}")



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./experiment_configs/lm/decoder_pg19.yaml')
    
    parser.add_argument('--max_len', type=int, default=1862)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--max_gpu_duration', type=float, default=-1)
    parser.add_argument('-mgap','--max_allowed_utterance_gap', type=float, default=10.0, help='max allowed gap between utterances in seconds')
    parser.add_argument('--token_level', action='store_true')
    parser.add_argument('--split', type=str, default='test')
    
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

