import argparse
from ast import parse
import tools
from importlib import reload as rl
import non_iid_dataloader as niiddl, lhotse
from tqdm import tqdm
import torch
from omegaconf.omegaconf import OmegaConf
import lm_utils
import model_utils
from tools import isfalse, istrue, exists
import non_iid_dataloader as niiddl
import lm_utils
import os 
import numpy as np
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import non_iid_dataloader as niiddl

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
    200.0
]

class argsclass:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)



@torch.no_grad()
def main(args):
    device = torch.device(args.device)
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    corpus = tools.load_corpus()
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    model.to(device)
    model.eval()

    meetings = niiddl.prepare_partition(corpus[args.split])
    loss_fct = torch.nn.CrossEntropyLoss(reduction='none')

    if args.shuffle_sentences:
        np.random.seed(0)
        for key in meetings:
            meetings[key] = np.random.permutation(meetings[key])

    for duration in durations:
        if duration > args.max_gpu_duration:
            device = torch.device('cpu')
            model.to(device)
            torch.cuda.empty_cache() 

        
        samples = niiddl.prepare_samples(
            meetings=meetings, 
            max_duration=duration, 
            max_allowed_utterance_gap=5.0,
        )
        samples = niiddl.get_text(samples) 


        #print(samples[0])

        losses = 0
        ttl_tokens = 0

        for text_sample in tqdm(samples):
            input_ids = tokenizer(text_sample, return_tensors='pt')
            token_lens = input_ids['input_ids'].shape[1]
            input_ids = lm_utils.batch_to_device(input_ids, device, return_all=True)
            outputs = model(**input_ids)
            logits = outputs.logits
            labels = input_ids['input_ids']
            lm_logits = logits
            shift_logits = lm_logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            loss = loss.view(shift_labels.size())
            losses += loss.sum().item()
            #print(token_lens)
            ttl_tokens += token_lens

        ppl = torch.exp(torch.tensor(losses / ttl_tokens))
        avg_token_len = ttl_tokens / len(samples)
        print(f'avg_token_len: {avg_token_len}')
        print(f'perplexity for duration {duration} is {ppl}')




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./lm/decoder_test.yaml')
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument('--max_len', type=int, default=1862)
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--max_gpu_duration', type=float, default=-1)
    parser.add_argument('--checkpoint', type=str, default='')
    parser.add_argument('--shuffle_sentences', action='store_true')

    args = parser.parse_args()


    if 'cuda' in args.device and isfalse(torch.cuda.is_available()):
        print('CUDA not available, defaulting to CPU')
        args.device = 'cpu'

    if args.max_gpu_duration == -1:
        args.max_gpu_duration = float('inf')

    main(args)

