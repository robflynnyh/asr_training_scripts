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
from transformers import AutoTokenizer, AutoModelForCausalLM

import non_iid_dataloader as niiddl

durations = [
    0.0,
    15.0,
    30.0,
    45.0,
    60.0,
    75.0,
    90.0,
    105.0,
    120.0,
    135.0,
    150.0,
    165.0,
    180.0,
    200.0,
    250.0,
    350.0,
    450.0,
    550.0,
    650.0,
    750.0,
    850.0,
    1000.0,
    1500.0,
    1750.0,
    2000.0,
]

class argsclass:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)



@torch.no_grad()
def main(args):
    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained('EleutherAI/gpt-neo-125M')
    corpus = tools.load_corpus()
    model = AutoModelForCausalLM.from_pretrained('EleutherAI/gpt-neo-125M')
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
            max_allowed_utterance_gap=10.0,
        )
        samples = niiddl.get_text(samples) 


        #print(samples[0])

        losses = 0
        ttl_tokens = 0
        ttl_words = 0

        skipped = 0
        for text_sample in tqdm(samples):
            input_ids = tokenizer(text_sample, return_tensors='pt')
            token_lens = input_ids['input_ids'].shape[1]
            
            if token_lens > 2048: ## NEEDS TO BE TOKEN LEVEL
                skipped += 1
                continue



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
            ttl_words += len(text_sample.split()) +1 # for the eos token

        print(f'percent skipped: {skipped/len(samples)}')
        if args.token_level:
            ppl = torch.exp(torch.tensor(losses / ttl_tokens))
        else:
            ppl = torch.exp(torch.tensor(losses / ttl_words))
        avg_token_len = ttl_tokens / len(samples)
        avg_word_len = ttl_words / len(samples)
        print(f'avg_token_len: {avg_token_len}, avg_word_len: {avg_word_len}')
        print(f'perplexity for duration {duration} is {ppl}')




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument('--max_len', type=int, default=1862)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--max_gpu_duration', type=float, default=-1)
    parser.add_argument('--checkpoint', type=str, default='')
    parser.add_argument('--shuffle_sentences', action='store_true')
    parser.add_argument('--token_level', action='store_true')

    args = parser.parse_args()


    if 'cuda' in args.device and isfalse(torch.cuda.is_available()):
        print('CUDA not available, defaulting to CPU')
        args.device = 'cpu'

    if args.max_gpu_duration == -1:
        args.max_gpu_duration = float('inf')

    main(args)

