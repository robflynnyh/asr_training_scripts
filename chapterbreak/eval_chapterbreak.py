import argparse
import pickle as pkl
from Levenshtein import distance
from speachy.rescoring.tools import ( sort_hypothesis_by_recording, order_recordings_by_start_time, )
import numpy as np
from functools import reduce
import torch
import os
from os import listdir as ls
from os.path import join as pj


from speachy.utils.misc import ( add_common_args, get_parameters, load_pkl )

from speachy.utils.general import (
    load_config,
    load_checkpoint,
    load_tokenizer,
    save_checkpoint,
    save_schedular_data,
    write_to_log,
    draw_text,
)

from speachy.lm.tools.train import (
    loss_ce,
    batch_to_device,
    token_lens_to_mask,
    add_bos,
    add_eos,
    mark_padding
)

from speachy.utils.general import load_checkpoint
from speachy.lm.tools.loading import autoload
from speachy.utils.helpers import  exists, isfalse, istrue
from speachy.utils.general.training_loop import optimizer, update_schedular
from contextlib import nullcontext
from tqdm import tqdm
import wandb
from torch.cuda.amp import GradScaler   
from torch_ema import ExponentialMovingAverage
import json
from typing import List, Dict, Tuple, Union, Optional
import pandas as pd

TOP_LEVEL_PATH = './eval_data'
FILES = [
    'chapterbreak_ctx_4096.json',
]

def load_test_examples(split):
    text_examples = []
    for file in FILES:
        with open(pj(TOP_LEVEL_PATH, file), 'r') as f:
            examp = json.load(f)
            for key in examp[split].keys():
                text_examples.extend(examp[split][key])
    return text_examples

@torch.no_grad()
def score_example_set(model, tokenizer, example_set):
    context, positive, negatives = example_set['ctx'], example_set['pos'], example_set['negs']
    device = model.device
    context = torch.tensor(tokenizer.text_to_ids(context))[None, :].to(device)
    positive = torch.tensor(tokenizer.text_to_ids(positive))[None, :].to(device)
    negatives = [torch.tensor(tokenizer.text_to_ids(neg))[None, :].to(device) for neg in negatives]

    # cat context to positive and negatives
    positive = torch.cat([context, positive], dim=1)
    negatives = [torch.cat([context, neg], dim=1) for neg in negatives]

    positive = add_bos(positive, bos_token_id=0) 
    negatives = [add_bos(neg, bos_token_id=0) for neg in negatives]

    #_, _, cache = model(labels = context, length = torch.LongTensor([context.shape[1]]).to(device), cache=None, calc_loss=False)
    pos_score = model(x = positive, length = torch.LongTensor([positive.shape[1]]).to(device), calc_loss=True)
    pos_score =  pos_score.item() * positive.shape[1]
    neg_scores = []
    for neg in negatives:
        neg_score = model(x = neg, length = torch.LongTensor([neg.shape[1]]).to(device), calc_loss=True)
        neg_scores.append((neg_score ).item() * neg.shape[1])
    print(f'pos_score: {pos_score}, neg_scores: {neg_scores}')
    min_neg = max(neg_scores)
    winning = False
    if pos_score > min_neg:
        winning = True
    #print(f'Winning: {winning}')
    return winning
    



def run_eval(model, tokenizer, test_data):
    won = 0
    total = 0
    for example_set in test_data:
        winning = score_example_set(model, tokenizer, example_set)
        won += winning
        total += 1
        print(f'Won: {won}, Total: {total}, Winrate: {won/total}')



def main(args):
    device, config = torch.device(args.device), load_config(args.config)
    tokenizer_path = os.path.join(config['model']['tokenizer']['dir'], 'tokenizer.model')
    tokenizer = load_tokenizer(model_path=tokenizer_path)
    model = autoload(config=config, tokenizer=tokenizer)

    epoch_prev = 0
    if args.checkpoint != '':
        epoch_prev, val_loss = load_checkpoint(args=args, model=model, force_cpu=True)
        modeltype = config['model']['modeltype']
        print(f'Loaded model {args.checkpoint} with epoch {epoch_prev} and val_loss {val_loss}\n Model type: {modeltype}')
    model.to(device)
    model.device = device
    model.eval()

    test_data = load_test_examples(args.split) 
    print(f'Loaded {len(test_data)} test examples')
    run_eval(model, tokenizer, test_data)
    
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', type=str, default='pg19')
    parser.add_argument('--config', type=str, default='./experiment_configs/pg19_feedbacktlm.yaml')
    parser.add_argument('--checkpoint', type=str, default='./checkpoints/checkpoint_101_id_8.pt')
    parser.add_argument('--device', type=str, default='auto')
    args = parser.parse_args()

    args.device = 'cuda' if torch.cuda.is_available() and args.device == 'auto' else 'cpu'

    main(args)