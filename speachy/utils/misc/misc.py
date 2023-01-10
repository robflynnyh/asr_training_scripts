
from typing import List, Dict, Any
from os import path
import subprocess, datetime, json, os, re
import lhotse
from lhotse import CutSet
from tqdm import tqdm
import torch
from collections import OrderedDict
import argparse



def model_surgery(state_dict, tofind, toreplace):
    '''
    Replaces "tofind" in state dict keys with "toreplace"
    '''
    new_state_dict = {}
    for k, v in state_dict.items():
        new_state_dict[k.replace(tofind, toreplace)] = v
    return new_state_dict


def convert_lhotse_to_manifest(split:CutSet, target:str):
    '''
    Converts a lhotse cutset to a nvidia nemo manifest file
    split: lhotse cutset split from load_corpus
    target: path to save (including filename)
    '''
    manifest = []
    for entry in tqdm(split):
        manifest.append({
            'text': " ".join([el.text for el in entry.supervisions]).replace('  ', ' '),
            'audio_path': "Not used",
            'duration': entry.duration
        })
    with open(target, 'w') as f:
        for line in manifest:
            f.write(json.dumps(line) + '\n')
    print(f'Saved manifest to {target}')

def list_checkpoint_val_losses(checkpoint_dir:str, verbose:bool=True, return_data:bool=False) -> Dict[str, float]:
    checkpoints = {}
    for file in os.listdir(checkpoint_dir):
        if file.endswith('.pt'):
            checkpoints[file] = None
    for file in checkpoints.keys():
        path_ = os.path.join(checkpoint_dir, file)
        checkpoint = torch.load(path_, map_location='cpu')
        checkpoints[file] = checkpoint
        if verbose:
            print(f'{file}: {checkpoints[file]["val_loss"]}')
    print('\n') if verbose else None
    if return_data:
        return checkpoints

def merge_top_checkpoints(checkpoint_dir:str, top_n:int, target:str):
    '''
    Merges the top n checkpoints in a directory into a single checkpoint
    checkpoint_dir: directory containing checkpoints
    top_n: number of checkpoints to merge
    target: path to save (including filename)
    '''
    checkpoints = list_checkpoint_val_losses(checkpoint_dir, verbose=False, return_data=True)
    checkpoints = sorted(checkpoints.items(), key=lambda x: x[1]['val_loss'])

    checkpoints = checkpoints[:top_n]

    checkpoint_weights = [checkpoints[i][1]['model_state_dict'] for i in range(top_n)]
    
    new_model_weights = OrderedDict() 
    for key in checkpoint_weights[0].keys():
        new_model_weights[key] = None
        for i in range(top_n):
            weights_to_add = checkpoint_weights[i][key] / top_n
            if new_model_weights[key] is None:
                new_model_weights[key] = weights_to_add
            else:
                new_model_weights[key] += weights_to_add
    torch.save({'model_state_dict': new_model_weights}, target)


def get_corpus_duration(split:CutSet):
    '''Returns the total duration of the corpus in hours (duh)
       split: lhotse cutset split from load_corpus
    '''
    dur = 0
    for entry in tqdm(split):
        dur += entry.supervisions[0].duration
    print(f'Corpus duration: {dur/60/60:.2f} hours')


def draw_text(text):
    print(f'\n \n ------------------------------------------------- ')
    print(f' ----------------- {text} ----------------- ')
    print(f' ------------------------------------------------- \n \n')

def save_json(obj:Dict, path:str):
    with open(path, 'w') as f:
        json.dump(obj, f)

def load_json(path:str) -> Dict:
    with open(path, 'r') as f:
        return json.load(f)

def run_cmd(cmd:str):
    print(f'Running {cmd}')
    subprocess.run(cmd, shell=True, check=True)

def get_date():
    return str(datetime.datetime.now()).replace(' ', '_').replace(':', '-').replace('.', '-')

def add_common_args(parser:argparse.ArgumentParser):
    parser.add_argument('--epochs', type=int, default=5000)
    parser.add_argument('--save_top_k', type=int, default=6)

    parser.add_argument('--accumulate_gradients', type=int, default=4)
    parser.add_argument('--min_lr', type=float, default=1e-5)
    parser.add_argument('--max_lr', type=float, default=8e-3)
    parser.add_argument('--step_size', type=int, default=400)

    parser.add_argument('--wandb', action='store_false')
    parser.add_argument('--wandb_id', type=str, default='', help='Provide this if you want to resume a previous run')

    parser.add_argument('--mixed_precision', action='store_true')
    parser.add_argument('--no_load_optim', action='store_true', help='if set, will not load optimizer state from checkpoint')

    parser.add_argument('--clip_gradients', action='store_true')
    parser.add_argument('--clip_gradients_value', type=float, default=10.0)

    parser.add_argument('--optimizer_type', type=str, default='madgrad', help='type of optimizer to use')
    parser.add_argument('--num_workers', type=int, default=1, help='number of workers for dataloader')

    return parser