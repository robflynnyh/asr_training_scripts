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

INTERPOLATE = 0.5

def load_txt(fname):
    with open(fname, 'r') as f:
        txt = f.read()
    return txt
    
def tokenize_and_pad(utterances, tokenizer, max_len): # returns padded utterances and their lengths
    tokenized = [tokenizer.text_to_ids(utt) for utt in utterances]
    # limit the length of the utterances by splitting each utterance into multiple utterances
    tokenized = [utt[i:i+max_len] for utt in tokenized for i in range(0, len(utt), max_len)] if max_len != -1 else tokenized
    max_len = max(map(len, tokenized))
    padded_utts = np.array([utt + [0] * (max_len - len(utt)) for utt in tokenized])
    return padded_utts, np.array(list(map(len, tokenized)))

def load_batch_files(fnames, tokenizer, max_len, max_batch): 
    strings = [load_txt(fname) for fname in fnames]
    tokenized_strings, tokenized_lengths = tokenize_and_pad(strings, tokenizer, max_len) # limit the number of items per batch
    tokenized_strings = [tokenized_strings[i:i+max_batch] for i in range(0, len(tokenized_strings), max_batch)]
    tokenized_lengths = [tokenized_lengths[i:i+max_batch] for i in range(0, len(tokenized_lengths), max_batch)]
    return tokenized_strings, tokenized_lengths



class Sampler():
    def __init__(self, f_names, batch_size, tokenizer, shuffle=True, split_into_splits=30, max_txt_len=4096):
        self.f_names = f_names
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.shuffle = shuffle
        self.fname_indices = np.arange(len(self.f_names))
        np.random.shuffle(self.fname_indices) if self.shuffle else None
        self.cur_split = 0
        self.split_into_splits = split_into_splits
        self.max_txt_len = max_txt_len
        self.fname_indice_splits = np.array_split(self.fname_indices, self.split_into_splits)
        
        self.generator = self.__generator__() 

    def __generator__(self): 
        for i in range(0, len(self.fname_indice_splits[self.cur_split]), self.batch_size):
            yield load_batch_files(
                [self.f_names[i] for i in self.fname_indice_splits[self.cur_split][i:i+self.batch_size]], self.tokenizer, self.max_txt_len, self.batch_size)

    def __list__(self):
        return list(self.generator)

    def __iter__(self):
        return self

    def __len__(self):
        return len(self.fname_indice_splits[self.cur_split]) // self.batch_size

    def __next__(self):
        return next(self.generator)

    def step(self):
        self.cur_split += 1
        self.generator = self.__generator__()
        if self.cur_split >= self.split_into_splits:
            print(f'FINISHED FULL EPOCH WOOOO')
            return True

@torch.no_grad()
def validate_one_epoch(args, model, val_dataloader, device, sanity_check=False):
    model.eval()
    #val_dataloader = list(val_dataloader)
    pbar = tqdm(val_dataloader, total=len(val_dataloader))
    losses = []
    print('Evaluation epoch')
    for tokenized_strings_set, tokenized_lengths_set in pbar:
        for ix, (tokenized_strings, tokenized_lengths) in enumerate(zip(tokenized_strings_set, tokenized_lengths_set)):
            #print(f'sub_batch {ix+1} / {len(batch)}') 
            tokenized_strings = torch.tensor(tokenized_strings, device=device)
            tokenized_lengths = torch.tensor(tokenized_lengths, device=device)

            using_bos = True
            tokenized_lengths += 1 if using_bos else 0 # add 1 for bos if using
            tokens = add_bos(tokenized_strings, bos_token_id=0) if using_bos else tokenized_strings # add bos only if this is the first sub-batch
            
            loss = model(x = tokens, length = tokenized_lengths, calc_loss = True)
            losses.append(loss.item())

            if sanity_check:
                return True

    torch.cuda.empty_cache()
    return  sum(losses) / len(losses)



def train_one_epoch(args, epoch, model, optim, schedular, train_dataloader, device, scaler, ema=None):
    model.train()
    #train_dataloader = list(train_dataloader)
    pbar = tqdm(train_dataloader, total=len(train_dataloader))
    losses = []
    loss_iterim = []
    autocast_device = 'cuda' if torch.cuda.is_available() else 'cpu' # for autocast if using mixed precision
    print('Training epoch')
    tokenized_lengths_set = []
    tokenized_strings_set = []
    for tokenized_strings_set_, tokenized_lengths_set_ in pbar:
        # shuffle the order of the sub-batches
        tokenized_lengths_set.extend(tokenized_lengths_set_)
        tokenized_strings_set.extend(tokenized_strings_set_)

    rnd_indices = np.arange(len(tokenized_strings_set))
    np.random.shuffle(rnd_indices)
    tokenized_strings_set_shuffled = [tokenized_strings_set[i] for i in rnd_indices]
    tokenized_lengths_set_shuffled = [tokenized_lengths_set[i] for i in rnd_indices]
    tokenized_strings_set = tokenized_strings_set_shuffled
    tokenized_lengths_set = tokenized_lengths_set_shuffled


    for ix, (tokenized_strings, tokenized_lengths) in tqdm(enumerate(zip(tokenized_strings_set, tokenized_lengths_set)), total=len(tokenized_strings_set)):
        with torch.autocast(device_type=autocast_device) if exists(scaler) else nullcontext(): # for mixed precision
            #print(f'sub_batch {ix+1} / {len(batch)}') 
            tokenized_strings = torch.tensor(tokenized_strings, device=device)
            tokenized_lengths = torch.tensor(tokenized_lengths, device=device)

            using_bos = True
            tokenized_lengths += 1 if using_bos else 0 # add 1 for bos if using
            tokens = add_bos(tokenized_strings, bos_token_id=0) if using_bos else tokenized_strings # add bos only if this is the first sub-batch
            #print(tokens[0])
            loss = model(x = tokens, length = tokenized_lengths, calc_loss = True)
            

        losses.append(loss.item())
        #print('backpass started')
        scaler.scale(loss).backward() if exists(scaler) else loss.backward()
        #print('backpass ended')
        if args.clip_gradients == True:
            scaler.unscale_(optim) if exists(scaler) else None
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_gradients_value) 
        scaler.step(optim) if exists(scaler) else optim.step()
        scaler.update() if exists(scaler) else None
        optim.zero_grad()

        if exists(schedular):
            schedular.step()
        if exists(ema):
            ema.update()
        cur_lr = schedular.get_last_lr()[0] if exists(schedular) else optim.param_groups[0]['lr']

        if args.wandb:
            wandb.log({'train_loss': losses[-1], 'lrate': cur_lr})
            
    loss_end = sum(losses) / len(losses)
    if args.wandb:
        wandb.log({'train_loss_end': loss_end, 'epoch': epoch})

    torch.cuda.empty_cache()
    return loss_end



def load_file_list(path): 
    return [pj(path, f) for f in os.listdir(path) if f.endswith('.txt')]

def main(args):

    device, config = torch.device(args.device), load_config(args.config)
    tokenizer_path = os.path.join(config['model']['tokenizer']['dir'], 'tokenizer.model')
    tokenizer = load_tokenizer(model_path=tokenizer_path)
    model = autoload(config=config, tokenizer=tokenizer)

    train_data, dev_data = map(load_file_list, [args.train_data, args.dev_data])
    print(f'Loaded {len(train_data)} train samples and {len(dev_data)} dev samples')

    epoch_prev = 0
    if args.checkpoint != '':
        epoch_prev, val_loss = load_checkpoint(args=args, model=model, force_cpu=True)
        modeltype = config['model']['modeltype']
        print(f'Loaded model {args.checkpoint} with epoch {epoch_prev} and val_loss {val_loss}\n Model type: {modeltype}')
    model.to(device)

    optim, schedular = optimizer(model, args)
    save_schedular_data(args)

    total_params = get_parameters(model=model, verbose=True)

    validate_one_epoch(
        args,
        model = model,
        val_dataloader = Sampler(f_names = dev_data, batch_size = args.batch_size, tokenizer = tokenizer, shuffle = False, split_into_splits = 15),
        device = device,
        sanity_check = True
    )

    train_sampler = Sampler(f_names=train_data, batch_size=args.batch_size, tokenizer=tokenizer, shuffle=True, split_into_splits=400)

  
    ema = ExponentialMovingAverage(model.parameters(), decay=0.999) if args.ema else None
    scaler = GradScaler() if args.mixed_precision else None

    run_config = {'run_args': args.__dict__, 'model_config': config, 'total_params': total_params}
    wandb_init = {'project':args.project_name, 'config':run_config}
    if args.wandb:
        wandb.init(**wandb_init
        ) if args.wandb_id == '' else wandb.init(id=args.wandb_id, resume='allow', **wandb_init)
        wandb.watch(model, log='all')
        print(f'\n\nWandb initialized with id {wandb.run.id}\n\n')

    results = {}
    #text_len_lim = 4096
    saved_checkpoints = []
    for epoch_ in range(args.epochs): # todo move epoch loop to model utils as is consistent across model types
        try: # for cuda out of memory errors
            epoch = epoch_ + epoch_prev
            #train_dataloader.sampler.set_epoch(epoch)

            # args, epoch, model, optim, schedular, train_dataloader, device, scaler, ema=None
            loss = train_one_epoch(args, epoch, model, optim, schedular, # create new train samples each epoch so the negatives change
                train_dataloader=train_sampler,
                device=device, 
                scaler=scaler, 
                ema=ema
            )
            #train_sampler.max_txt_len = min(text_len_lim, train_sampler.max_txt_len + 100) # increase max text len each epoch
            #print(f'Text length limit: {train_sampler.max_txt_len}')          

            try: # don't want this to crash if I accidentally delete the schedular config file
                schedular = update_schedular(args, optim, schedular) # allows changing of min n max learning rate during training
            except Exception as e:
                if os.path.exists(args.schedular_data) == False:
                    print('Schedular config file not found, creating new one')
                    save_schedular_data(args)
                else:
                    print(e, '\n') # todo move all this to update_schedular it looks ugly :P

            with ema.average_parameters() if exists(ema) else nullcontext():
                vloss = validate_one_epoch(
                    args,
                    model, 
                    Sampler(f_names = dev_data, batch_size = args.batch_size, tokenizer = tokenizer, shuffle = False, split_into_splits = 600), 
                    device, 
                    sanity_check=False
                )

            if args.wandb:
                wandb.log({'val_loss': vloss, 'epoch': epoch})

            print(f'\n\n\n--- Epoch {epoch}, Validation Loss: {vloss} ---\n\n\n')
            results[epoch] = {'loss': loss, 'vloss': vloss}
        
            if len(saved_checkpoints) < args.save_top_k:
                if exists(ema):
                    ema.store()
                    ema.copy_to() # save ema of the parameters
                path = save_checkpoint(args, model, optim, epoch, vloss)
                if exists(ema):
                    ema.restore()
                draw_text('High Score')
                saved_checkpoints.append({
                    'path': path,
                    'epoch': epoch,
                    'vloss': vloss
                })
            elif vloss < max(saved_checkpoints, key=lambda x: x['vloss'])['vloss']:
                if exists(ema):
                    ema.store()
                    ema.copy_to()
                path = save_checkpoint(args, model, optim, epoch, vloss)
                if exists(ema):
                    ema.restore()
                draw_text('High Score')
                saved_checkpoints.append({
                    'path': path,
                    'epoch': epoch,
                    'vloss': vloss
                })
                saved_checkpoints.sort(key=lambda x: x['vloss'])
                to_remove = saved_checkpoints.pop()
                os.remove(to_remove['path']) 

        except RuntimeError as e:
            if str(e).startswith('CUDA out of memory'):
                print('): CUDA out of memory!!!! trying to recover ); \n Please download more GPUs XD')
                torch.cuda.empty_cache()
                model = model.to(device)
                continue
            else:
                raise e

        finished = train_sampler.step()
        if finished:
            print('FINISHED TRAINING (: -- EXITING')
            break
            
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data', type=str, default='/store/store1/data/PG19/train/')
    parser.add_argument('--dev_data', type=str, default='/store/store1/data/PG19/validation')
    parser.add_argument('-batch','--batch_size', type=int, default=10)
    parser.add_argument('-ema', '--ema', default=None, type=float)


    parser.add_argument('--config', type=str, default='./experiment_configs/lm/decoder_pg19.yaml', required=True)
    parser.add_argument('--checkpoint', type=str, default='')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints')

    parser.add_argument('--schedular_data', type=str, default='./schedular_data.json')

    parser.add_argument('-device', '--device', type=str, default='auto')
    parser.add_argument('--project_name', default='FINETUNE-PG19-INTERSPEECH', type=str)
    parser = add_common_args(parser)

    args = parser.parse_args()

    args.device = 'cuda' if torch.cuda.is_available() else 'cpu' if args.device == 'auto' else args.device 


    if args.checkpoint != '':
        args.checkpoint = os.path.join(args.checkpoint_dir, args.checkpoint)

    if os.path.exists(args.checkpoint_dir) == False:
        os.mkdir(args.checkpoint_dir)

    if os.path.exists(args.schedular_data) == True:
        print(f'A schedular data with the name {args.schedular_data} already exists, please delete it if you want to start a new run')
        exit()

    main(args=args)