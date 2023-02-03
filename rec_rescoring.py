import argparse
import pickle as pkl
from Levenshtein import distance
from speachy.rescoring.tools import ( sort_hypothesis_by_recording, order_recordings_by_start_time, )
import numpy as np
from functools import reduce
import torch
import os

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

INTERPOLATE = 0.5

def get_edit_distance(hyps, target):
    return list(map(lambda x: distance(x, target) / len(target), hyps))

flatten_nested_list = lambda l: [item for sublist in l for item in sublist]
    
def tokenize_and_pad(utterances, tokenizer): # returns padded utterances and their lengths
    tokenized = [tokenizer.text_to_ids(utt) for utt in utterances]
    max_len = max(map(len, tokenized))
    padded_utts = np.array([utt + [0] * (max_len - len(utt)) for utt in tokenized])
    return padded_utts, np.array(list(map(len, tokenized)))

def get_sub_batches(batch, tokenizer):  
    proc_utts = lambda utts: tokenize_and_pad(flatten_nested_list(utts), tokenizer)
    sub_batches = []
    max_len = max(map(len, batch))
    for i in range(max_len):
        sub_batches.append({'utterances': [],'scores': [],'sb_lengths': [],})
        for el in batch:
            case = len(el) > i
            sub_batches[-1]['utterances'].append(el[i][0] if case else -1)
            sub_batches[-1]['scores'].append(el[i][1] if case else -1)
            sub_batches[-1]['sb_lengths'].append(len(el[i][0]) if case else -1)
     
    non_empty_indices = np.arange(len(sub_batches[0]['sb_lengths']))

    for i, sub_batch in enumerate(sub_batches):
        sb_utts, sb_scores, sb_lengths = [np.array(el, dtype=object) for el in (sub_batch['utterances'], sub_batch['scores'], sub_batch['sb_lengths'])]
        non_empty = non_empty_indices[sb_lengths != -1]
        # slice based on non empty of previous sub batch
        prev_fetch = None
        if i != 0:
            prev_lengths = sub_batches[i-1]['sb_lengths']
            prev_non_empty = sub_batches[i-1]['non_empty']
            diff_from = ((prev_lengths != -1) == (sb_lengths != -1))[prev_non_empty]
            prev_fetch = np.arange(len(prev_non_empty))[diff_from]
            
        sub_batches[i] = {
            'utterances': sb_utts,
            'scores': sb_scores,
            'sb_lengths': sb_lengths,
            'non_empty': non_empty,
            'prev_fetch': prev_fetch 
        }
    
    for i, sub_batch in enumerate(sub_batches):
        padded_utts, acc_lengths = proc_utts(sub_batch['utterances'][sub_batch['non_empty']].tolist())
        sub_batches[i] = {
            'utterances': torch.as_tensor(padded_utts),
            'lengths': torch.as_tensor(acc_lengths),
            'scores': torch.tensor(flatten_nested_list((sub_batch['scores'][sub_batch['non_empty']]).tolist())),
            'sb_lengths': torch.as_tensor(sub_batch['sb_lengths'][sub_batch['non_empty']].astype(int)),
            'prev_fetch': torch.as_tensor(sub_batch['prev_fetch']) if sub_batch['prev_fetch'].__class__.__name__ != 'NoneType' else None# indices to fetch the states from previous sub batch
        }
    return sub_batches


    
def create_samples_from_recording(recording, num_utterances, num_negatives, max_gap=10.0, shuffle=True):
    samples = []
    prev_end = None
    if shuffle == False:
        np.random.seed(42) # deterministic selection of negatives
    for i, utterance in enumerate(recording):
        start_t, end_t = utterance['meta_data']['timings'].values()
        if prev_end is None or (start_t - prev_end) > max_gap:
            samples.append([]) 
        if len(samples[-1]) >= num_utterances:
            samples.append([])
        hyps = utterance['beams'][0]
        hyps = list(map(lambda x: x['text'], list(hyps.values())))
        target = utterance['targets'][0]
        hyps = list(filter(lambda el:el != target, hyps))
        hyps = np.random.choice(hyps, min(num_negatives, len(hyps)), replace=False).tolist()
        examples = [target] + hyps
        error_rates = get_edit_distance(examples, target)
        samples[-1].append((examples, error_rates))
        prev_end = end_t

    return samples

def create_dataset_samples(recordings, num_utterances, num_negatives, max_gap=10.0, shuffle=True):
    samples = []
    for recording in recordings.keys():
        samples += create_samples_from_recording(recordings[recording], num_utterances, num_negatives, max_gap, shuffle)
    if shuffle:
        np.random.shuffle(samples) # shuffle samples
    return samples


class Sampler(object):
    def __init__(self, samples, batch_size, tokenizer, shuffle=True):
        self.samples = samples
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.shuffle = shuffle
        self.sample_indices = np.arange(len(self.samples))
        np.random.shuffle(self.sample_indices) if self.shuffle else None

        self.generator = self.__generator__() 

    def __generator__(self): # pre-compute this, prepping batches on-the-fly is hurting utilization )::::
        for i in range(0, len(self.samples), self.batch_size):
            yield get_sub_batches([self.samples[i] for i in self.sample_indices[i:i+self.batch_size]], self.tokenizer)

    def __list__(self):
        return list(self.generator)

    def __iter__(self):
        return self

    def __len__(self):
        return len(self.samples) // self.batch_size

    def __next__(self):
        return next(self.generator)
      
@torch.no_grad()
def validate_one_epoch_________(model, val_dataloader, device, sanity_check=False):
    model.eval()
    pbar = tqdm(val_dataloader, total=len(val_dataloader))
    wers = []
    losses = []
    print('Evaluation epoch')
    for batch in pbar:
        tokens, token_lens = batch_to_device(batch, device)
    
        token_lens += 1 # add 1 for bos
        tokens = add_bos(tokens, bos_token_id=0)
        targets = tokens.clone()
        targets[:, :-1] = tokens[:, 1:]
        targets = add_eos(targets, eos_id=0, token_lens=token_lens)
        
        mask = token_lens_to_mask(token_lens)
        targets = mark_padding(targets, mask, pad_id=-100)

        model_args = {'x': tokens, 'mask': mask} if isfalse(callable(getattr(model, 'get_args', False))) \
            else model.get_args(tokens=tokens, mask=mask, lengths=token_lens)

        model_out = model(**model_args)
        
        if callable(getattr(model, 'process_labels', None)):
            targets = model.process_labels(targets)

        loss = loss_ce(logits=model_out, labels=targets, ignore_index=-100)
        losses.append(loss.item())

        if sanity_check:
            return True

    return  sum(losses) / len(losses)

'''
                    sb_idxs = torch.arange(sb_lengths.size(0))
                    sb_idxs = torch.cat([sb_idxs[ix].repeat(sb_lengths[ix]) for ix in range(sb_lengths.size(0))]).to(device)
                    prev_fetch = prev_fetch[sb_idxs]
                    prev_states['cache'] = prev_states['cache'][:, :, prev_fetch] 
                    prev_states['cache_lengths'] = prev_states['cache_lengths'][prev_fetch]
'''

@torch.no_grad()
def validate_one_epoch(args, model, val_dataloader, device, sanity_check=False):
    model.eval()
    val_dataloader = list(val_dataloader)
    pbar = tqdm(val_dataloader, total=len(val_dataloader))
    losses = []
    print('Evaluation epoch')
    for batch in pbar:
        prev_states = None
        total_sb_losses = []
        total_lengths = 0
        for ix, sub_batch_ in enumerate(batch):
            sub_batch = batch_to_device(batch=sub_batch_, device=device, return_all=True)
            #print(f'sub_batch {ix+1} / {len(batch)}')

            prev_fetch, sb_lengths = sub_batch['prev_fetch'], sub_batch['sb_lengths']
            tokens, token_lens = sub_batch['utterances'], sub_batch['lengths']
     
            if exists(prev_states) and exists(prev_fetch): # prev fetch is a list of indices that are present in the current sub-batch 
                sb_idxs = torch.arange(sb_lengths.size(0))
                sb_idxs = torch.cat([sb_idxs[ix].repeat(sb_lengths[ix]) for ix in range(sb_lengths.size(0))]).to(device)
                prev_fetch = prev_fetch[sb_idxs]
                prev_states['cache'] = prev_states['cache'][:, :, prev_fetch] 
                prev_states['cache_lengths'] = prev_states['cache_lengths'][prev_fetch]
                if args.shift_states:
                    top_states = prev_states['cache'][-1][None]
                    new_states = torch.cat([prev_states['cache'], top_states])
                    prev_states['cache'] = new_states[1:] # shift states down by 1
            
            token_lens += 1 if not exists(prev_states) else 0 # add 1 for bos if using
            tokens = add_bos(tokens, bos_token_id=0) if not exists(prev_states) else tokens # add bos only if this is the first sub-batch
            targets = tokens.clone()
            targets[:, :-1] = tokens[:, 1:] # shift 
            targets = add_eos(targets, eos_id=-100, token_lens=token_lens)
            mask = token_lens_to_mask(token_lens)
            targets = mark_padding(targets, mask, pad_id=-100)

            logits, _, cached_kvs = model(x=tokens, length=token_lens, cache=prev_states)
            loss = loss_ce(logits=logits, labels=targets, ignore_index=-100)

            total_lengths += token_lens.sum().item()
            total_sb_losses.append(loss.item() * token_lens.sum().item())
            
            target_positions = sb_lengths.cumsum(dim=0) - sb_lengths # get the first position of each sub-batch
            
            prev_states = { # cache is [L, KV, B, H, N, D] ! (L = layers, KV = key/value, B = batch, H = heads, N = num tokens, D = dim)
                'cache_lengths': cached_kvs['cache_lengths'][target_positions],
                'cache': cached_kvs['cache'][:, :, target_positions] # only keep states from the "gold" hypothesis
            } if exists(cached_kvs) else None

        losses.append(sum(total_sb_losses) / total_lengths)
  
        if sanity_check:
            return True

    torch.cuda.empty_cache()
    return  sum(losses) / len(losses)




def intermediate_loss(loss_fn, interim_logits, targets):
    if not exists(interim_logits):
        return None
    interims = torch.empty(interim_logits.shape[0], device=interim_logits.device)
    for i in range(interim_logits.shape[0]):
        interims[i] = loss_fn(interim_logits[i], targets)
    return torch.mean(interims)



def train_one_epoch(args, epoch, model, optim, schedular, train_dataloader, device, scaler, ema=None):
    model.train()
    train_dataloader = list(train_dataloader)
    pbar = tqdm(train_dataloader, total=len(train_dataloader))
    losses = []
    loss_iterim = []
    autocast_device = 'cuda' if torch.cuda.is_available() else 'cpu' # for autocast if using mixed precision
    print('Training epoch')
    for batch in pbar:
        prev_states = None
        total_sub_batch_loss = 0
        total_sub_batch_lengths = 0
        with torch.autocast(device_type=autocast_device) if exists(scaler) else nullcontext(): # for mixed precision
            for ix, sub_batch_ in enumerate(batch):
                #print(f'sub_batch {ix+1} / {len(batch)}')
                sub_batch = batch_to_device(batch=sub_batch_, device=device, return_all=True) 

                tokens, token_lens = sub_batch['utterances'], sub_batch['lengths']
                prev_fetch, sb_lengths = sub_batch['prev_fetch'], sub_batch['sb_lengths'] # scores are negative for the softmax
                sb_starts = sb_lengths.cumsum(dim=0) - sb_lengths 


                if exists(prev_states) and exists(prev_fetch): # prev fetch is a list of indices that are present in the current sub-batch 
                    sb_idxs = torch.arange(sb_lengths.size(0))
                    sb_idxs = torch.cat([sb_idxs[ix].repeat(sb_lengths[ix]) for ix in range(sb_lengths.size(0))]).to(device)
                    prev_fetch = prev_fetch[sb_idxs]
                    prev_states['cache'] = prev_states['cache'][:, :, prev_fetch] 
                    prev_states['cache_lengths'] = prev_states['cache_lengths'][prev_fetch]
                    if args.shift_states:
                        top_states = prev_states['cache'][-1][None]
                        new_states = torch.cat([prev_states['cache'], top_states])
                        prev_states['cache'] = new_states[1:] # shift states down by 1
        

                token_lens += 1 if not exists(prev_states) else 0 # add 1 for bos if using
                tokens = add_bos(tokens, bos_token_id=0) if not exists(prev_states) else tokens # add bos only if this is the first sub-batch
                targets = tokens.clone()
                targets[:, :-1] = tokens[:, 1:] # shift 
                targets = add_eos(targets, eos_id=-100, token_lens=token_lens)
                mask = token_lens_to_mask(token_lens)
                targets = mark_padding(targets, mask, pad_id=-100)

                logits, interim_posteriors, cached_kvs = model(x=tokens, length=token_lens, cache=prev_states)
                
                loss = loss_ce(logits=logits, labels=targets, ignore_index=-100)
                
                interim_loss = intermediate_loss(loss_ce, interim_posteriors, targets)
                loss = interim_loss * INTERPOLATE + loss * (1 - INTERPOLATE) if exists(interim_loss) else loss
                total_sub_batch_lengths += token_lens.sum()
                total_sub_batch_loss += loss * token_lens.sum() 
  
                
                target_positions = sb_starts
                prev_states = { # cache is [L, KV, B, H, N, D] ! (L = layers, KV = key/value, B = batch, H = heads, N = num tokens, D = dim)
                    'cache_lengths': cached_kvs['cache_lengths'][target_positions],
                    'cache': cached_kvs['cache'][:, :, target_positions] # only keep states from the "gold" hypothesis
                } if exists(cached_kvs) else None


        total_sub_batch_loss = total_sub_batch_loss / total_sub_batch_lengths
        total_sub_batch_lengths = 0
        losses.append(total_sub_batch_loss.item())
        scaler.scale(total_sub_batch_loss).backward() if exists(scaler) else total_sub_batch_loss.backward()
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
            wandb.log({'train_loss': total_sub_batch_loss, 'lrate': cur_lr})
            
    loss_end = sum(losses) / len(losses)
    if args.wandb:
        wandb.log({'train_loss_end': loss_end, 'epoch': epoch})

    torch.cuda.empty_cache()
    return loss_end


def main(args):

    device, config = torch.device(args.device), load_config(args.config)
    tokenizer_path = os.path.join(config['model']['tokenizer']['dir'], 'tokenizer.model')
    tokenizer = load_tokenizer(model_path=tokenizer_path)
    model = autoload(config=config, tokenizer=tokenizer)

    get_data = lambda hyp: order_recordings_by_start_time(sort_hypothesis_by_recording(load_pkl(hyp)))
    train_data, dev_data = get_data(args.train_hyp), get_data(args.dev_hyp)

    epoch_prev = 0
    if args.checkpoint != '':
        epoch_prev, val_loss = load_checkpoint(args=args, model=model, force_cpu=True)
        modeltype = config['model']['modeltype']
        print(f'Loaded model {args.checkpoint} with epoch {epoch_prev} and val_loss {val_loss}\n Model type: {modeltype}')
    model.to(device)

    optim, schedular = optimizer(model, args)
    save_schedular_data(args)

    total_params = get_parameters(model=model, verbose=True)
    
    val_samples = create_dataset_samples(
        dev_data,
        num_utterances = args.utts_per_sample, 
        num_negatives = 0,
        max_gap = args.max_allowed_utterance_gap,
        shuffle = False 
    )

    validate_one_epoch(
        args,
        model = model,
        val_dataloader = Sampler(val_samples, batch_size=args.batch_size, tokenizer=tokenizer, shuffle=False),
        device = device,
        sanity_check = True
    )

    train_sample_args = {
        'recordings': train_data,
        'num_utterances': args.utts_per_sample,
        'num_negatives': 0,
        'max_gap': args.max_allowed_utterance_gap,
        'shuffle': True,
    }

    ema = ExponentialMovingAverage(model.parameters(), decay=0.9999) 
    scaler = GradScaler() if args.mixed_precision else None

    run_config = {'run_args': args.__dict__, 'model_config': config, 'total_params': total_params}
    wandb_init = {'project':args.project_name, 'config':run_config}
    if args.wandb:
        wandb.init(**wandb_init
        ) if args.wandb_id == '' else wandb.init(id=args.wandb_id, resume='allow', **wandb_init)
        wandb.watch(model, log='all')
        print(f'\n\nWandb initialized with id {wandb.run.id}\n\n')

    results = {}
    saved_checkpoints = []
    for epoch_ in range(args.epochs): # todo move epoch loop to model utils as is consistent across model types
        epoch = epoch_ + epoch_prev
        #train_dataloader.sampler.set_epoch(epoch)

        # args, epoch, model, optim, schedular, train_dataloader, device, scaler, ema=None
        loss = train_one_epoch(args, epoch, model, optim, schedular, # create new train samples each epoch so the negatives change
            train_dataloader=Sampler(create_dataset_samples(**train_sample_args), batch_size=args.batch_size, tokenizer=tokenizer, shuffle=True),
            device=device, 
            scaler=scaler, 
            ema=ema
        )          

        try: # don't want this to crash if I accidentally delete the schedular config file
            schedular = update_schedular(args, optim, schedular) # allows changing of min n max learning rate during training
        except Exception as e:
            if os.path.exists(args.schedular_data) == False:
                print('Schedular config file not found, creating new one')
                save_schedular_data(args)
            else:
                print(e, '\n') # todo move all this to update_schedular it looks ugly :P

        with ema.average_parameters(): # evaluate using ema of the parameters
            vloss = validate_one_epoch(
                args,
                model, 
                Sampler(val_samples, batch_size=args.batch_size, tokenizer=tokenizer, shuffle=False), 
                device, 
                sanity_check=False
            )

        if args.wandb:
            wandb.log({'val_loss': vloss, 'epoch': epoch})

        print(f'\n\n\n--- Epoch {epoch}, Validation Loss: {vloss} ---\n\n\n')
        results[epoch] = {'loss': loss, 'vloss': vloss}
    
        if len(saved_checkpoints) < args.save_top_k:
            ema.store()
            ema.copy_to() # save ema of the parameters
            path = save_checkpoint(args, model, optim, epoch, vloss)
            ema.restore()
            draw_text('High Score')
            saved_checkpoints.append({
                'path': path,
                'epoch': epoch,
                'vloss': vloss
            })
        elif vloss < max(saved_checkpoints, key=lambda x: x['vloss'])['vloss']:
            ema.store()
            ema.copy_to()
            path = save_checkpoint(args, model, optim, epoch, vloss)
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
            
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_hyp', type=str, required=True)
    parser.add_argument('--dev_hyp', type=str, required=True)
    parser.add_argument('-utts','--utts_per_sample', type=int, default=5)
    parser.add_argument('-batch','--batch_size', type=int, default=3)
    parser.add_argument('-mgap','--max_allowed_utterance_gap', type=float, default=10.0, help='max allowed gap between utterances in seconds')

    parser.add_argument('--shift_states', action='store_true', help='shift states upwards for enhanced reccurence https://arxiv.org/abs/2012.15688')

    parser.add_argument('--config', type=str, default='./experiment_configs/lm/decoder_pg19.yaml', required=True)
    parser.add_argument('--checkpoint', type=str, default='')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints')

    parser.add_argument('--schedular_data', type=str, default='./schedular_data.json')

    parser.add_argument('--max_utt_gap', type=float, default=10.0)

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