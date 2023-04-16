import argparse
import pickle as pkl
from Levenshtein import distance
from speachy.rescoring.tools import ( sort_hypothesis_by_recording, order_recordings_by_start_time, )
import numpy as np
from functools import reduce
import torch
import os
import traceback
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

def get_edit_distance(hyps, target): # THIS ISN'T RIGHT
    return list(map(lambda x: distance(x, target) / (len(target)+1), hyps))

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
        sub_batches.append({'utterances': [],'scores': [],'sb_lengths': [], 'durations': []})
        for el in batch:
            case = len(el) > i
            sub_batches[-1]['utterances'].append(el[i][0] if case else -1)
            sub_batches[-1]['scores'].append(el[i][1] if case else -1)
            sub_batches[-1]['sb_lengths'].append(len(el[i][0]) if case else -1)
            sub_batches[-1]['durations'].append(el[i][2] if case else -1)
     
    non_empty_indices = np.arange(len(sub_batches[0]['sb_lengths']))

    for i, sub_batch in enumerate(sub_batches):
        sb_utts, sb_scores, sb_lengths, sb_durations = [np.array(el, dtype=object) for el in (sub_batch['utterances'], sub_batch['scores'], sub_batch['sb_lengths'], sub_batch['durations'])]
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
            'durations': sb_durations,
            'non_empty': non_empty,
            'prev_fetch': prev_fetch 
        }
    
    for i, sub_batch in enumerate(sub_batches):
        padded_utts, acc_lengths = proc_utts(sub_batch['utterances'][sub_batch['non_empty']].tolist())
        sub_batches[i] = {
            'utterances': torch.as_tensor(padded_utts),
            'lengths': torch.as_tensor(acc_lengths),
            'scores': torch.tensor(flatten_nested_list((sub_batch['scores'][sub_batch['non_empty']]).tolist())),
            'durations': torch.tensor(sub_batch['durations'][sub_batch['non_empty']].astype(float)).to(torch.float32),
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
        duration = end_t - start_t
        if prev_end is None or (start_t - prev_end) > max_gap:
            samples.append([]) 
        if len(samples[-1]) >= num_utterances:
            samples.append([])
        hyps = utterance['beams'][0]
        hyps = list(map(lambda x: x['text'], list(hyps.values())))
        target = utterance['targets'][0]
        if target == '':
            continue
        hyps = list(filter(lambda el:el != target, hyps))
        hyps = np.random.choice(hyps, min(num_negatives, len(hyps)), replace=False).tolist()
        examples = [target] + hyps
        
        error_rates = get_edit_distance(examples, target)
        samples[-1].append((examples, error_rates, duration))
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

    def __generator__(self): 
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
        total_sub_batch_loss = torch.zeros(model.layers.depth, device=device)
        total_sub_batch_lengths = torch.zeros(model.layers.depth, device=device)
        for ix, sub_batch_ in enumerate(batch):
            sub_batch = batch_to_device(batch=sub_batch_, device=device, return_all=True)
            #print(f'sub_batch {ix+1} / {len(batch)}')

            prev_fetch, sb_lengths = sub_batch['prev_fetch'], sub_batch['sb_lengths']
            durations = sub_batch['durations'][:,None] if args.length_prediction else None
            
            tokens, token_lens = sub_batch['utterances'], sub_batch['lengths']
    
            if exists(prev_states) and exists(prev_fetch): # prev fetch is a list of indices that are present in the current sub-batch 
                sb_idxs = torch.arange(sb_lengths.size(0))
                sb_idxs = torch.cat([sb_idxs[ix].repeat(sb_lengths[ix]) for ix in range(sb_lengths.size(0))]).to(device)
                prev_fetch = prev_fetch[sb_idxs]
                for i in range(len(prev_states['layers'])):
                    prev_states['layers'][i]['cache'] = prev_states['layers'][i]['cache'][:, :, prev_fetch] 
                    prev_states['layers'][i]['cache_lengths'] = prev_states['layers'][i]['cache_lengths'][prev_fetch]
                for i in range(len(prev_states['next_sentence_pred'])):
                    prev_states['next_sentence_pred'][i] = prev_states['next_sentence_pred'][i][prev_fetch]
                    
    
            
            using_bos = not exists(prev_states) or args.bos_eos
            token_lens += 1 if using_bos else 0 # add 1 for bos if using
            tokens = add_bos(tokens, bos_token_id=0) if using_bos else tokens # add bos only if this is the first sub-batch
    
            outputs = model(
                labels=tokens,
                length=token_lens,
                cache=prev_states,
                calc_loss=True,
            )
            

            token_len_thing = (outputs['lengths']).sum(-1)
            token_len_thing[len(outputs['token_losses']):] = 0
            total_sub_batch_lengths += token_len_thing
            total_sub_batch_loss[:len(outputs['token_losses'])] += outputs['token_losses'] * token_len_thing[:len(outputs['token_losses'])]

            prev_states = outputs['cache']

        losses.append((total_sub_batch_loss / total_sub_batch_lengths).mean().item())
  
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


import random

def get_depth(p_stop):
    depth = 1
    while torch.rand(1).item() < p_stop:
        depth += 1
    return depth

def train_one_epoch(args, epoch, model, optim, schedular, train_dataloader, device, scaler, ema=None):
    model.train()
    train_dataloader = list(train_dataloader)
    pbar = tqdm(train_dataloader, total=len(train_dataloader))
    losses = []
    loss_iterim = []
    autocast_device = 'cuda' if torch.cuda.is_available() else 'cpu' # for autocast if using mixed precision
    print('Training epoch') 
    
    normal_depth = 5
    model.layers.depth = normal_depth
    p_stop = 0.5

    prev_loss = torch.zeros(10, device=device)
    for batch in pbar:
        model.layers.depth = get_depth(p_stop)
        print(f'current depth: {model.layers.depth}')
        prev_states = None
        total_sub_batch_loss = torch.zeros(model.layers.depth, device=device)
        total_sub_batch_lengths = torch.zeros(model.layers.depth, device=device)
        total_ntmseloss = torch.zeros(model.layers.depth, device=device)
        total_sub_batch_commit_loss = []



        with torch.autocast(device_type=autocast_device) if exists(scaler) else nullcontext(): # for mixed precision
            for ix, sub_batch_ in enumerate(batch):
                #print(f'sub_batch {ix+1} / {len(batch)}')
                sub_batch = batch_to_device(batch=sub_batch_, device=device, return_all=True) 

                tokens, token_lens = sub_batch['utterances'], sub_batch['lengths']
                durations = sub_batch['durations'][:,None] if args.length_prediction else None
                prev_fetch, sb_lengths = sub_batch['prev_fetch'], sub_batch['sb_lengths'] # scores are negative for the softmax
                sb_starts = sb_lengths.cumsum(dim=0) - sb_lengths 

           
                if exists(prev_states) and exists(prev_fetch): # prev fetch is a list of indices that are present in the current sub-batch 
                    sb_idxs = torch.arange(sb_lengths.size(0))
                    sb_idxs = torch.cat([sb_idxs[ix].repeat(sb_lengths[ix]) for ix in range(sb_lengths.size(0))]).to(device)
                    prev_fetch = prev_fetch[sb_idxs]
                    for i in range(len(prev_states['layers'])):
                        prev_states['layers'][i]['cache'] = prev_states['layers'][i]['cache'][:, :, prev_fetch] 
                        prev_states['layers'][i]['cache_lengths'] = prev_states['layers'][i]['cache_lengths'][prev_fetch]
                    for i in range(len(prev_states['next_sentence_pred'])):
                        prev_states['next_sentence_pred'][i] = prev_states['next_sentence_pred'][i][prev_fetch]
                       
                using_bos = not exists(prev_states) or args.bos_eos
                token_lens += 1 if using_bos else 0 # add 1 for bos if using
                tokens = add_bos(tokens, bos_token_id=0) if using_bos else tokens # add bos only if this is the first sub-batch


                outputs = model(
                    labels=tokens,
                    length=token_lens,
                    cache=prev_states,
                    calc_loss=True,
                )


                token_len_thing = (outputs['lengths']).sum(-1)
                token_len_thing[len(outputs['token_losses']):] = 0
                total_sub_batch_lengths += token_len_thing
                total_sub_batch_loss[:len(outputs['token_losses'])] += outputs['token_losses'] * token_len_thing[:len(outputs['token_losses'])]
                total_ntmseloss[:len(outputs['ntmselosses'])] += outputs['ntmselosses'] * token_len_thing[:len(outputs['ntmselosses'])]
                #total_codebook_usage += outputs['codebook_usage'] * token_len_thing[:len(outputs['token_losses'])][1:]
                total_sub_batch_commit_loss.append(outputs['commitment_loss'].sum() * token_len_thing.sum())
                prev_states = outputs['cache']
                
        total_sub_batch_loss = (total_sub_batch_loss / total_sub_batch_lengths)
        total_sub_batch_commit_loss = sum(total_sub_batch_commit_loss) / total_sub_batch_lengths.sum()
        
        total_ntmseloss = (total_ntmseloss / total_sub_batch_lengths)
      
        #total_codebook_usage = (total_codebook_usage / total_sub_batch_lengths[1:])
        #per_layer_codebook_usage = {f'codebook_usage_layer_{i+1}': total_codebook_usage[i].item() for i in range(len(total_codebook_usage))}
        
        per_layer_loss = {f'loss_layer_{i}': total_sub_batch_loss[i].item() for i in range(len(total_sub_batch_loss))}
        per_layer_loss_ntmse = {f'loss_ntmse_layer_{i}': total_ntmseloss[i].item() for i in range(len(total_ntmseloss))}
        

        total_sub_batch_loss = total_sub_batch_loss.mean()
        total_sub_batch_loss_todisplay = total_sub_batch_loss.item()
        total_sub_batch_loss = total_sub_batch_loss + total_ntmseloss.mean() + total_sub_batch_commit_loss

        
        #total_sub_batch_commit_loss = sum(total_sub_batch_commit_loss) / len(total_sub_batch_commit_loss)
        #total_sub_batch_loss += total_sub_batch_commit_loss
        total_sub_batch_lengths = 0
        losses.append(total_sub_batch_loss_todisplay)
        scaler.scale(total_sub_batch_loss).backward() if exists(scaler) else total_sub_batch_loss.backward()
        '''if args.clip_gradients == True:
            scaler.unscale_(optim) if exists(scaler) else None
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_gradients_value) '''
        scaler.step(optim) if exists(scaler) else optim.step()
        scaler.update() if exists(scaler) else None
        optim.zero_grad()

        if exists(schedular):
            schedular.step()
        if exists(ema):
            ema.update()
        cur_lr = schedular.get_last_lr()[0] if exists(schedular) else optim.param_groups[0]['lr']

        if args.wandb:
            wandb.log({
                'train_loss': total_sub_batch_loss_todisplay,
                'lrate': cur_lr,
                'epoch': epoch,
                **per_layer_loss,
                **per_layer_loss_ntmse,
            })

            
    loss_end = sum(losses) / len(losses)
    if args.wandb:
        wandb.log({'train_loss_end': loss_end, 'epoch': epoch})

    torch.cuda.empty_cache()
    model.layers.depth = normal_depth

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

    ema = ExponentialMovingAverage(model.parameters(), decay=0.99999) 
    scaler = GradScaler() if args.mixed_precision else None

    run_config = {'run_args': args.__dict__, 'model_config': config, 'total_params': total_params}
    wandb_init = {'project':args.project_name, 'config':run_config}
    if args.wandb:
        wandb.init(**wandb_init
        ) if args.wandb_id == '' else wandb.init(id=args.wandb_id, resume='allow', **wandb_init)
        wandb.watch(model, log='all')
        print(f'\n\nWandb initialized with id {wandb.run.id}\n\n')

    for p in model.parameters(): # backward hook to clip gradients
        if p.requires_grad:
            p.register_hook(lambda grad: torch.clamp(grad, -args.clip_gradients_value, args.clip_gradients_value))

    results = {}
    saved_checkpoints = []
    for epoch_ in range(args.epochs): # todo move epoch loop to model utils as is consistent across model types
        try:
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
        except RuntimeError as e:
            print(f'RuntimeError: {e}')
            # get full traceback
            traceback.print_exc()

            if str(e).startswith('CUDA out of memory'):
                print('CUDA out of memory, trying to recover')
                torch.cuda.empty_cache()
                continue
            
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_hyp', type=str, required=True)
    parser.add_argument('--dev_hyp', type=str, required=True)
    parser.add_argument('-utts','--utts_per_sample', type=int, default=5)
    parser.add_argument('-batch','--batch_size', type=int, default=3)
    parser.add_argument('-mgap','--max_allowed_utterance_gap', type=float, default=10.0, help='max allowed gap between utterances in seconds')

    parser.add_argument('-bos_eos', '--bos_eos', action='store_true', help='always use bos and eos tokens')
    parser.add_argument('-length_pred','--length_prediction', action='store_true', help='use length prediction')

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

    if args.length_prediction:
        assert args.bos_eos, 'bos eos must be used if length prediction is used'

    if args.checkpoint != '':
        args.checkpoint = os.path.join(args.checkpoint_dir, args.checkpoint)

    if os.path.exists(args.checkpoint_dir) == False:
        os.mkdir(args.checkpoint_dir)

    if os.path.exists(args.schedular_data) == True:
        print(f'A schedular data with the name {args.schedular_data} already exists, please delete it if you want to start a new run')
        exit()

    main(args=args)