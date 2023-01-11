import argparse
from random import sample
import torch

import tools
import non_iid_dataloader
import os
from tqdm import tqdm
import numpy as np
import wandb
from torch.optim.lr_scheduler import CyclicLR
from model_utils import load_checkpoint, load_nemo_checkpoint, write_to_log, \
    squeeze_batch_and_to_device, save_checkpoint, draw_text, load_schedular_data, save_schedular_data

from contextlib import nullcontext
from nemo.collections.asr.models.rnn_scctc_bpe_models import RNNEncDecSCCTCModelBPE as ModelClass


from speachy.asr.utils import load_audio_model as load_model
from speachy.utils.helpers import  exists, isfalse, istrue
from speachy.utils.general.training_loop import optimizer, update_schedular


import torch_optimizer
from torch_ema import ExponentialMovingAverage
# gradscaler
from torch.cuda.amp import GradScaler   

INTERPOLATE = 0.5



def create_subbatches(audio, audio_lens, tokens, token_lens, segment_lens): # for loops ):
    max_segment_len = segment_lens.max()

    culm_seglens = segment_lens.cumsum(dim=0)
    cur_positions = culm_seglens - segment_lens
    sub_batches_indices = []

    # first get indices for each sub batch of the "rnn"
    for ix in range(max_segment_len):
        indices = []
        for iz in range(len(segment_lens)):
            pos = cur_positions[iz].item()
            if pos < culm_seglens[iz]:
                indices.append(pos)
                cur_positions[iz] += 1
            else:
                indices.append(-1)
        sub_batches_indices.append(torch.tensor(indices, dtype=torch.long))
    ####
    ### after each forward pass the model will return the cached kvs
    # this gets the indices of the correct kvs for the next forward pass
    non_empty_indices = torch.arange(len(segment_lens), dtype=torch.long)
    prev_non_empty_fetch = []
    for i in range(len(sub_batches_indices)):
        cur = sub_batches_indices[i]
        cur = cur[sub_batches_indices[i-1] != -1] if i > 0 else cur
        non_empty_indices = non_empty_indices[cur != -1]
        prev_non_empty_fetch.append(non_empty_indices.clone())
        non_empty_indices = torch.arange(len(non_empty_indices), dtype=torch.long)
    ####
    sub_batches = []
    for i, ix in enumerate(sub_batches_indices):
        sbi = ix[ix != -1]
        cur_audio, cur_audio_lens, cur_tokens, cur_token_lens = audio[sbi], audio_lens[sbi], tokens[sbi], token_lens[sbi]
        # trim audio and tokens to max length in sub batch
        max_cur_audio_len, max_cur_token_len = cur_audio_lens.max(), cur_token_lens.max()
        cur_audio, cur_tokens = cur_audio[:, :max_cur_audio_len], cur_tokens[:, :max_cur_token_len]
        sub_batches.append({
            'audio': cur_audio,
            'audio_lens': cur_audio_lens,
            'tokens': cur_tokens,
            'token_lens': cur_token_lens,
            'prev_state_indices': prev_non_empty_fetch[i] if i > 0 else None, # for the first sub batch there is no previous state  
        })
    return sub_batches
    
def move_to_device(sub_batch, device):
    for k, v in sub_batch.items():
        if isinstance(v, torch.Tensor):
            sub_batch[k] = v.to(device)
    return sub_batch



@torch.no_grad()
def validate_one_epoch(epoch, model, val_dataloader, device, sanity_check=False):
    model.eval()
    pbar = tqdm(val_dataloader, total=len(val_dataloader))
    wers = []
    losses = []
    loss_iterim = []
    
    print('Evaluation epoch')
    for ix, batch in enumerate(pbar):
        sub_batches = create_subbatches(**batch)
        prev_states, prev_state_lens = None, None
      
        for iz, sub_batch in enumerate(sub_batches):
            sub_batch = move_to_device(sub_batch, device)
            targets, target_lengths = sub_batch['tokens'], sub_batch['token_lens']

            if exists(prev_states) and exists(sub_batch['prev_state_indices']) and exists(prev_state_lens):
                prev_states, prev_state_lens = prev_states[:,sub_batch['prev_state_indices']], prev_state_lens[sub_batch['prev_state_indices']]
                state_data = move_to_device({'kvs': prev_states, 'kv_lens': prev_state_lens}, device=device)
                prev_states, prev_state_lens = state_data['kvs'], state_data['kv_lens']
                
       
            model_inputs = {'input_signal': sub_batch['audio'], 'input_signal_length': sub_batch['audio_lens'], 'cached_kvs': prev_states, 'cached_kv_lens': prev_state_lens}
            model_out = model.forward(**model_inputs)
            log_probs, interim_posteriors, encoded_len, additional_outputs = model_out[0], model_out[1], model_out[2], model_out[-1]
            cached_kvs, full_kv_lens = additional_outputs['kvs_to_cache'], additional_outputs['full_kv_lens']
            state_data = move_to_device({'kvs': cached_kvs, 'kv_lens': full_kv_lens}, 'cpu')
            prev_states, prev_state_lens = state_data['kvs'], state_data['kv_lens']
    
            #print(prev_states.shape, prev_state_lens.shape)

            if exists(interim_posteriors):
                interims = torch.empty(interim_posteriors.shape[0]).to(device)
                for ix, layer in enumerate(interim_posteriors):
                    interim = model.loss(log_probs=layer, targets=targets, input_lengths=encoded_len, target_lengths=target_lengths)
                    interims[ix] = interim
                interim_loss = torch.mean(interims)

            loss_final = model.loss(log_probs=log_probs, targets=targets, input_lengths=encoded_len, target_lengths=target_lengths)
            loss_a = loss_final if not exists(interim_posteriors) else INTERPOLATE * interim_loss + (1 - INTERPOLATE) * loss_final
            # diff from batch size
            loss_fraction = sub_batch['audio'].shape[0] / batch['audio'].shape[0]
            loss_iterim.append(loss_a * sub_batch['audio'].shape[0])
            loss_a = loss_a * loss_fraction # so lr is not affected when sbatch size is different from batch size (different seq len)

        cur_loss = sum(loss_iterim) / batch['audio'].shape[0]
        loss_iterim = []
        losses.append(cur_loss)

        if sanity_check:
            return True

    loss_end = sum(losses) / len(losses)

    if args.wandb:
        wandb.log({'val_loss': loss_end, 'epoch': epoch})   

    torch.cuda.empty_cache() # to avoid memory leaks
    return loss_end


def train_one_epoch(epoch, model, optim, schedular, train_dataloader, device, scaler=None, ema=None):
    model.train()
    losses = [] # for storing effective losses
    loss_iterim = [] # for storing loss of each step
    print('Training epoch')
    pbar = tqdm(train_dataloader, total=len(train_dataloader))
    autocast_device = 'cuda' if torch.cuda.is_available() else 'cpu' # for autocast if using mixed precision
 
    for ix, batch in enumerate(pbar):
        sub_batches = create_subbatches(**batch)
        prev_states = None
        prev_state_lens = None
        for iz, sub_batch in enumerate(sub_batches):
            sub_batch = move_to_device(sub_batch, device)
            targets, target_lengths = sub_batch['tokens'], sub_batch['token_lens']

            if exists(prev_states) and exists(sub_batch['prev_state_indices']) and exists(prev_state_lens):
                prev_states, prev_state_lens = prev_states[:,sub_batch['prev_state_indices']], prev_state_lens[sub_batch['prev_state_indices']]
                state_data = move_to_device({'kvs': prev_states, 'kv_lens': prev_state_lens}, device=device)
                prev_states, prev_state_lens = state_data['kvs'], state_data['kv_lens']

            with torch.autocast(device_type=autocast_device) if exists(scaler) else nullcontext(): # for mixed precision
                model_inputs = {
                    'input_signal': sub_batch['audio'],
                    'input_signal_length': sub_batch['audio_lens'],
                    'cached_kvs': prev_states,
                    'cached_kv_lens': prev_state_lens,
                }

                model_out = model.forward(**model_inputs)
                log_probs, interim_posteriors, encoded_len, additional_outputs = model_out[0], model_out[1], model_out[2], model_out[-1]
                cached_kvs, full_kv_lens, commit_loss = additional_outputs['kvs_to_cache'], additional_outputs['full_kv_lens'], additional_outputs['commit_loss']
                state_data = move_to_device({'kvs': cached_kvs, 'kv_lens': full_kv_lens}, 'cpu')
                prev_states, prev_state_lens = state_data['kvs'], state_data['kv_lens']

                if exists(interim_posteriors):
                    interims = torch.empty(interim_posteriors.shape[0]).to(device)
                    for ix, layer in enumerate(interim_posteriors):
                        interim = model.loss(log_probs=layer, targets=targets, input_lengths=encoded_len, target_lengths=target_lengths)
                        interims[ix] = interim
                    interim_loss = torch.mean(interims)

                loss_final = model.loss(log_probs=log_probs, targets=targets, input_lengths=encoded_len, target_lengths=target_lengths)
                loss_a = loss_final if not exists(interim_posteriors) else INTERPOLATE * interim_loss + (1 - INTERPOLATE) * loss_final
                #print(commit_loss, loss_a)
                loss_a += commit_loss
                # diff from batch size
                loss_fraction = sub_batch['audio'].shape[0] / batch['audio'].shape[0]
                loss_iterim.append(loss_a * sub_batch['audio'].shape[0])
                loss_a = loss_a * loss_fraction # so lr is not affected when sbatch size is different from batch size (different seq len)
        

            scaler.scale(loss_a).backward() if exists(scaler) else loss_a.backward() # no BPTT 
            if args.clip_gradients == True:
                scaler.unscale_(optim) if exists(scaler) else None
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_gradients_value)

            scaler.step(optim) if exists(scaler) else optim.step()
            scaler.update() if exists(scaler) else None
            optim.zero_grad() 

            if exists(schedular):
                schedular.step()
            if exists(ema):
                ema.update() # update the moving average of the parameters
            cur_lr = schedular.get_last_lr()[0] if exists(schedular) else optim.param_groups[0]['lr']

            if exists(prev_states):
                prev_states = prev_states.detach()

        cur_loss = sum(loss_iterim) / batch['audio'].shape[0]
        loss_iterim = []
        losses.append(cur_loss)
        if args.wandb:
            wandb.log({'train_loss': cur_loss, 'lrate': cur_lr})
        pbar.set_description(f'Loss: {cur_loss}, lrate: {cur_lr}')  

    loss_end = sum(losses) / len(losses)
    if args.wandb:
        wandb.log({'train_loss_end': loss_end, 'epoch': epoch})

    torch.cuda.empty_cache() # to avoid memory leaks

    return loss_end

def main(args):
    model = load_model(args=args, model_class=ModelClass)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
 
    ami_dict = tools.load_corpus()
    tokenizer = model.tokenizer
 
    get_dl = lambda split: non_iid_dataloader.get_data_loader(
        split=ami_dict[split], 
        tokenizer=tokenizer, 
        shuffle=True if split == 'train' else False,
        max_duration=args.micro_batch_duration, 
        num_workers=args.num_workers, 
        batch_size=args.micro_batch_number, 
        concat_samples=False,
    )

    train_dataloader, dev_dataloader = get_dl('train'), get_dl('dev')
    if args.run_test == True:
        test_dataloader = get_dl('test')
 
    optim, schedular = optimizer(model, args)
    save_schedular_data(args)

    epoch_prev = 0
    if args.checkpoint != '':
        epoch_prev, _ = load_checkpoint(args, model, optim) if args.nemo_checkpoint == False else load_nemo_checkpoint(args, model, optim)

    # print parameters
    total_params = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            total_params += param.numel()
    print(f'\nTotal number of parameters: {total_params/1e6}M\n')
           

    if args.run_test == True:
        lossval = validate_one_epoch(None, model, test_dataloader, device, sanity_check=False)
        print(f' Test loss: {lossval}')
        write_to_log(args.log_file, f'Test loss: {lossval} checkpoint: {args.checkpoint}')
        return
    
    validate_one_epoch(epoch=None, model=model, val_dataloader=dev_dataloader, device=device, sanity_check=True) # check no crash

    ema = ExponentialMovingAverage(model.parameters(), decay=0.9999) 
    scaler = GradScaler() if args.mixed_precision else None

    if args.wandb:
        wandb.init(project="deliberation-Custom_ami", config=args) if args.wandb_id == '' else wandb.init(project="deliberation-Custom_ami", id=args.wandb_id, resume="must", config=args)
        wandb.watch(model, log="all")
        wandb.config.update({'total_params': total_params})
        print(f'\nLoggging with Wandb id: {wandb.run.id}\n')

    results = {}
    saved_checkpoints = []
    for epoch_ in range(args.epochs): # todo move epoch loop to model utils as is consistent across model types
        epoch = epoch_ + epoch_prev
        #train_dataloader.sampler.set_epoch(epoch)

        loss = train_one_epoch(epoch, model, optim, schedular, train_dataloader, device, scaler, ema)          

        try: # don't want this to crash if I accidentally delete the schedular config file
            schedular = update_schedular(args, optim, schedular) # allows changing of min n max learning rate during training
        except Exception as e:
            if os.path.exists(args.schedular_config_file) == False:
                print('Schedular config file not found, creating new one')
                save_schedular_data(args)
            else:
                print(e, '\n') # todo move all this to update_schedular it looks ugly :P

        with ema.average_parameters(): # evaluate using ema of the parameters
            vloss = validate_one_epoch(epoch, model, dev_dataloader, device, sanity_check=False)


        print(f'\n\n\n--- Epoch {epoch}, Validation Loss: {vloss} ---\n\n\n')
        write_to_log(args.log_file, f'{epoch} - {vloss}')
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
    parser.add_argument('--load_pretrained', action='store_true')
    parser.add_argument('--pretrained', type=str, default='stt_en_conformer_ctc_small')
    parser.add_argument('--run_test', action='store_true')
  
    parser.add_argument('--tokenizer', type=str, default='./tokenizer_spe_bpe_v128', help='path to tokenizer dir')
    parser.add_argument('--epochs', type=int, default=500)

    parser.add_argument('--log_file', type=str, default='log.txt')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints')
    parser.add_argument('--checkpoint', type=str, default='')
    parser.add_argument('--nemo_checkpoint', action='store_true')
    parser.add_argument('--model_config', type=str, default='')

    parser.add_argument('--save_top_k', type=int, default=5)


    parser.add_argument('--min_lr', type=float, default=1e-5)
    parser.add_argument('--max_lr', type=float, default=8e-3)
    parser.add_argument('--step_size', type=int, default=400)

    parser.add_argument('--schedular_data', type=str, default='./schedular_data_ctc.json')
    parser.add_argument('--wandb', action='store_false')
    parser.add_argument('--wandb_id', type=str, default='', help='Provide this if you want to resume a previous run')
    parser.add_argument('--mixed_precision', action='store_true')
    parser.add_argument('--no_load_optim', action='store_true', help='if set, will not load optimizer state from checkpoint')

    parser.add_argument('--clip_gradients', action='store_true')
    parser.add_argument('--clip_gradients_value', type=float, default=10.0)

    parser.add_argument('--micro_batch_duration', type=int, default=45, help='batch size for non-i.i.d micro batches')
    parser.add_argument('--micro_batch_number', type=int, default=1, help='number of i.i.d micro batches per mini-batch')

    parser.add_argument('--optimizer_type', type=str, default='madgrad', help='type of optimizer to use')
   
    parser.add_argument('--split_speakers', action='store_true', help='if set, will split speakers into different micro batches')
  
    parser.add_argument('--num_workers', type=int, default=1, help='number of workers for dataloader')


    args = parser.parse_args()


    if args.checkpoint != '':
        args.checkpoint = os.path.join(args.checkpoint_dir, args.checkpoint)

    if os.path.exists(args.checkpoint_dir) == False:
        os.mkdir(args.checkpoint_dir)

    if os.path.exists(args.log_file) == True:
        write_to_log(args.log_file, f'\n{"-"*50}\n---- New run ----\n{"-"*50}\n')

    if os.path.exists(args.schedular_data) == True:
        print(f'A schedular data with the name {args.schedular_data} already exists, please delete it if you want to start a new run')
        exit()

    if args.load_pretrained == True:
        if args.pretrained == '':
            raise ValueError('Please provide a pretrained model')
    elif args.model_config == '':
        raise ValueError('Please provide a model config, or load a pretrained model')


    main(args)
