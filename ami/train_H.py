import argparse
from random import sample
import torch
from nemo.collections.asr.models.ctc_bpe_models import EncDecCTCModelBPE
import tools
import non_iid_dataloader
import os
from tqdm import tqdm
import numpy as np
import wandb
from torch.optim.lr_scheduler import CyclicLR
from model_utils import load_checkpoint, load_nemo_checkpoint, load_sc_model as load_model, write_to_log, \
    squeeze_batch_and_to_device, save_checkpoint, draw_text, load_schedular_data, save_schedular_data

from contextlib import nullcontext

import torch_optimizer
from torch_ema import ExponentialMovingAverage
# gradscaler
from torch.cuda.amp import GradScaler   

INTERPOLATE = 0.5


def exists(var):
    return var is not None

def isfalse(var):
    return var == False
def istrue(var):
    return var == True


def update_schedular(args, optim, schedular):
    if schedular is None:
        return None
    max_lr, min_lr, step_size = load_schedular_data(args)
    if max_lr != args.max_lr or min_lr != args.min_lr or step_size != args.step_size:
        print('Updating schedular')
        args.max_lr = max_lr
        args.min_lr = min_lr
        args.step_size = step_size
        schedular = CyclicLR(optim, base_lr=args.min_lr, max_lr=args.max_lr, step_size_up=args.step_size, step_size_down=args.step_size*2, mode='triangular', cycle_momentum=False)
    return schedular


def optimizer(model, args):
    implemented_optimizers = ['adamw','madgrad']
    if args.optimizer_type == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), betas=(0.9, 0.98), weight_decay=1e-6, lr=args.min_lr)
    elif args.optimizer_type == 'madgrad':
        optimizer = torch_optimizer.MADGRAD(model.parameters(), lr=args.min_lr, momentum=0.9, weight_decay=1e-6, eps=1e-6)
    else:
        raise ValueError(f'Unknown optimizer type: {args.optimizer_type}, implemented optimizers: {implemented_optimizers}')

    schedular = CyclicLR(optimizer, base_lr=args.min_lr, max_lr=args.max_lr, step_size_up=args.step_size, step_size_down=args.step_size*4, mode='triangular', cycle_momentum=False)

    return optimizer, schedular

@torch.no_grad()
def validate_one_epoch(model, val_dataloader, device, sanity_check=False):
    model.eval()
    pbar = tqdm(val_dataloader, total=len(val_dataloader))
    wers = []
    losses = []
    print('Evaluation epoch')
    for batch in pbar:
        if batch['tokens'].shape[-1] == 0: # edge case
            continue
        input_signal, input_signal_lengths, targets, target_lengths, batch_size = squeeze_batch_and_to_device(batch, device)
        segment_lens = batch['segment_lens'].to(device)
        #pbar.update(batch_size)
       
        model_out = model.forward(input_signal=input_signal, input_signal_length=input_signal_lengths, segment_lens=segment_lens if isfalse(args.do_not_pass_segment_lens) else None)
        log_probs, interim_posteriors, encoded_len = model_out[0], model_out[1], model_out[2] #just validate with final layer
        
        if exists(interim_posteriors):
            interims = torch.empty(interim_posteriors.shape[0]).to(device)
            for ix, layer in enumerate(interim_posteriors):
                interim = model.loss(log_probs=layer, targets=targets, input_lengths=encoded_len, target_lengths=target_lengths)
                interims[ix] = interim
            interim_loss = torch.mean(interims)
        loss = model.loss(log_probs=log_probs, targets=targets, input_lengths=encoded_len, target_lengths=target_lengths)
        loss = loss if not exists(interim_posteriors) else interim_loss*INTERPOLATE + loss*(1-INTERPOLATE)
        losses.append(loss.item())

        if sanity_check:
            return True

    return  sum(losses) / len(losses)

@torch.no_grad()
def return_psuedo_targets(model, ema, model_inputs):
    assert exists(ema), 'EMA is needed for psuedo target self-training'
    initial_model_state = 'eval'
    if model.training:
        initial_model_state = 'train'
        model.eval()
    
    with ema.average_parameters():
        model_out = model.forward(**model_inputs)
        log_probs, iterim_posteriors = model_out[0], model_out[1]

    if initial_model_state == 'train':
        model.train()

    return log_probs, iterim_posteriors

def self_distillation_kl(log_probs, iterim_posteriors, log_probs_targets, iterim_posteriors_targets):
    b, n, c = log_probs_targets.shape
    # now calculate kl divergence
    loss_lp = torch.nn.functional.kl_div(log_probs, log_probs_targets, reduction='batchmean', log_target=True)
    loss_ip = torch.nn.functional.kl_div(iterim_posteriors, iterim_posteriors_targets, reduction='batchmean', log_target=True)
   
    return loss_lp + loss_ip


def train_one_epoch(model, optim, schedular, train_dataloader, device, scaler=None, ema=None):
    model.train()

    losses = [] # for storing effective losses
    loss_iterim = [] # for storing loss of each accumulation step
    print('Training epoch')
    pbar = tqdm(train_dataloader, total=len(train_dataloader))
    autocast_device = 'cuda' if torch.cuda.is_available() else 'cpu' # for autocast if using mixed precision

    #torch.autograd.set_detect_anomaly(True)

    for i, batch in enumerate(pbar):
        if batch['tokens'].shape[-1] == 0: # edge case
            continue
        input_signal, input_signal_lengths, targets, target_lengths, batch_size = squeeze_batch_and_to_device(batch, device)
        segment_lens = batch['segment_lens'].to(device)

        with torch.autocast(device_type=autocast_device) if exists(scaler) else nullcontext(): # for mixed precision
            model_inputs = {
                'input_signal': input_signal,
                'input_signal_length': input_signal_lengths,
                'segment_lens': segment_lens if isfalse(args.do_not_pass_segment_lens) else None
            }
            model_out = model.forward(**model_inputs)
            log_probs, interim_posteriors, encoded_len = model_out[0], model_out[1], model_out[2] 

            if exists(interim_posteriors):
                interims = torch.empty(interim_posteriors.shape[0]).to(device)
                for ix, layer in enumerate(interim_posteriors):
                    interim = model.loss(log_probs=layer, targets=targets, input_lengths=encoded_len, target_lengths=target_lengths)
                    interims[ix] = interim
                interim_loss = torch.mean(interims)

            loss_final = model.loss(log_probs=log_probs, targets=targets, input_lengths=encoded_len, target_lengths=target_lengths)
            
            loss_a = loss_final if not exists(interim_posteriors) else INTERPOLATE * interim_loss + (1 - INTERPOLATE) * loss_final

            if args.self_distillation: # might not work just me messing around
                log_probs_targets, interim_posteriors_targets = return_psuedo_targets(model, ema, model_inputs)
                loss_distillation = self_distillation_kl(log_probs, interim_posteriors, log_probs_targets, interim_posteriors_targets)
                if args.wandb:
                    wandb.log({'distillation_loss': loss_distillation.item()})
                loss_a += loss_distillation


            loss_a = loss_a / args.accumulate_gradients

            loss_iterim.append(loss_a.item())

        scaler.scale(loss_a).backward() if exists(scaler) else loss_a.backward()
        
        if (i + 1) % args.accumulate_gradients == 0:
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

            cur_loss = sum(loss_iterim)
            loss_iterim = []
            losses.append(cur_loss)

            cur_lr = schedular.get_last_lr()[0] if exists(schedular) else optim.param_groups[0]['lr']
          
            if args.wandb:
                wandb.log({'train_loss': cur_loss, 'lrate': cur_lr})
            pbar.set_description(f'Loss: {cur_loss}, lrate: {cur_lr}')


    loss_end = sum(losses) / len(losses)
    if args.wandb:
        wandb.log({'train_loss_end': loss_end})

    torch.cuda.empty_cache() # to avoid memory leaks

    return loss_end
    


def main(args):
    model = load_model(args)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
 
    ami_dict = tools.load_corpus()
    tokenizer = model.tokenizer
 
    get_dl = lambda split: non_iid_dataloader.get_data_loader(
        split=ami_dict[split], 
        tokenizer=tokenizer, 
        shuffle=True, 
        max_duration=args.micro_batch_duration, 
        num_workers=args.num_workers, 
        batch_size=args.micro_batch_number, 
        concat_samples=args.concat_samples,
        split_speakers=args.split_speakers,
        gap=args.gap,
        speaker_gap=args.speaker_gap,
        single_speaker_with_gaps=args.single_speaker_with_gaps,
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
        lossval = validate_one_epoch(model, test_dataloader, device, sanity_check=False)
        print(f' Test loss: {lossval}')
        write_to_log(args.log_file, f'Test loss: {lossval} checkpoint: {args.checkpoint}')
        return
    
    validate_one_epoch(model, dev_dataloader, device, sanity_check=True) # check no crash

    ema = ExponentialMovingAverage(model.parameters(), decay=0.9999) 
    scaler = GradScaler() if args.mixed_precision else None

    if args.wandb:
        wandb.init(project=args.wandb_project, config=args) if args.wandb_id == '' else wandb.init(project=args.wandb_project, id=args.wandb_id, resume="must", config=args)
        wandb.watch(model, log="all")
        wandb.config.update({'total_params': total_params})
        print(f'\nLoggging with Wandb id: {wandb.run.id}\n')

    results = {}
    saved_checkpoints = []
    for epoch_ in range(args.epochs): # todo move epoch loop to model utils as is consistent across model types
        epoch = epoch_ + epoch_prev
        #train_dataloader.sampler.set_epoch(epoch)

        loss = train_one_epoch(model, optim, schedular, train_dataloader, device, scaler, ema)          

        try: # don't want this to crash if I accidentally delete the schedular config file
            schedular = update_schedular(args, optim, schedular) # allows changing of min n max learning rate during training
        except Exception as e:
            if os.path.exists(args.schedular_config_file) == False:
                print('Schedular config file not found, creating new one')
                save_schedular_data(args)
            else:
                print(e, '\n') # todo move all this to update_schedular it looks ugly :P

        with ema.average_parameters(): # evaluate using ema of the parameters
            vloss = validate_one_epoch(model, dev_dataloader, device, sanity_check=False)

        if args.wandb:
            wandb.log({'val_loss': vloss, 'epoch': epoch})

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

    parser.add_argument('--accumulate_gradients', type=int, default=4)

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
    parser.add_argument('--concat_samples', action='store_true', help='if set, will concat cuts from same meeting instead of stacking them')
    parser.add_argument('--split_speakers', action='store_true', help='if set, will split speakers into different micro batches')
    
    parser.add_argument('-gap', '--gap', type=float, default=0.1, help='gap between samples when concat_samples is True')
    
    parser.add_argument('--single_speaker_with_gaps', action='store_true', help='if set, utterances will contain 1 speaker and additional gaps of speaker_gap will be added if there is a speaker change between two utternces of the same speaker')
    parser.add_argument('--speaker_gap', type=float, default=1.0, help='for use with single_speaker_with_gaps, will add this many seconds of silence between utterances of the same speaker when there is a speaker change in between them')

    parser.add_argument('--do_not_pass_segment_lens', action='store_true', help='if set, will not pass segment lens to the model')
    parser.add_argument('--num_workers', type=int, default=1, help='number of workers for dataloader')

    parser.add_argument('--self_distillation', action='store_true', help='if set, will use self distillation')

    parser.add_argument('-wp','--wandb_project', type=str, default='deliberation-Custom_ami', help='wandb project name')
  

    args = parser.parse_args()

    assert args.self_distillation == False or args.mixed_precision == False, 'Self distillation does not work with mixed precision (yet)'

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
