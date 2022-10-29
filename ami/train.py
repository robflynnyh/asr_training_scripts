import argparse
import torch
from nemo.collections.asr.models.ctc_bpe_models import EncDecCTCModelBPE
import tools
import os
from tqdm import tqdm
import numpy as np
import wandb
from torch.optim.lr_scheduler import CyclicLR
from model_utils import load_checkpoint, load_nemo_checkpoint, load_model, write_to_log, \
    squeeze_batch_and_to_device, save_checkpoint, draw_text, load_schedular_data, save_schedular_data

from torch_ema import ExponentialMovingAverage


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
    optimizer = torch.optim.AdamW(model.parameters(), betas=(0.9, 0.98), weight_decay=1e-6, lr=args.min_lr)
    schedular = CyclicLR(optimizer, base_lr=args.min_lr, max_lr=args.max_lr, step_size_up=args.step_size, step_size_down=args.step_size*4, mode='triangular', cycle_momentum=False)

    return optimizer, schedular


def validate_one_epoch(model, val_dataloader, device, sanity_check=False):
    model.eval()
    pbar = tqdm(val_dataloader, total=len(val_dataloader.sampler.data_source))
    wers = []
    losses = []
    print('Evaluation epoch')
    for batch in pbar:
        input_signal, input_signal_lengths, targets, target_lengths, batch_size = squeeze_batch_and_to_device(batch, device)
        pbar.update(batch_size)
        with torch.no_grad():
            log_probs, encoded_len, greedy_predictions = model.forward(input_signal=input_signal, input_signal_length=input_signal_lengths)
            loss = model.loss(log_probs=log_probs, targets=targets, input_lengths=encoded_len, target_lengths=target_lengths)
            losses.append(loss.item())

        if sanity_check:
            return True

    return  sum(losses) / len(losses)

def train_one_epoch(model, optim, schedular, train_dataloader, device, ema=None):
    model.train()

    losses = [] # for storing effective losses
    loss_iterim = [] # for storing loss of each accumulation step
    print('Training epoch')
    pbar = tqdm(train_dataloader, total=len(train_dataloader.sampler.data_source))
    for i, batch in enumerate(pbar):
        input_signal, input_signal_lengths, targets, target_lengths, batch_size = squeeze_batch_and_to_device(batch, device)
        pbar.update(batch_size)
       
        log_probs, encoded_len, greedy_predictions = model.forward(input_signal=input_signal, input_signal_length=input_signal_lengths)
        loss_a = model.loss(log_probs=log_probs, targets=targets, input_lengths=encoded_len, target_lengths=target_lengths)
        loss_a = loss_a / args.accumulate_gradients
        loss_iterim.append(loss_a.item())
        loss_a.backward()
        
        if (i + 1) % args.accumulate_gradients == 0: 
            optim.step()
            optim.zero_grad()

            if schedular is not None:
                schedular.step()
            if ema is not None:
                ema.update() # update the moving average of the parameters

            cur_loss = sum(loss_iterim)
            loss_iterim = []
            losses.append(cur_loss)
            if schedular is not None:
                cur_lr = schedular.get_last_lr()[0]
            else:
                cur_lr = optim.param_groups[0]['lr']
            if args.wandb:
                wandb.log({'train_loss': cur_loss, 'lrate': cur_lr})
            pbar.set_description(f'Loss: {cur_loss}, lrate: {cur_lr}')
        
    loss_end = sum(losses) / len(losses)
    if args.wandb:
        wandb.log({'train_loss_end': loss_end})

    return loss_end
    


def main(args):
    model = load_model(args)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
 
    ami_dict = tools.load_corpus()
    tokenizer = model.tokenizer
 
    train_dataloader = tools.load_dataloader(ami_dict['train'], tokenizer, args.batch_length, True) # add to args as args.batch_duration
    dev_dataloader = tools.load_dataloader(ami_dict['dev'], tokenizer, args.batch_length, False)

    if args.run_test == True:
        test_dataloader = tools.load_dataloader(ami_dict['test'], tokenizer, 350, False)
 
    optim, schedular = optimizer(model, args)
    save_schedular_data(args)

    epoch_prev = 0
    if args.checkpoint != '':
        epoch_prev, _ = load_checkpoint(args, model, optim) if args.nemo_checkpoint == False else load_nemo_checkpoint(args, model, optim)
       

    if args.run_test == True:
        lossval = validate_one_epoch(model, test_dataloader, device, sanity_check=False)
        print(f' Test loss: {lossval}')
        write_to_log(args.log_file, f'Test loss: {lossval} checkpoint: {args.checkpoint}')
        return
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print(f'Total parameters: {total_params} M')

    validate_one_epoch(model, dev_dataloader, device, sanity_check=True) # check no crash

    ema = ExponentialMovingAverage(model.parameters(), decay=0.9999) 


    if args.wandb:
        wandb.init(project="deliberation-Custom_ami") if args.wandb_id == '' else wandb.init(project="deliberation-Custom_ami", id=args.wandb_id, resume="must")
        wandb.config.parameters = total_params
        wandb.config.update(args)

    results = {}
    saved_checkpoints = []
    for epoch_ in range(args.epochs): # todo move epoch loop to model utils as is consistent across model types
        epoch = epoch_ + epoch_prev
        train_dataloader.sampler.set_epoch(epoch)

        
        loss = train_one_epoch(model, optim, schedular, train_dataloader, device, ema)          

        try: # don't want this to crash if I accidentally delete the schedular config file
            schedular = update_schedular(args, optim, schedular) # allows changing of min n max learning rate during training
        except Exception as e:
            if 'FileNotFoundError' in str(e):
                print('Schedular config file not found, creating new one')
                save_schedular_data(args)
            else:
                print(e, '\n') # todo move all this to update_schedular it looks ugly :P

        with ema.average_parameters(): # evaluate using ema of the parameters
            vloss = validate_one_epoch(model, dev_dataloader, device, sanity_check=False)

        if args.wandb:
            wandb.log({'val_loss': vloss})

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

    parser.add_argument('--batch_length', type=int, default=350)

    parser.add_argument('--accumulate_gradients', type=int, default=16)
    #parser.add_argument('--lr', type=float, default=2e-3)
    parser.add_argument('--min_lr', type=float, default=1e-5)
    parser.add_argument('--max_lr', type=float, default=8e-3)
    parser.add_argument('--step_size', type=int, default=400)
    parser.add_argument('--schedular_data', type=str, default='./schedular_data_ctc.json')
    parser.add_argument('--wandb', action='store_false')
    parser.add_argument('--wandb_id', type=str, default='', help='Provide this if you want to resume a previous run')
   
    args = parser.parse_args()

    if args.checkpoint != '':
        args.checkpoint = os.path.join(args.checkpoint_dir, args.checkpoint)

    if os.path.exists(args.checkpoint_dir) == False:
        os.mkdir(args.checkpoint_dir)

    if os.path.exists(args.log_file) == True:
        write_to_log(args.log_file, f'\n{"-"*50}\n---- New run ----\n{"-"*50}\n')

    if args.load_pretrained == True:
        if args.pretrained == '':
            raise ValueError('Please provide a pretrained model')
    elif args.model_config == '':
        raise ValueError('Please provide a model config, or load a pretrained model')


    main(args)
