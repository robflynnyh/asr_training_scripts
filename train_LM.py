import argparse
import torch

import tools
import non_iid_dataloader
import os
from tqdm import tqdm
import wandb
from os.path import join

from speachy.utils.general import (
    load_config,
    load_checkpoint,
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
    mark_padding,
)

from speachy.utils.misc import add_common_args

from speachy.lm.tools.loading import autoload
from speachy.utils.general.training_loop import optimizer, update_schedular

from speachy.utils.helpers import  exists, isfalse, istrue
from contextlib import nullcontext

from torch_ema import ExponentialMovingAverage
# gradscaler
from torch.cuda.amp import GradScaler   


INTERPOLATE = 0.5


@torch.no_grad()
def validate_one_epoch(args, model, val_dataloader, device, sanity_check=False):
    model.eval()
    pbar = tqdm(val_dataloader, total=len(val_dataloader))
    wers = []
    losses = []
    eos_id = 0 if not args.no_eos else -100 # set eos_id to the ignore_index if no_eos is True
    print('Evaluation epoch')
    for batch in pbar:
        tokens, token_lens = batch_to_device(batch, device)
    
        token_lens += 1 # add 1 for bos
        tokens = add_bos(tokens, bos_token_id=0)
        targets = tokens.clone()
        targets[:, :-1] = tokens[:, 1:]
        targets = add_eos(targets, eos_id=eos_id, token_lens=token_lens)
        
        mask = token_lens_to_mask(token_lens)
        targets = mark_padding(targets, mask, pad_id=-100)

        logits, _, _ = model(x=tokens, length=token_lens, cache=None)

        loss = loss_ce(logits=logits, labels=targets, ignore_index=-100)
        losses.append(loss.item())

        if sanity_check:
            return True

    return  sum(losses) / len(losses)

def intermediate_loss(loss_fn, interim_logits, targets):
    if interim_logits == None:
        return None
    interims = torch.empty(interim_logits.shape[0], device=interim_logits.device)
    for i in range(interim_logits.shape[0]):
        interims[i] = loss_fn(interim_logits[i], targets)

    return torch.mean(interims)
    

def train_one_epoch(args, model, optim, schedular, train_dataloader, device, scaler=None, ema=None):
    model.train()

    losses = [] # for storing effective losses
    loss_iterim = [] # for storing loss of each accumulation step
    print('Training epoch')
    pbar = tqdm(train_dataloader, total=len(train_dataloader))
    autocast_device = 'cuda' if torch.cuda.is_available() else 'cpu' # for autocast if using mixed precision

    loss_fn = lambda logits, targets: loss_ce(
            logits=logits, 
            labels=targets, 
            ignore_index=-100,
            label_smoothing=args.label_smoothing
        )
    #torch.autograd.set_detect_anomaly(True)
    eos_id = 0 if not args.no_eos else -100 # set eos_id to the ignore_index if no_eos is True

    for i, batch in enumerate(pbar):
        tokens, token_lens = batch_to_device(batch, device)
      
        #print(token_lens.shape)
        token_lens += 1 # for <bos> and <eos>
        tokens = add_bos(tokens, bos_token_id=0)
        targets = tokens.clone()
        targets[:, :-1] = tokens[:, 1:] # shift right
        targets = add_eos(targets, eos_id=eos_id, token_lens=token_lens) # add <eos> to the end of each sequence

        with torch.autocast(device_type=autocast_device) if exists(scaler) else nullcontext(): # for mixed precision
            mask = token_lens_to_mask(token_lens) 
            targets = mark_padding(targets, mask, pad_id=-100) 

          
            logits, interim_posteriors, _ = model(x=tokens, length=token_lens, cache=None)
            interim_loss = intermediate_loss(loss_fn, interim_posteriors, targets)
            loss_a = loss_fn(logits=logits, targets=targets)
            
            loss_a = loss_a if not exists(interim_loss) else INTERPOLATE * interim_loss + (1 - INTERPOLATE) * loss_a
        
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
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
 
    config = load_config(args.model_config)
    assert 'dataset' in config, 'please add a dataset name to the config, and specify the path in the .env file'
    dataset = config['dataset']

    ami_dict = tools.load_corpus(
        target_folder = tools.request_env(dataset+'_PATH'),
        prefix_path = tools.request_env(dataset+'_BASE'),
        file_name = tools.request_env(dataset+'_NAME')
    ) 
    
    tokenizer_path = config['model']['tokenizer']['dir']
    tokenizer = tools.load_tokenizer(model_path=join(tokenizer_path, 'tokenizer.model'))
 
    get_dl = lambda split: non_iid_dataloader.get_data_loader(
        split=ami_dict[split], 
        tokenizer=tokenizer, 
        shuffle=True if split == 'train' else False,
        max_duration=args.micro_batch_duration, 
        num_workers=args.num_workers, 
        batch_size=args.micro_batch_number, 
        concat_samples=True,
        split_speakers=args.split_speakers,
        gap=0.0,
        speaker_gap=0.0,
        single_speaker_with_gaps=False,
        text_only=True, 
        pad_id=0,
        max_allowed_utterance_gap=args.max_allowed_utterance_gap,
    )
    

    train_dataloader, dev_dataloader = get_dl('train'), get_dl('dev')
    if args.run_test == True:
        test_dataloader = get_dl('test')

    args.max_tokens = -1
    model = autoload(config=config, tokenizer=tokenizer)
    model.tokenizer = tokenizer
    
    model.to(device)
 
    optim, schedular = optimizer(model, args)
    save_schedular_data(args)

    epoch_prev = 0
    if args.checkpoint != '':
        epoch_prev, _ = load_checkpoint(args, model, optim) 

    # print parameters
    total_params = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            total_params += param.numel()
    print(f'\nTotal number of parameters: {total_params/1e6}M\n')
           

    if args.run_test == True:
        lossval = validate_one_epoch(args, model, test_dataloader, device, sanity_check=False)
        print(f' Test loss: {lossval}')
        write_to_log(args.log_file, f'Test loss: {lossval} checkpoint: {args.checkpoint}')
        return
    
    validate_one_epoch(args, model, dev_dataloader, device, sanity_check=True) # check no crash

    ema = ExponentialMovingAverage(model.parameters(), decay=args.ema_decay) if args.ema_decay != 1.0 else None
    scaler = GradScaler() if args.mixed_precision else None

    if args.wandb:
        wandb.init(project=args.project_name, config=args) if args.wandb_id == '' else wandb.init(project=args.project_name, id=args.wandb_id, resume="must", config=args)
        wandb.watch(model, log="all")
        wandb.config.update({'total_params': total_params})
        print(f'\nLoggging with Wandb id: {wandb.run.id}\n')

    results = {}
    saved_checkpoints = []
    for epoch_ in range(args.epochs): # todo move epoch loop to model utils as is consistent across model types
        epoch = epoch_ + epoch_prev
        #train_dataloader.sampler.set_epoch(epoch)

        loss = train_one_epoch(args, model, optim, schedular, train_dataloader, device, scaler, ema)          

        try: # don't want this to crash if I accidentally delete the schedular config file
            schedular = update_schedular(args, optim, schedular) # allows changing of min n max learning rate during training
        except Exception as e:
            if os.path.exists(args.schedular_data) == False:
                print('Schedular config file not found, creating new one')
                save_schedular_data(args)
            else:
                print(e, '\n') # todo move all this to update_schedular it looks ugly :P

        with ema.average_parameters() if exists(ema) else nullcontext():
            vloss = validate_one_epoch(args, model, dev_dataloader, device, sanity_check=False)

        if args.wandb:
            wandb.log({'val_loss': vloss, 'epoch': epoch})

        print(f'\n\n\n--- Epoch {epoch}, Validation Loss: {vloss} ---\n\n\n')
        write_to_log(args.log_file, f'{epoch} - {vloss}')
        results[epoch] = {'loss': loss, 'vloss': vloss}
    
        if len(saved_checkpoints) < args.save_top_k:
            ema.store() if exists(ema) else None
            ema.copy_to() if exists(ema) else None # save ema of the parameters
            path = save_checkpoint(args, model, optim, epoch, vloss)
            ema.restore() if exists(ema) else None
            draw_text('High Score')
            saved_checkpoints.append({
                'path': path,
                'epoch': epoch,
                'vloss': vloss
            })
        elif vloss < max(saved_checkpoints, key=lambda x: x['vloss'])['vloss']:
            ema.store() if exists(ema) else None
            ema.copy_to() if exists(ema) else None
            path = save_checkpoint(args, model, optim, epoch, vloss)
            ema.restore() if exists(ema) else None
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
    parser.add_argument('--run_test', action='store_true')

    parser.add_argument('--log_file', type=str, default='log.txt')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints')
    parser.add_argument('--checkpoint', type=str, default='')
    parser.add_argument('--model_config', type=str, default='')

    parser.add_argument('-ema','--ema_decay', type=float, default=0.9999, help='decay rate for exponential moving average of parameters')

    parser.add_argument('--schedular_data', type=str, default='./schedular_data_lm.json')

    parser.add_argument('--micro_batch_duration', type=int, default=45, help='batch size for non-i.i.d micro batches')
    parser.add_argument('--micro_batch_number', type=int, default=1, help='number of i.i.d micro batches per mini-batch')

    parser.add_argument('-mgap','--max_allowed_utterance_gap', type=float, default=3.0, help='max allowed gap between utterances in seconds')
    
    parser.add_argument('--single_speaker_with_gaps', action='store_true', help='if set, utterances will contain 1 speaker and additional gaps of speaker_gap will be added if there is a speaker change between two utternces of the same speaker')
  
    parser.add_argument('--split_speakers', action='store_true', help='if set, will split speaker into multiple utterances')
    parser.add_argument('--label_smoothing', type=float, default=0.0)

    parser.add_argument('--project_name', default='deliberation-LM', type=str)
    parser.add_argument('-no_eos','--no_eos', action='store_true', help='if set, will not add eos token to the end of the utterances https://aclanthology.org/2020.blackboxnlp-1.26.pdf')

    parser.add_argument('-weight_decay','--weight_decay', type=float, default=1e-6, help='weight decay for optimizer')
    parser = add_common_args(parser)

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

    if args.model_config == '':
        raise ValueError('Please provide a model config, or load a pretrained model')


    main(args)
