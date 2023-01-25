import torch
from nemo.collections.asr.models.ctc_bpe_models import EncDecCTCModelBPE
from nemo.collections.asr.models.rnnt_bpe_models import EncDecRNNTBPEModel
from nemo.collections.asr.models.scctc_bpe_models import EncDecSCCTCModelBPE
import os
import numpy as np
from omegaconf.omegaconf import OmegaConf
import json

def load_config(config:str):
    return OmegaConf.load(config)

def write_to_log(log_file, data):
    with open(log_file, 'a') as f:
        f.write(data)
        f.write('\n')


def load_checkpoint(args, model, optim=None, force_cpu=False):
    checkpoint_path = args.checkpoint
    print(checkpoint_path)
    map_location = torch.device('cuda' if torch.cuda.is_available() and not force_cpu else 'cpu')
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    print(checkpoint['model_state_dict'].keys())
    try:
        model.load_state_dict(checkpoint['model_state_dict'])
    except Exception as e:
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        print(f'Error loading model state_dict: {e}, loading attempted with strict=False')
    if 'no_load_optim' in args.__dict__ and args.no_load_optim == True:
        print('Not loading optimizer')
    elif optim is not None:
        optim.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch'] if 'epoch' in checkpoint else 0
    val_loss = checkpoint['val_loss'] if 'val_loss' in checkpoint else None
    print(f'Loaded checkpoint from {checkpoint_path}')
    print(f'Epoch: {epoch}, Validation loss: {val_loss}')
    return epoch, val_loss

def load_nemo_checkpoint(args, model, optim):
    checkpoint_path = args.checkpoint
    print(checkpoint_path)
    checkpoint = torch.load(checkpoint_path)
    print(checkpoint['state_dict'].keys())
    model.load_state_dict(checkpoint['state_dict'])
    optim.load_state_dict(checkpoint['optimizer_states'][0])
    epoch = checkpoint['epoch']
    val_loss = None
    print(f'Loaded checkpoint from {checkpoint_path}')
    print(f'Epoch: {epoch}, Validation loss: {val_loss}')
    return epoch, val_loss


def save_checkpoint(args, model, optim, epoch, val_loss):
    path = os.path.join(args.checkpoint_dir, f'checkpoint_{epoch}_id_{np.random.randint(0,100)}.pt')
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optim.state_dict(),
        'val_loss': val_loss
    }, path)
    print(f'Saved checkpoint to {path}')
    return path

def load_schedular_data(args):
    with open(args.schedular_data, 'r') as f:
        data = json.load(f)
    return data['max_lr'], data['min_lr'], data['step_size']


def save_schedular_data(args):
    tosave = {
        'max_lr': args.max_lr,
        'min_lr': args.min_lr,
        'step_size': args.step_size
    }
    with open(args.schedular_data, 'w') as f:
        json.dump(tosave, f)

def load_tokenizer(model_path:str):
    tokenizer_spe = nemo_nlp.modules.get_tokenizer(tokenizer_name="sentencepiece", tokenizer_model=model_path)
    return tokenizer_spe


def draw_text(text):
    print(f'\n \n ------------------------------------------------- ')
    print(f' ----------------- {text} ----------------- ')
    print(f' ------------------------------------------------- \n \n')