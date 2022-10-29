import argparse
import torch
from nemo.collections.asr.models.ctc_bpe_models import EncDecCTCModelBPE
from nemo.collections.asr.models.rnnt_bpe_models import EncDecRNNTBPEModel
import tools
import os
from tqdm import tqdm
import numpy as np

import wandb
import kenlm
from pyctcdecode import build_ctcdecoder
import multiprocessing

from nemo.collections.asr.metrics.wer import word_error_rate
from omegaconf.omegaconf import OmegaConf

def load_checkpoint(args, model):
    checkpoint_path = args.checkpoint
    print(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    try:
        model.load_state_dict(checkpoint['state_dict'])
    except:
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        print('Warning: Strict loading failed, loading with non-strict loading')


def load_nemo_checkpoint(args, model, optim):
    checkpoint_path = args.checkpoint
    print(checkpoint_path)
    checkpoint = torch.load(checkpoint_path)
    print(checkpoint['state_dict'].keys())
    model.load_state_dict(checkpoint['state_dict'])
    optim.load_state_dict(checkpoint['optimizer_states'][0])
    epoch = checkpoint['epoch']
    wer = None
    val_loss = None
    print(f'Loaded checkpoint from {checkpoint_path}')
    print(f'Epoch: {epoch}, WER: {wer}, Validation loss: {val_loss}')
    return epoch, wer, val_loss


def load_model(args):
    print(args.pretrained)
    if args.load_pretrained == True:
        if 'ctc' in args.pretrained:
            model = EncDecCTCModelBPE.from_pretrained(args.pretrained)
        elif 'transducer' in args.pretrained:
            model = EncDecRNNTBPEModel.from_pretrained(args.pretrained)
            if args.greedyrnnt == False:
                decode_cfg = model.cfg.decoding
                decode_cfg['strategy'] = 'beam'
                decode_cfg['beam']['beam_size'] = 10
                decode_cfg['beam']['return_best_hypothesis'] = True
                model.change_decoding_strategy(decode_cfg)
        else:
            raise ValueError('Pretrained model not supported')
        if args.tokenizer != '':
            model.change_vocabulary(new_tokenizer_dir=args.tokenizer, new_tokenizer_type='bpe')
        return model
    else:
        cfg = OmegaConf.load(args.model_config)
        model = EncDecCTCModelBPE(cfg['model'])
        print(f'Loaded model from config file {args.model_config}')
        return model


def write_to_log(log_file, data):
    with open(log_file, 'a') as f:
        f.write(data)
        f.write('\n')


def kenlm_decoder(arpa_, vocab, alpha=0.5, beta=1.0):  
    arpa = arpa_ if arpa_ != '' else None
    alpha = alpha if arpa_ != '' else None
    beta = beta if arpa_ != '' else None
    decoder = build_ctcdecoder(vocab, kenlm_model_path=arpa, alpha=alpha, beta=beta)
    print(f'Loaded KenLM model from {arpa} with alpha={alpha} and beta={beta}')
    return decoder

def decode_lm(logits_list, decoder, beam_width=100):
    #with multiprocessing.get_context('fork').Pool() as pool:
    decoded = []
    for logits in logits_list:
        decode = decoder.decode(logits=logits, beam_width=beam_width)
        decoded.append(decode)
    #decoded = " ".join(decoded).strip()
    return decoded



def main(args):
    model = load_model(args)
    if args.checkpoint != '':
        load_checkpoint(args, model)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    print('\nTrainable parameters:'+str(sum(p.numel() for p in model.parameters() if p.requires_grad)))
    print(f'Total parameters: {sum(p.numel() for p in model.parameters())}\n')
 
    ami_dict = tools.load_corpus()
 
    tokenizer = model.tokenizer

    sources = []
    targets = []
    ids = []
    outputs = {}
    for entry in ami_dict[args.split]:
        targets.append(entry.supervisions[0].text)
        sources.append(entry.recording.sources[0].source)
        ids.append(entry.supervisions[0].recording_id)
        if entry.supervisions[0].recording_id not in outputs:
            outputs[entry.supervisions[0].recording_id] = {'decoded': [], 'target': []}

    decoder = kenlm_decoder(args.language_model, tokenizer.vocab)   
    decoder_beams = 1 if args.language_model == '' else 100
    
    
    batch_size = args.batch_size
    for i in tqdm(range(0, len(sources), batch_size)):
        batch_sources = sources[i:i+batch_size]
        batch_targets = targets[i:i+batch_size]
        batch_ids = ids[i:i+batch_size]
        if 'ctc' in args.pretrained:
            batch_logits = model.transcribe(batch_sources, logprobs=True, batch_size=batch_size)
            batch_decoded = decode_lm(batch_logits, decoder, beam_width=decoder_beams)
          

        elif 'transducer' in args.pretrained:
            batch_decoded = model.transcribe(batch_sources, batch_size=batch_size)[0]

        print(f'Decoded {i}/{len(sources)}')
        for j in range(len(batch_decoded)):
            outputs[batch_ids[j]]['decoded'].append(batch_decoded[j].strip())
            outputs[batch_ids[j]]['target'].append(batch_targets[j].strip())
            print(f'<--- {batch_decoded[j]} -> {batch_targets[j]} --->')
        print('\n')
    
   

    decoded_all = []
    target_all = []
    for id in outputs.keys():
        decoded = outputs[id]['decoded']
        target = outputs[id]['target']
        decoded_all.extend(decoded)
        target_all.extend(target)
        wer = word_error_rate(decoded, target)
        print(f'{id}: {wer}')

    wer_all = word_error_rate(decoded_all, target_all)
    print(f'All: {wer_all}')
        



        
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser() 
    parser.add_argument('--load_pretrained', action='store_false')
    parser.add_argument('--pretrained', type=str, default='stt_en_conformer_transducer_small') # stt_en_conformer_ctc_large stt_en_conformer_transducer_large
    parser.add_argument('--model_config', type=str, default='') 

    parser.add_argument('--tokenizer', type=str, default='./tokenizer_spe_bpe_v1000', help='path to tokenizer dir')
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--split', type=str, default='test')

    parser.add_argument('--log_file', type=str, default='eval_log.txt')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints')
    parser.add_argument('--checkpoint', type=str, default='checkpoint_92_id_49.pt')
    parser.add_argument('--beam_size', type=int, default=100)
    parser.add_argument('-greedy','--greedyrnnt', action='store_true')
    parser.add_argument('-lm', '--language_model', type=str, default='', help='n-gram model for decoding can be arpa or bin')
   
    args = parser.parse_args()

    if args.checkpoint != '':
        args.checkpoint = os.path.join(args.checkpoint_dir, args.checkpoint)

    if os.path.exists(args.checkpoint_dir) == False:
        os.mkdir(args.checkpoint_dir)
    if os.path.exists(args.log_file) == True:
        write_to_log(args.log_file, f'\n{"-"*50}\n---- New run ----\n{"-"*50}\n')

  
  

    main(args)
