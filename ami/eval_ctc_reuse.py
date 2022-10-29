import argparse
from typing import List
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
from model_utils import load_checkpoint, load_nemo_checkpoint, load_model, load_sc_model, write_to_log, decode_lm
import pickle as pkl



def kenlm_decoder(arpa_, vocab, alpha=0.6, beta=0.8):  
    arpa = arpa_ if arpa_ != '' else None
    alpha = alpha if arpa_ != '' else None
    beta = beta if arpa_ != '' else None
    decoder = build_ctcdecoder(vocab, kenlm_model_path=arpa, alpha=alpha, beta=beta)
    print(f'Loaded KenLM model from {arpa} with alpha={alpha} and beta={beta}')
    return decoder


def flatten_logits(logits_list: list):
    return [item for sublist in logits_list for item in sublist]


def save_logits(logits:List[np.ndarray], targets:List[str], encoded_len_list:List[np.ndarray], filename:str):
    with open(filename, 'wb') as f:
        pkl.dump({
            'logits': logits,
            'encoded_len_list': encoded_len_list,
            'targets': targets
        }, f)

def load_logits(filename):
    with open(filename, 'rb') as f:
        datum = pkl.load(f)
    return datum['logits'], datum['targets']

@torch.no_grad()
def evaluate(args, model, corpus, decoder):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    if args.load_logits_location == "":
        logit_list = []
        refs = []
        encoded_len_list = []
        dataloader = tools.eval_dataloader(corpus, args.batch_size, shuffle=args.shuffle)
        pbar = tqdm(dataloader, total=len(dataloader.sampler.data_source))
        for batch in pbar:
            audios = batch['audio'].reshape(-1, batch['audio'].shape[-1]).to(device)
            pbar.update(audios.shape[0])
            
            if args.early_exit == True and len(logit_list) > 10:
                break 

            audio_lengths = batch['audio_lens'].reshape(-1).to(device)
            targets = [el[0] for el in batch['text']]
            out = model.forward(input_signal=audios, input_signal_length=audio_lengths)
            log_probs = out[0]
            if args.ctx_model == True:
                encoded_len = out[2]
            else:
                encoded_len = out[1]
            #print(decode_lm(log_probs.cpu().numpy(), decoder, beam_width=100, encoded_lengths=encoded_len.cpu().numpy()))

            log_probs = log_probs.detach().cpu().numpy()
            encoded_len_list.extend(encoded_len.cpu().numpy())
            logit_list.extend(log_probs)
            refs.extend(targets)
        if args.save_logits_location != '':
            save_logits(logit_list, refs, encoded_len_list, args.save_logits_location)
    else:
        logit_list, refs = load_logits(args.load_logits_location)



    decoded = decode_lm(logit_list, decoder, beam_width=args.beam_size, encoded_lengths=encoded_len_list)
    
    wer = word_error_rate(decoded, refs)
    print(f'WER: {wer}')
    
    if args.sweep == True:
        wandb.log({"WER": wer})


    if args.save_text == True:
        rnum = np.random.randint(0, 100)
        path = f'./txt_outputs/{rnum}_decoded.txt'
        with open(f'{path}', 'w') as f:
            for i, dec in enumerate(decoded):
                f.write(f'{refs[i]}\n')
                f.write(f'{dec}\n')
                f.write('\n')

    return wer




def main(args):
    model = load_model(args) if args.self_conditioned == False else load_sc_model(args)
    if args.checkpoint != '':
        load_checkpoint(args, model)

    print('\nTrainable parameters:'+str(sum(p.numel() for p in model.parameters() if p.requires_grad)))
    print(f'Total parameters: {sum(p.numel() for p in model.parameters())}\n')

    if args.self_conditioned == True:
        print(f'Using self-conditioned CTC model\n')
        assert '_sc_' in args.model_config, 'Self-conditioned model must be used with self-conditioned model config'

    ami_dict = tools.load_corpus()

    if args.sweep == True:
        wandb.init(config=args, project="ami-ngram-lm-sweep")
        decoder = kenlm_decoder(args.language_model, model.tokenizer.vocab, alpha=wandb.config['alpha'], beta=wandb.config['beta'])
    else:
        decoder = kenlm_decoder(args.language_model, model.tokenizer.vocab, alpha=args.alpha, beta=args.beta)
    decoder_beams = 1 if args.language_model == '' else args.beam_size
    args.beam_size = decoder_beams #

    print(f'Split: {args.split}')
    evaluate(args, model, ami_dict[args.split], decoder)



if __name__ == '__main__':
    parser = argparse.ArgumentParser() 
    parser.add_argument('--load_pretrained', action='store_true')
    parser.add_argument('--pretrained', type=str, default='stt_en_conformer_ctc_small') # stt_en_conformer_ctc_large stt_en_conformer_transducer_large
    parser.add_argument('--model_config', type=str, default='/exp/exp1/acp21rjf/deliberation/Custom/model_configs/conformer_ctc_bpe_medium.yaml') 

    parser.add_argument('--tokenizer', type=str, default='./tokenizer_spe_bpe_v128', help='path to tokenizer dir')
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('-save_logits', '--save_logits_location', default='', help='path to save logits')   
    parser.add_argument('-load_logits', '--load_logits_location', default='', help='path to load logits')

    parser.add_argument('--log_file', type=str, default='eval_log.txt')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints')
    parser.add_argument('--checkpoint', type=str, default='')
    parser.add_argument('--beam_size', type=int, default=100)
    parser.add_argument('-lm', '--language_model', type=str, default='', help='arpa n-gram model for decoding')
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument('--alpha', type=float, default=0.6)
    parser.add_argument('--beta', type=float, default=1.0)
    parser.add_argument('--sweep', action='store_true', help='run wandb search for language model weight')
    parser.add_argument('-sc','--self_conditioned', action='store_true', help='use self-conditioned model')
    parser.add_argument('--beam_prune_logp', type=float, default=-10.0)
    parser.add_argument('--token_min_logp', type=float, default=-5.0)
    parser.add_argument('--save_text', action='store_true', help='save text output')

    parser.add_argument('--config_from_checkpoint_dir', action='store_true', help='load config from checkpoint dir')
    parser.add_argument('-ee', '--early_exit', action='store_true', help='early exit')

    parser.add_argument('--ctx_model', action='store_true', help='use context model')

    parser.add_argument('--shuffle', action='store_true', help='shuffle dataset')
   
    args = parser.parse_args()

    if args.checkpoint != '':
        args.checkpoint = os.path.join(args.checkpoint_dir, args.checkpoint)

    if args.config_from_checkpoint_dir == True:
        dir_contents = os.listdir(args.checkpoint_dir)
        config = [el for el in dir_contents if el.endswith('.yaml')]
        assert len(config) == 1, 'Exactly one config file must be in checkpoint dir'
        args.model_config = os.path.join(args.checkpoint_dir, config[0])
        print(f'Loading config from checkpoint dir: {args.model_config}')


    if args.load_logits_location != '' and os.path.exists(args.load_logits_location) == False:
        raise ValueError(f'{args.load_logits_location} does not exist')

    if args.save_text == True and os.path.exists('./txt_outputs') == False:
        os.mkdir('./txt_outputs')
  
    main(args)
