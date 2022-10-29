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
from model_utils import load_checkpoint, load_nemo_checkpoint, load_model, load_sc_model, write_to_log



def kenlm_decoder(arpa_, vocab, alpha=0.6, beta=0.8):  
    arpa = arpa_ if arpa_ != '' else None
    alpha = alpha if arpa_ != '' else None
    beta = beta if arpa_ != '' else None
    decoder = build_ctcdecoder(vocab, kenlm_model_path=arpa, alpha=alpha, beta=beta)
    print(f'Loaded KenLM model from {arpa} with alpha={alpha} and beta={beta}')
    return decoder

def decode_lm(logits_list, decoder, beam_width=100):
    decoded = []
    for logits in logits_list:
        decode = decoder.decode(logits=logits, beam_width=beam_width)
        decoded.append(decode)
    return decoded

@torch.no_grad()
def evaluate(args, model, corpus, decoder):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    hyps = []
    refs = []
    dataloader = tools.eval_dataloader(corpus, args.batch_size)
    pbar = tqdm(dataloader, total=len(dataloader.sampler.data_source))
    for batch in pbar:
        audios = batch['audio'].reshape(-1, batch['audio'].shape[-1]).to(device)
        pbar.update(audios.shape[0])

        audio_lengths = batch['audio_lens'].reshape(-1).to(device)
        targets = [el[0] for el in batch['text']]
        out = model.forward(input_signal=audios, input_signal_length=audio_lengths)
        log_probs = out[0]
        log_probs = log_probs.detach().cpu().numpy()
        decoded = decode_lm(log_probs, decoder, beam_width=args.beam_size)

        print(f'Decoded: {" - ".join([el for el in decoded])}\n')
        print(f'Targets: {" - ".join([el for el in targets])}\n')
        
        hyps.extend(decoded)
        refs.extend(targets)
    wer = word_error_rate(hyps, refs)
    print(f'WER: {wer}')
    
    if args.sweep == True:
        wandb.log({"WER": wer})

    return wer



def grid_search_lm_weight(args, model, corpus):
    alpha_weights = np.linspace(0.1, 0.9, 7)


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
    #decoder_beams = 1 if args.language_model == '' else args.beam_size
    #args.beam_size = decoder_beams #
    evaluate(args, model, ami_dict[args.split], decoder)



if __name__ == '__main__':
    parser = argparse.ArgumentParser() 
    parser.add_argument('--load_pretrained', action='store_true')
    parser.add_argument('--pretrained', type=str, default='stt_en_conformer_ctc_small') # stt_en_conformer_ctc_large stt_en_conformer_transducer_large
    parser.add_argument('--model_config', type=str, default='/exp/exp1/acp21rjf/deliberation/Custom/model_configs/conformer_ctc_bpe_medium.yaml') 

    parser.add_argument('--tokenizer', type=str, default='./tokenizer_spe_bpe_v128', help='path to tokenizer dir')
    parser.add_argument('--batch_size', type=int, default=10)

    parser.add_argument('--log_file', type=str, default='eval_log.txt')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints')
    parser.add_argument('--checkpoint', type=str, default='checkpoint_134_id_26.pt')
    parser.add_argument('--beam_size', type=int, default=100)
    parser.add_argument('-lm', '--language_model', type=str, default='', help='arpa n-gram model for decoding')
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument('--alpha', type=float, default=0.6)
    parser.add_argument('--beta', type=float, default=1.0)
    parser.add_argument('--sweep', action='store_true', help='run wandb search for language model weight')
    parser.add_argument('-sc','--self_conditioned', action='store_true', help='use self-conditioned model')
   
    args = parser.parse_args()

    if args.checkpoint != '':
        args.checkpoint = os.path.join(args.checkpoint_dir, args.checkpoint)

    
  
    main(args)
