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
from speachy.asr.decoding.ngram import decode_beams_lm
import non_iid_dataloader

import pickle as pkl

from tools import isfalse, istrue, exists, save_json


def kenlm_decoder(arpa_, vocab, alpha=0.5, beta=0.8):  
    arpa = arpa_ if arpa_ != '' else None
    alpha = alpha if arpa_ != '' else None
    beta = beta if arpa_ != '' else None
    decoder = build_ctcdecoder(vocab, kenlm_model_path=arpa, alpha=alpha, beta=beta)
    print(f'Loaded KenLM model from {arpa} with alpha={alpha} and beta={beta}')
    return decoder


@torch.no_grad()
def evaluate(args, model, corpus, decoder):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    hyp_data = []

    dataloader = non_iid_dataloader.get_eval_dataloader(
        corpus, 
        max_duration=args.max_duration, 
        return_speaker=True, 
        batch_size=1, 
        concat_samples=args.concat_samples,
        split_speakers=args.split_speakers,
        gap=args.gap,
        speaker_gap=args.speaker_gap,
        single_speaker_with_gaps=args.single_speaker_with_gaps,
        return_meta_data=True,
        max_allowed_utterance_gap=args.max_allowed_utterance_gap,
    )

    pbar = tqdm(dataloader, total=len(dataloader))
    for batch_num, batch in enumerate(pbar):

        audios = batch['audio'].reshape(1, -1).to(device)
      
        speaker_ids = ["_".join(el[0]) for el in batch['speakers']]
        
        #print(batch['audio_lens'])
        audio_lengths = batch['audio_lens'].reshape(1).to(device)
        targets = [el[0] for el in batch['text']]
        targets = [el.replace(" '", "'") for el in targets] # change this in training so that it's not needed here but i'll keep it for now
        meta_data = batch['metadata'][0][0]
        print(targets, meta_data)
        model_out = model.forward(
            input_signal=audios, 
            input_signal_length=audio_lengths,
            segment_lens=batch['segment_lens'] if isfalse(args.do_not_pass_segment_lens) else None
        ) 

        log_probs, _, encoded_len = model_out[:3]
        additional_outputs = model_out[-1]


        log_probs = log_probs.detach().cpu().numpy()

   
        decoded_beams = decode_beams_lm(
            logits_list = log_probs, 
            decoder = decoder, 
            beam_width = args.beam_size, 
            encoded_lengths = encoded_len,
            token_min_logp = args.token_min_logp,
            beam_prune_logp = args.beam_prune_logp,
        )

        outputs = {
            'beams': decoded_beams,
            'meta_data': meta_data,
            'speaker_ids': speaker_ids,
            'targets': targets,
            'batch_num': batch_num,
        }

        hyp_data.append(outputs)

    with open(args.save_outputs, 'wb') as f:
        pkl.dump(hyp_data, f)




def main(args):
    
    model = load_model(args) if args.self_conditioned == False else load_sc_model(args)
    if args.checkpoint != '':
        load_checkpoint(args, model)

    print('\nTrainable parameters:'+str(sum(p.numel() for p in model.parameters() if p.requires_grad)))
    print(f'Total parameters: {sum(p.numel() for p in model.parameters())}\n')

    if args.self_conditioned == True:
        print(f'Using self-conditioned CTC model\n')
        assert '_sc_' in args.model_config, 'Self-conditioned model must be used with self-conditioned model config'

    corpus_dict = tools.load_corpus()


    decoder = kenlm_decoder(args.language_model, model.tokenizer.vocab, alpha=args.alpha, beta=args.beta)
    decoder_beams = args.beam_size
    args.beam_size = decoder_beams #
    evaluate(args, model, corpus_dict[args.split], decoder)





if __name__ == '__main__':
    ''''
    Note I've only written this for a batch size of 1 (lazy)
    '''

    parser = argparse.ArgumentParser() 
    parser.add_argument('--load_pretrained', action='store_true')
    parser.add_argument('--pretrained', type=str, default='stt_en_conformer_ctc_small') # stt_en_conformer_ctc_large stt_en_conformer_transducer_large
    parser.add_argument('--model_config', type=str, default='../model_configs/conformer_sc_ctc_bpe_small.yaml') 

    parser.add_argument('--tokenizer', type=str, default='./tokenizer_spe_bpe_v128', help='path to tokenizer dir')
    parser.add_argument('--max_duration', type=float, default=0, help='max duration of audio in seconds')


    parser.add_argument('--log_file', type=str, default='eval_log.txt')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints')
    parser.add_argument('--checkpoint', type=str, default='checkpoint_68_id_15.pt')
    

    parser.add_argument('--beam_size', type=int, default=100)
    parser.add_argument('-lm', '--language_model', type=str, default='', help='arpa n-gram model for decoding')#./ngrams/3gram-6mix.arpa
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--beta', type=float, default=0.8)

    parser.add_argument('-token_skip', '--token_min_logp', default=-5, type=float)
    parser.add_argument('-beam_prune', '--beam_prune_logp', default=-10000, type=float)

    parser.add_argument('-nsc','--not_self_conditioned', action='store_true', help='use for non self-conditioned models')

    parser.add_argument('-mgap','--max_allowed_utterance_gap', type=float, default=3.0, help='max allowed gap between utterances in seconds')


    parser.add_argument('-gap','--gap', default=0.1, type=float, help='gap between utterances when concatenating')

    parser.add_argument('--single_speaker_with_gaps', action='store_true', help='if set, utterances will contain 1 speaker and additional gaps of speaker_gap will be added if there is a speaker change between two utternces of the same speaker')
    parser.add_argument('--speaker_gap', type=float, default=1.0, help='for use with single_speaker_with_gaps, will add this many seconds of silence between utterances of the same speaker when there is a speaker change in between them')

    parser.add_argument('--split_speakers', action='store_true', help='if set, wont concat samples from different speakers, (concat_samples must be enabled)')

    parser.add_argument('-psl','--pass_segment_lengths', action='store_true', help='if set, will pass segment lens to the model, used with concat_samples for multi segment models')
    


    parser.add_argument('-save','--save_outputs', default='', type=str, help='save outputs to file')
    args = parser.parse_args()


    args.do_not_pass_segment_lens = not args.pass_segment_lengths
    args.self_conditioned = not args.not_self_conditioned
    args.concat_samples = True
    args.config_from_checkpoint_dir = True
    
    if args.language_model == '':
        print('\n\nNO LM SPECIFIED!\n\n')

    if args.save_outputs == '':
        save_outputs = ''
        while save_outputs == '':
            save_outputs = input('Please provide a name for the output file: ').strip()
        args.save_outputs = save_outputs

    if args.checkpoint != '':
        args.checkpoint = os.path.join(args.checkpoint_dir, args.checkpoint)



    if args.config_from_checkpoint_dir == True:
        dir_contents = os.listdir(args.checkpoint_dir)
        config = [el for el in dir_contents if el.endswith('.yaml')]
        assert len(config) == 1, 'Exactly one config file must be in checkpoint dir'
        args.model_config = os.path.join(args.checkpoint_dir, config[0])
        print(f'Loading config from checkpoint dir: {args.model_config}')

    main(args)
