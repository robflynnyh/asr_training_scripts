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
from model_utils import load_checkpoint, load_nemo_checkpoint, load_model, load_sc_model, write_to_log, decode_lm
import non_iid_dataloader

import pickle as pkl

from tools import isfalse, istrue, exists, save_json


def kenlm_decoder(args, arpa_, vocab, alpha=0.6, beta=0.8):  
    arpa = arpa_ if arpa_ != '' else None
    alpha = alpha if arpa_ != '' else None
    beta = beta if arpa_ != '' else None
    decoder = build_ctcdecoder(
        vocab, 
        kenlm_model_path=arpa, 
        alpha=alpha, 
        beta=beta,
        lm_score_boundary=not args.lm_dont_score_boundary,
        unk_score_offset=args.lm_unk_score_offset,
    )
    print(f'Loaded KenLM model from {arpa} with alpha={alpha} and beta={beta}')
    return decoder

def save_attention_information(args, batch_num, additional_outputs, speaker_ids, targets):
    if args.save_attention == False:
        return
    # move all tensors to cpu
    additional_outputs = {k: v.cpu().numpy() for k, v in additional_outputs.items() if v.__class__.__name__ == 'Tensor'}
    data = {**additional_outputs, 'speaker_ids': speaker_ids, 'targets': targets}
    torch.save(data, os.path.join(args.save_attn_dir, f'attns_{batch_num}.pt'))

def noise_audio(audio, noise_level):
    if noise_audio == 0.0:
        return audio
    noise = torch.randn_like(audio)
    noise = noise * noise_level
    audio = audio + noise
    return audio


def enable_dropout(model, dropout_rate=0.0):
    if dropout_rate == 0.0:
        return model
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()
            m.p = dropout_rate
            m.inplace = True
            print(f'Enabled dropout with rate {dropout_rate} in {m.__class__.__name__}')
    return model


def evaluate(args, model, corpus, decoder):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    model = enable_dropout(model, args.dropout_rate)

    hyps = []
    refs = []
    speakers = []
    encoded_lens = []
    #dataloader = tools.eval_dataloader(corpus, args.batch_size)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

    dataloader = non_iid_dataloader.get_eval_dataloader(
        corpus.filter(lambda x: x.supervisions[0].recording_id == "AimeeMullins_2009P"), 
        max_duration=args.max_duration, 
        return_speaker=True, 
        batch_size=args.num_meetings, 
        concat_samples=args.concat_samples,
        split_speakers=args.split_speakers,
        gap=args.gap,
        speaker_gap=args.speaker_gap,
        single_speaker_with_gaps=args.single_speaker_with_gaps,
        max_allowed_utterance_gap=args.max_allowed_utterance_gap,
        return_meta_data=True,
        shuffle = True
    )

    epochs = 2

    for epoch in range(epochs):
        pbar = tqdm(dataloader, total=len(dataloader))
        for batch_num, batch in enumerate(pbar):
        
            audios = batch['audio'].reshape(-1, batch['audio'].shape[-1]).to(device)
        
            speaker_ids = ["_".join(el[0]) for el in batch['speakers']]

            audio_lengths = batch['audio_lens'].reshape(-1).to(device)
            targets = [el[0] for el in batch['text']]
            targets = [el.replace(" '", "'") for el in targets] # change this in training so that it's not needed here

            audios = noise_audio(audios, args.noise_level)

            r_ids = [el[0]['recording_id'] for el in batch['metadata']]

            processed_signal, processed_signal_length = model.preprocessor(
                input_signal=audios, 
                length=audio_lengths
            )
            processed_signal = processed_signal.repeat(3, 1, 1)
            processed_signal_length = processed_signal_length.repeat(3)
            
            processed_signal[:2] = model.spec_augmentation(input_spec=processed_signal[:2], length=processed_signal_length[:2])

            model_out = model.forward(
                processed_signal=processed_signal, 
                processed_signal_length=processed_signal_length,
            ) 

            log_probs, _, encoded_len = model_out[:3]

            aug_log_probs = log_probs[:2]
            target_log_probs = log_probs[-1, None]
            print(aug_log_probs.shape, target_log_probs.shape)

    
            decoded = decode_lm(target_log_probs.detach().cpu().numpy(), decoder, beam_width=args.beam_size, encoded_lengths=encoded_len)
            decoded = [el.replace(" '", "'") for el in decoded] # change this in training so that it's not needed here

            pseudo_targets = torch.LongTensor(model.tokenizer.text_to_ids(decoded[0])).unsqueeze(0).to(device).repeat(2, 1)

            print(encoded_len.shape, pseudo_targets.shape)
        
            loss =  model.loss(
                log_probs=aug_log_probs, 
                targets=pseudo_targets, 
                input_lengths=encoded_len[:2],
                target_lengths=torch.LongTensor([len(pseudo_targets[0]), len(pseudo_targets[1])]).to(device)
            )
            
            # loss.backward()
            # optimizer.step()
            # optimizer.zero_grad()

            print(f'Decoded: {" - ".join([el for el in decoded])}\n')
            print(f'Targets: {" - ".join([el for el in targets])}')

            if epoch == epochs - 1:
                hyps.extend(decoded)
                refs.extend(targets)
                speakers.extend(speaker_ids)
                encoded_lens.extend(encoded_len.cpu().tolist())

## CUDA_VISIBLE_DEVICES='1' python eval_ctc_dynamic.py --checkpoint_dir ../checkpoints_done/ASR/TEDLIUM/checkpoints_cosine_noths/ --checkpoint 'checkpoint_359_id_52.pt' --max_duration 1


    if args.sclite:
        refname, hypname = tools.write_trn_files(refs=refs, hyps=hyps, speakers=speakers, encoded_lens=encoded_lens, out_dir=args.sclite_dir)
        sclite_wer = tools.eval_with_sclite(refname, hypname, mode='dtl all')
        print(f'WER (sclite): {sclite_wer}')

    nemo_wer = word_error_rate(hyps, refs, use_cer=args.cer)
    print(f'WER (nemo): {nemo_wer}')
    
    if args.sweep == True:
        wandb.log({"WER": nemo_wer})


    if args.save_outputs.strip() != '':
            save_json({'hyps': hyps, 'refs': refs, 'speakers': speakers, 'encoded_lens': encoded_lens}, args.save_outputs)

    return nemo_wer


def main(args):
    print(args.self_conditioned, 'DD')
    model = load_model(args) if args.self_conditioned == False else load_sc_model(args)
    if args.checkpoint != '':
        load_checkpoint(args, model)

    print('\nTrainable parameters:'+str(sum(p.numel() for p in model.parameters() if p.requires_grad)))
    print(f'Total parameters: {sum(p.numel() for p in model.parameters())}\n')

    if args.self_conditioned == True:
        print(f'Using self-conditioned CTC model\n')
        assert '_sc_' in args.model_config, 'Self-conditioned model must be used with self-conditioned model config'

    corpus_dict = tools.load_corpus()

    if args.sweep == True:
        wandb.init(config=args, project="ami-ngram-lm-sweep")
        decoder = kenlm_decoder(args, args.language_model, model.tokenizer.vocab, alpha=wandb.config['alpha'], beta=wandb.config['beta'])
    else:
        decoder = kenlm_decoder(args, args.language_model, model.tokenizer.vocab, alpha=args.alpha, beta=args.beta)
    decoder_beams = 1 if args.language_model == '' else args.beam_size
    args.beam_size = decoder_beams #
    evaluate(args, model, corpus_dict[args.split], decoder)



if __name__ == '__main__':
    parser = argparse.ArgumentParser() 
    parser.add_argument('--load_pretrained', action='store_true')
    parser.add_argument('--pretrained', type=str, default='stt_en_conformer_ctc_small') # stt_en_conformer_ctc_large stt_en_conformer_transducer_large
    parser.add_argument('--model_config', type=str, default='../model_configs/conformer_sc_ctc_bpe_small.yaml') 

    parser.add_argument('--tokenizer', type=str, default='./tokenizer_spe_bpe_v128', help='path to tokenizer dir')
    parser.add_argument('--max_duration', type=float, default=60, help='max duration of audio in seconds')
    parser.add_argument('--num_meetings', type=int, default=1, help='number of meetings per batch')

    parser.add_argument('--log_file', type=str, default='eval_log.txt')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints')
    parser.add_argument('--checkpoint', type=str, default='checkpoint_68_id_15.pt')
    parser.add_argument('--beam_size', type=int, default=100)
    parser.add_argument('-lm', '--language_model', type=str, default='', help='arpa n-gram model for decoding')#./ngrams/3gram-6mix.arpa
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--beta', type=float, default=0.8)
    parser.add_argument('--sweep', action='store_true', help='run wandb search for language model weight')
    parser.add_argument('-nsc','--not_self_conditioned', action='store_true', help='use for non self-conditioned models')
    parser.add_argument('--config_from_checkpoint_dir', action='store_false', help='load config from checkpoint dir') ##chane
    parser.add_argument('-cer', '--cer', action='store_true', help='compute CER instead of WER')

    parser.add_argument('--lm_dont_score_boundary', action='store_true', help='dont score boundary tokens')
    parser.add_argument('--lm_unk_score_offset', type=float, default=-10, help='offset for unk score')

    parser.add_argument('--return_attention', action='store_true', help='return attention')
    parser.add_argument('--save_attention', action='store_true', help='save attention')
    parser.add_argument('--save_attn_dir', type=str, default='./attns')

    parser.add_argument('-gap','--gap', default=0.1, type=float, help='gap between utterances when concatenating')

    parser.add_argument('--single_speaker_with_gaps', action='store_true', help='if set, utterances will contain 1 speaker and additional gaps of speaker_gap will be added if there is a speaker change between two utternces of the same speaker')
    parser.add_argument('--speaker_gap', type=float, default=1.0, help='for use with single_speaker_with_gaps, will add this many seconds of silence between utterances of the same speaker when there is a speaker change in between them')

    parser.add_argument('-mgap','--max_allowed_utterance_gap', type=float, default=3.0, help='max allowed gap between utterances in seconds')

    parser.add_argument('--concat_samples', action='store_true', help='if set, will concat cuts from same meeting instead of stacking them')
    parser.add_argument('--split_speakers', action='store_true', help='if set, wont concat samples from different speakers, (concat_samples must be enabled)')

    parser.add_argument('-psl','--pass_segment_lengths', action='store_true', help='if set, will pass segment lens to the model, used with concat_samples for multi segment models')
    
    parser.add_argument('-sclite','--sclite', action='store_true', help='if set, will eval with sclite')
    parser.add_argument('-sclite_dir', '--sclite_dir', type=str, default='./trns')
    parser.add_argument('-save','--save_outputs', default='', type=str, help='save outputs to file')
    parser.add_argument('-noise', '--noise_level', help='add noise to audio', default=0.0, type=float)
    parser.add_argument('-dropout', '--dropout_rate', help='dropout at inference', default=0.0, type=float)
    
    args = parser.parse_args()

    assert isfalse(args.split_speakers) or args.concat_samples, 'seperate_speakers can only be enabled if concat_samples is enabled'

    args.do_not_pass_segment_lens = not args.pass_segment_lengths
    args.self_conditioned = not args.not_self_conditioned

    if args.checkpoint != '':
        args.checkpoint = os.path.join(args.checkpoint_dir, args.checkpoint)

    if args.sclite == True:
        assert os.path.exists(args.sclite_dir), 'sclite dir does not exist'

    args.return_attention = True if args.save_attention else args.return_attention 

    if args.save_attention and not os.path.exists(args.save_attn_dir):
        os.makedirs(args.save_attn_dir)
        print(f'Created directory {args.save_attn_dir}, saving attention to this directory')

    if args.config_from_checkpoint_dir == True:
        dir_contents = os.listdir(args.checkpoint_dir)
        config = [el for el in dir_contents if el.endswith('.yaml')]
        assert len(config) == 1, 'Exactly one config file must be in checkpoint dir'
        args.model_config = os.path.join(args.checkpoint_dir, config[0])
        print(f'Loading config from checkpoint dir: {args.model_config}')

    main(args)
