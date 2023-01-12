import argparse
import torch
from nemo.collections.asr.models.ctc_bpe_models import EncDecCTCModelBPE
from nemo.collections.asr.models.rnnt_bpe_models import EncDecRNNTBPEModel
import tools
import os
from tqdm import tqdm
import numpy as np
import random
import wandb
import kenlm
from pyctcdecode import build_ctcdecoder
import multiprocessing

from nemo.collections.asr.metrics.wer import word_error_rate

from speachy.asr.decoding.ngram import decode_lm
from nemo.collections.asr.models.rnn_scctc_bpe_models import RNNEncDecSCCTCModelBPE as ModelClass
from speachy.asr.utils import load_audio_model as load_model
from speachy.utils.misc import save_json

import non_iid_dataloader

import pickle as pkl

from speachy.utils.general import load_checkpoint
from speachy.utils.helpers import  exists, isfalse, istrue

def kenlm_decoder(arpa_, vocab, alpha=0.6, beta=0.8):  
    arpa = arpa_ if arpa_ != '' else None
    alpha = alpha if arpa_ != '' else None
    beta = beta if arpa_ != '' else None
    decoder = build_ctcdecoder(vocab, kenlm_model_path=arpa, alpha=alpha, beta=beta)
    print(f'Loaded KenLM model from {arpa} with alpha={alpha} and beta={beta}')
    return decoder

def create_subbatches_eval(audio, audio_lens, text, speakers, segment_lens, metadata):
    max_segment_len = segment_lens.max()
    text = np.array(text)
    speakers = np.array(speakers)
    metadata = np.array(metadata)

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
        cur_audio, cur_audio_lens, cur_text, cur_speakers, cur_metadata = audio[sbi], audio_lens[sbi], text[sbi], speakers[sbi], metadata[sbi]
 
        # trim audio and tokens to max length in sub batch
        max_cur_audio_len = cur_audio_lens.max()
        cur_audio = cur_audio[:, :max_cur_audio_len]
        sub_batches.append({
            'audio': cur_audio,
            'audio_lens': cur_audio_lens,
            'text': cur_text.tolist(),
            'speakers': cur_speakers.tolist(),
            'metadata': cur_metadata.tolist(),
            'prev_state_indices': prev_non_empty_fetch[i] if i > 0 else None, # for the first sub batch there is no previous state  
        })
        
    return sub_batches

@torch.no_grad()
def ___evaluate(args, model, corpus, decoder):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    hyps = []
    refs = []
    speakers = []
    encoded_lens = []

    dataloader = non_iid_dataloader.get_eval_dataloader(
        corpus, 
        max_duration=args.max_duration, 
        return_speaker=True, 
        batch_size=args.num_meetings, 
        concat_samples=False,
        split_speakers=args.split_speakers,
        gap=args.gap,
        speaker_gap=args.speaker_gap,
        single_speaker_with_gaps=args.single_speaker_with_gaps,
        max_allowed_utterance_gap=args.max_allowed_utterance_gap,
        shuffle=False,
    )
    audio_total = 0
    pbar = tqdm(dataloader, total=len(dataloader))
    for ix, batch in enumerate(pbar):
        sub_batch = {
            'audio': batch['audio'],
            'audio_lens': batch['audio_lens'],
            'text': batch['text'],
            'speakers': batch['speakers'],
            'segment_lens': batch['segment_lens'],
        }
        prev_states, prev_state_lens = None, None
        print('###########')
        
        
        #print(f'Step: {iz+1} / {len(sub_batches)}')
        audios = sub_batch['audio'].reshape(-1, sub_batch['audio'].shape[-1]).to(device)
        

        audio_lengths = sub_batch['audio_lens'].reshape(-1).to(device)

    
        sub_batch['text'] = [sub_batch['text']] if sub_batch['text'][0].__class__ != list else sub_batch['text']
        sub_batch['speakers'] = [sub_batch['speakers']] if sub_batch['speakers'][0].__class__ != list else sub_batch['speakers']
        targets = [el[0].replace(" '", "'") for el in sub_batch['text']]
        speaker_ids = ["_".join(el[0]) for el in sub_batch['speakers']]

        if exists(prev_states) and exists(sub_batch['prev_state_indices']) and exists(prev_state_lens): 
            prev_states = prev_states[ : , sub_batch['prev_state_indices']]
            prev_state_lens = prev_state_lens[sub_batch['prev_state_indices']]

        model_inputs = {
            'input_signal': audios,
            'input_signal_length': audio_lengths,
            'cached_kvs': prev_states,
            'cached_kv_lens': prev_state_lens,
        }
    
        model_out = model.forward(**model_inputs)
        log_probs, _, encoded_len, additional_outputs = model_out[0], model_out[1], model_out[2], model_out[-1]
        cached_kvs, full_kv_lens = additional_outputs['kvs_to_cache'], additional_outputs['full_kv_lens']
        prev_states, prev_state_lens = None, None
        log_probs = log_probs.cpu().numpy()
        
        decoded = decode_lm(log_probs, decoder, beam_width=args.beam_size, encoded_lengths=encoded_len)
        decoded = [el.replace(" '", "'") for el in decoded] # change this in training so that it's not needed here

        print(f'Decoded: {" - ".join([el for el in decoded])}\n')
        print(f'Targets: {" - ".join([el for el in targets])}')
        
    
        hyps.extend(decoded)
        refs.extend(targets)
        speakers.extend(speaker_ids)
        encoded_lens.extend(encoded_len.cpu().tolist())
        print('###########')
    print(f'AUDIOTOTAL: {audio_total}')
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


@torch.no_grad()
def evaluate(args, model, corpus, decoder):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    hyps = []
    refs = []
    speakers = []
    encoded_lens = []

    ## deterministic stuff:
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)
    #torch.use_deterministic_algorithms(True)
    #torch.backends.cudnn.deterministic = True
    ##

    dataloader = non_iid_dataloader.get_eval_dataloader(
        corpus, 
        max_duration=args.max_duration, 
        return_speaker=True, 
        batch_size=args.num_meetings, 
        concat_samples=False,
        split_speakers=args.split_speakers,
        gap=args.gap,
        speaker_gap=args.speaker_gap,
        single_speaker_with_gaps=args.single_speaker_with_gaps,
        max_allowed_utterance_gap=args.max_allowed_utterance_gap,
        return_meta_data=True,
        shuffle=False,
    )

    pbar = tqdm(dataloader, total=len(dataloader))
    for ix, batch in enumerate(pbar):
        sub_batches = create_subbatches_eval(**batch)
        prev_states, prev_state_lens = None, None
        print('###########')
        
        
        for iz, sub_batch in enumerate(sub_batches):
            print(f'Step: {iz+1} / {len(sub_batches)}')
            audios = sub_batch['audio'].reshape(-1, sub_batch['audio'].shape[-1]).to(device)
            audio_lengths = sub_batch['audio_lens'].reshape(-1).to(device)
        
            sub_batch['text'] = [sub_batch['text']] if sub_batch['text'][0].__class__ != list else sub_batch['text']
            sub_batch['speakers'] = [sub_batch['speakers']] if sub_batch['speakers'][0].__class__ != list else sub_batch['speakers']
            targets = [el[0].replace(" '", "'") for el in sub_batch['text']]
            speaker_ids = ["_".join(el[0]) for el in sub_batch['speakers']]

            if exists(prev_states) and exists(sub_batch['prev_state_indices']) and exists(prev_state_lens): 
                prev_states = prev_states[ : , sub_batch['prev_state_indices']]
                prev_state_lens = prev_state_lens[sub_batch['prev_state_indices']]

            model_inputs = {
                'input_signal': audios,
                'input_signal_length': audio_lengths,
                'cached_kvs': prev_states,
                'cached_kv_lens': prev_state_lens,
            }
        
            model_out = model.forward(**model_inputs)
            log_probs, _, encoded_len, additional_outputs = model_out[0], model_out[1], model_out[2], model_out[-1]
            cached_kvs, full_kv_lens = additional_outputs['kvs_to_cache'], additional_outputs['full_kv_lens']
            prev_states, prev_state_lens = cached_kvs, full_kv_lens

            log_probs = log_probs.cpu().numpy()
            
            decoded = decode_lm(log_probs, decoder, beam_width=args.beam_size, encoded_lengths=encoded_len)
            decoded = [el.replace(" '", "'") for el in decoded] # change this in training so that it's not needed here

            print(f'Decoded: {" - ".join([el for el in decoded])}\n')
            print(f'Targets: {" - ".join([el for el in targets])}')
          
        
            hyps.extend(decoded)
            refs.extend(targets)
            speakers.extend(speaker_ids)
            encoded_lens.extend(encoded_len.cpu().tolist())
        print('###########')

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
    model = load_model(args=args, model_class=ModelClass)
    if args.checkpoint != '':
        load_checkpoint(args, model)

    print('\nTrainable parameters:'+str(sum(p.numel() for p in model.parameters() if p.requires_grad)))
    print(f'Total parameters: {sum(p.numel() for p in model.parameters())}\n')


    corpus_dict = tools.load_corpus()

    if args.sweep == True:
        wandb.init(config=args, project="ami-ngram-lm-sweep")
        decoder = kenlm_decoder(args.language_model, model.tokenizer.vocab, alpha=wandb.config['alpha'], beta=wandb.config['beta'])
    else:
        decoder = kenlm_decoder(args.language_model, model.tokenizer.vocab, alpha=args.alpha, beta=args.beta)
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
