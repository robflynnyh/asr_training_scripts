import argparse
import torch
from nemo.collections.asr.models.ctc_bpe_models import EncDecCTCModelBPE
from nemo.collections.asr.models.rnnt_bpe_models import EncDecRNNTBPEModel
import tools
import os
from tqdm import tqdm
import numpy as np
from typing import Dict, List, Optional, Tuple, Union

import wandb
import kenlm

import multiprocessing
import nemo

from nemo.collections.asr.metrics.wer import word_error_rate
from omegaconf.omegaconf import OmegaConf
from model_utils import load_checkpoint, load_nemo_checkpoint, load_model, load_sc_model, write_to_log

#import non_iid_dataloader
from speachy.asr.dataloading import non_iid_dataloader
import nemo.collections.asr as nemo_asr

import pickle as pkl

from tools import isfalse, istrue, exists, save_json
from nemo.collections.asr.metrics.wer import word_error_rate

import speachy
from speachy.ctc_beam_search import BeamSearch, LanguageModel

from functools import partial
import ray
import random

from speachy.rescoring.tools import sort_hypothesis_by_recording, order_recordings_by_start_time


@torch.no_grad()
def get_logits(args, model, corpus):
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

    utt_num = 0
    pbar = tqdm(dataloader, total=len(dataloader))
    for batch_num, batch in enumerate(pbar):
        audios = batch['audio'].reshape(1, -1).to(device)
        audio_lengths = batch['audio_lens'].reshape(1).to(device)
        speaker_ids = ["_".join(el[0]) for el in batch['speakers']]
        targets = [el[0] for el in batch['text']]
        targets = [el.replace(" '", "'") for el in targets] # change this in training so that it's not needed here but i'll keep it for now
        meta_data = batch['metadata'][0][0]
        meta_data['timings'] = {'segment_start':meta_data['timings'][0]['segment_start'], 'segment_end':meta_data['timings'][-1]['segment_end']}
        meta_data['speaker'] = list(set(meta_data['speaker']))
        meta_data['recording_id'] = meta_data['recording_id'][0]
        model_out = model.forward(
            input_signal=audios, 
            input_signal_length=audio_lengths,
            segment_lens=batch['segment_lens'] if isfalse(args.do_not_pass_segment_lens) else None
        )
        log_probs, _, encoded_len = model_out[:3]
        s_lprobs = log_probs.squeeze().cpu()
        outputs = {
            'meta_data': meta_data,
            'speaker_ids': speaker_ids,
            'targets': targets,
            'batch_num': batch_num,
            'probs': s_lprobs,
        }
        hyp_data.append(outputs)

    return hyp_data


def load_pickle(path):
    with open(path, 'rb') as f:
        pkl_data = pkl.load(f)
    return pkl_data['stage'], pkl_data['data']

def save_pickle(path, obj, stage='logits'):
    with open(path, 'wb') as f:
        pkl.dump({'stage':stage, 'data':obj}, f) if stage != 'finished' else pkl.dump(obj, f)

def delete_pickle(path):
    os.remove(path)

class argsclass():
    def __init__(self, args:Dict): self.__dict__.update(args)



@ray.remote(num_gpus=0, num_cpus=1)
def run_search_meeting(meeting, id, beam_fn, alpha, beta, randomly_verbose=False, max_gap=100000000, teacher_forcing=False):
    search = beam_fn(log_probs=meeting[0]['probs'], alpha=alpha, beta=beta)
    prev_end = None
    prev_cache = None
    for i, _ in tqdm(enumerate(meeting), total=len(meeting)):
        #print(f'Utterance {i+1}/{len(meeting)}')
        start_t, end_t = meeting[i]['meta_data']['timings'].values()
        search.run_search(use_tqdm=False)
        print(search.return_text(0)) if randomly_verbose and random.random() < 0.1 else None # print 5% of the time lol

        if i < len(meeting)-1:
            search.next_utterance(new_log_probs=meeting[i+1]['probs'])
            if prev_end is None or (start_t - prev_end) < max_gap:
                target = meeting[i]['targets'][0] if teacher_forcing else None
                search.next_utterance(new_log_probs=meeting[i+1]['probs'], teacher_forcing=target, prev_cache=prev_cache)
                prev_cache = {k:v.clone() for k,v in search.beams[0].state.items()} if teacher_forcing else None
            else:
                print('RESETTING HISORY')
                prev_cache = None
                lm_probs, state = search.language_model.get_initial_state()
                search.beams = [search.beams[0]] # take the best beam
                search.beams[0].state = state # reset the state
                search.beams[0].next_lm_token_lps = lm_probs # reset the language model probs
                if search.beams[0].am_sequence[-1] != search.blank_id:
                    search.beams[0].am_sequence.append(search.blank_id) # prevent collapse across utterances
                    
        prev_end = end_t
    #text_out = text_out.strip() + ' ' + search.return_text(0).strip()

    return {'id':id, 'text':search.return_text(0)}

def get_target_hyp_pairs(hyp_data, hyps):
    hyp_lst, targets = [], []
    for output in hyps:
        k, txt = output['id'], output['text']
        target = " ".join(el['targets'][0].strip() for el in hyp_data[k])
        hyp_lst.append(txt.strip())
        targets.append(target)
    return hyp_lst, targets

def write_to_log(log_path, text):
    with open(log_path, 'a') as f:
        f.write(text+'\n')

def main(args):
    model = load_model(args) if args.self_conditioned == False else load_sc_model(args)
    if args.checkpoint != '':
        load_checkpoint(args, model)
    print('\nTrainable parameters:'+str(sum(p.numel() for p in model.parameters() if p.requires_grad)))
    print(f'Total parameters: {sum(p.numel() for p in model.parameters())}\n')
    corpus_dict = tools.load_corpus()

    config = speachy.utils.general.load_config('../experiment_configs/lm/decoder_pg19_sep_token_ted_am.yaml')
    tokenizer_path = '.'+os.path.join(config['model']['tokenizer']['dir'], 'tokenizer.model')
    tokenizer = speachy.utils.general.load_tokenizer(tokenizer_path)
    lm_model = speachy.lm.tools.loading.autoload(config=config, tokenizer=tokenizer)
    _,_ = speachy.utils.general.load_checkpoint(
        args = argsclass({'checkpoint':'../checkpoints/open_sub_ft_ted/ft_ted_checkpoint_1259_id_36.pt'}),
        model = lm_model,
        force_cpu = True
    )

    temp_name_dev = f'dev_{args.load_tmp}'
    dev_stage, test_stage = None, None # stage corresponds to the last step of the pipeline that was completed
    print(f'Fetching logits for dev set...')
    if os.path.exists(os.path.join(args.tmp_dir, temp_name_dev)):
        dev_stage, dev_hyps = load_pickle(os.path.join(args.tmp_dir, temp_name_dev))

    if dev_stage == None:
        dev_hyps = get_logits(args, model, corpus_dict['dev'])
        save_pickle(os.path.join(args.tmp_dir, temp_name_dev), dev_hyps, stage='logits')
        dev_stage = 'logits'
    del model

    dev_hyps = sort_hypothesis_by_recording(dev_hyps)
    dev_hyps = order_recordings_by_start_time(dev_hyps)

    num_meetings = len(dev_hyps.keys())

    alpha_range = [0.7]
    beta_penalty = [0.0]

    write_to_log('beam_search_log.txt', f'alpha_range: {alpha_range}, beta_range: {beta_penalty} Initialising beam search...')
    
    ray.init(num_cpus=25, num_gpus=0)
    cutoff = -6
    beamsearch_fn = partial(
        BeamSearch, 
        language_model=LanguageModel(
            model=lm_model, 
            bos_id=0, 
            device='cuda' if torch.cuda.is_available() else 'cpu',
            half_precision=True if torch.cuda.is_available() else False
        ),
        tokenizer=tokenizer, 
        beam_width=25,
        blank_id=128,
        top_am_threshold=cutoff,
        max_cache_length = -1,
        debug=False
    )
    beamsearch_fn = ray.put(beamsearch_fn) # put beamsearch_fn on the ray object store so that it can be accessed by the remote function
    # select random sample of dev hyps (10%)
    # runs same random seed
    random.seed(36)
    dev_hyps_sample = dev_hyps 
    # select only 3 of the meetings (key)
    #dev_hyps_sample = {k:dev_hyps_sample[k] for k in random.sample(list(dev_hyps_sample.keys()), 3)}
    #k = random.choice(list(dev_hyps_sample.keys()))
    #dev_hyps_sample = {k:dev_hyps_sample[k]}
    while True:
        alpha = np.random.choice(alpha_range)
        beta = np.random.choice(beta_penalty)
        write_to_log('beam_search_log.txt', f'alpha: {alpha}, beta: {beta}')
  
        meetings = list(dev_hyps_sample.values())
        names = list(dev_hyps_sample.keys())
        outputs = [
            run_search_meeting.remote(
                meeting, 
                name, 
                beamsearch_fn, 
                alpha, 
                beta, 
                randomly_verbose=True, 
                max_gap=args.max_allowed_utterance_gap,
                teacher_forcing=args.teacher_forcing
            ) for meeting, name in zip(meetings, names)]
        outputs = ray.get(outputs)

        predictions, targets = get_target_hyp_pairs(dev_hyps_sample, outputs)

        '''with open('pred_tst.pkl', 'wb') as f:
            pkl.dump({'pred':predictions, 'targets':targets}, f)'''

        wer = word_error_rate(hypotheses=predictions, references=targets)
        print(f'WER: {wer}')
     
        write_to_log('beam_search_log.txt', f'WER: {wer} w/ alpha: {alpha}, beta: {beta}, cutoff: {cutoff}')
        exit()


        
    #save_pickle(os.path.join(args.tmp_dir, args.load_tmp+'_devBEAM.pkl'), dev_hyps, stage='finished')




if __name__ == '__main__':
    ''''
    Note I've only written this for a batch size of 1 (lazy)
    '''
    parser = argparse.ArgumentParser() 

    parser.add_argument('--teacher_forcing', action='store_true', help='whether to use teacher forcing')

    parser.add_argument('-load_tmp', '--load_tmp', default='ted_test.pkl', type=str, help='base name of logit hyp to load (full name = split+_+name')
    parser.add_argument('-tmp_dir','--tmp_dir', type=str, default='./tmp', help='path to tmp dir')

    parser.add_argument('-log_beta', '--log_beta', action='store_true', help='whether to use log scale for beta length penalty')

    parser.add_argument('--load_pretrained', action='store_true')
    parser.add_argument('--pretrained', type=str, default='stt_en_conformer_ctc_small') # stt_en_conformer_ctc_large stt_en_conformer_transducer_large
    parser.add_argument('--model_config', type=str, default='../model_configs/conformer_sc_ctc_bpe_small.yaml') 

    parser.add_argument('--tokenizer_model', type=str, default='./tokenizers/tokenizer_spe_bpe_v128/tokenizer.model', help='path to tokenizer model')
    parser.add_argument('--max_duration', type=float, default=0, help='max duration of audio in seconds')


    parser.add_argument('--log_file', type=str, default='eval_log.txt')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints')
    parser.add_argument('--checkpoint', type=str, default='checkpoint_68_id_15.pt')
    

    parser.add_argument('--beam_size', type=int, default=300)
    parser.add_argument('--bpe_lm_path', type=str, default='./ngrams/binary_bpe/ami_6grambpe.bin')

    parser.add_argument('-lm', '--language_model', type=str, default='./ngrams/cantab_interp_ami.arpa', help='arpa n-gram model for decoding')#./ngrams/3gram-6mix.arpa
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--beta', type=float, default=0.8)

    parser.add_argument('-token_skip', '--token_min_logp', default=-5, type=float)
    parser.add_argument('-beam_prune', '--beam_prune_logp', default=-10000, type=float)

    parser.add_argument('-nsc','--not_self_conditioned', action='store_true', help='use for non self-conditioned models')

    parser.add_argument('-mgap','--max_allowed_utterance_gap', type=float, default=10.0, help='max allowed gap between utterances in seconds')


    parser.add_argument('-gap','--gap', default=0.1, type=float, help='gap between utterances when concatenating')

    parser.add_argument('--single_speaker_with_gaps', action='store_true', help='if set, utterances will contain 1 speaker and additional gaps of speaker_gap will be added if there is a speaker change between two utternces of the same speaker')
    parser.add_argument('--speaker_gap', type=float, default=1.0, help='for use with single_speaker_with_gaps, will add this many seconds of silence between utterances of the same speaker when there is a speaker change in between them')

    parser.add_argument('--split_speakers', action='store_true', help='if set, wont concat samples from different speakers, (concat_samples must be enabled)')

    parser.add_argument('-psl','--pass_segment_lengths', action='store_true', help='if set, will pass segment lens to the model, used with concat_samples for multi segment models')
    parser.add_argument('-save','--save_outputs', default='', type=str, help='save outputs to file')

    args = parser.parse_args()

    assert args.language_model != '' and args.bpe_lm_path != '', 'Must provide a language model and a bpe language model'

    args.do_not_pass_segment_lens = not args.pass_segment_lengths
    args.self_conditioned = not args.not_self_conditioned
    args.concat_samples = True
    args.config_from_checkpoint_dir = True
    

    '''if args.save_outputs == '':
        save_outputs = ''
        while save_outputs == '':
            save_outputs = input('Please provide a name for the output file: ').strip()
        args.save_outputs = save_outputs'''

    if os.path.exists(args.tmp_dir) == False:
        os.mkdir(args.tmp_dir)

    if args.checkpoint != '':
        args.checkpoint = os.path.join(args.checkpoint_dir, args.checkpoint)



    if args.config_from_checkpoint_dir == True:
        dir_contents = os.listdir(args.checkpoint_dir)
        config = [el for el in dir_contents if el.endswith('.yaml')]
        assert len(config) == 1, 'Exactly one config file must be in checkpoint dir'
        args.model_config = os.path.join(args.checkpoint_dir, config[0])
        print(f'Loading config from checkpoint dir: {args.model_config}')

    main(args)
