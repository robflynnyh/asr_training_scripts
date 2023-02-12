from nemo.collections.asr.metrics.wer import word_error_rate
import argparse 
import pickle as pkl
from tqdm import tqdm

from importlib import reload as rl

import torch
import torch
import random

import os 
from einops import rearrange
from speachy.rescoring.tools import (
        sort_hypothesis_by_recording, 
        order_recordings_by_start_time,
        interpolate
)


from speachy.lm.tools.loading import autoload
from speachy.rescoring.scripts.compute_rescore_wer import main as compute_rescore_wer
from speachy.utils.misc import write_trn_files, eval_with_sclite
from speachy.utils.helpers import request_env, isfalse, istrue, exists

from speachy.lm.tools.train import add_bos as add_bos_token
import time

from speachy.utils.general import (
    load_config,
    load_checkpoint,
    load_tokenizer,
    write_to_log
)

class argsclass:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


@torch.no_grad()
def get_text_probability(args, model, tokenizer, text, cached_states=None, next_target=None, first_token=None, duration_data=None):
    def calc_length_penalty(lp_length):
        return torch.tensor(lp_length).float().log().item()

    device = torch.device(args.device)
    tokens = tokenizer.text_to_ids(text)
    tokens = torch.tensor(tokens).unsqueeze(0).long()

    if exists(first_token):
        tokens = torch.cat([first_token[None,None], tokens], dim=-1) # add first token from previous utterance
        if exists(cached_states):
            cached_states['cache_lengths'] -= 1 # trim last token from cache
            cached_states['cache'] = cached_states['cache'][:, :, :, :, :-1]

    last_token = tokens[0, -1] if len(tokens[0]) > 0 else None

    add_bos = cached_states is None or args.eosbos or args.bos 
  
    token_lens = torch.tensor([tokens.shape[-1]]) + (1 if add_bos else 0)
    if tokens.shape[-1] == 0: # (should carry over last token and remove it from cache)
        return torch.tensor(torch.nan), cached_states, calc_length_penalty(1), last_token
  
    tokens, token_lens = tokens.to(device), token_lens.to(device)
    tokens = add_bos_token(tokens, bos_token_id=0) if add_bos else tokens # don't add bos if we're starting from a cached state (bos is already there)
    targets = tokens.clone()
    sep = exists(cached_states)
    
    if not sep:
        targets[:, :-1] = tokens[:, 1:] # shift targets by 1 

    if not args.eosbos:
        targets = targets[:, :-1] # remove last token 
    else:
        targets[:, -1] = 0 # set last token to eos

    #print(token_lens)
    token_lens+=int(sep)
    logits, _, cached_states = model(x=tokens, length=token_lens, cache=cached_states, durations=duration_data, sep=sep)
    #print(cached_states['cache'].shape)

    # remove first and last token 
    toadd = 1 if add_bos else 0
    logits = logits[:, toadd:-1, :] if not exists(next_target) else logits[:, toadd:, :] # remove last target if not provided
    targets = targets[:, toadd:] # 
    
    if not args.eosbos:
        logits = logits[:, :, 1:] # remove eos/bos from probabilities
        targets -= 1 # shift targets by 1 (no more eos/bos)

    # temperature
    logits = logits / args.temperature
    logprobs = torch.nn.functional.log_softmax(logits, dim=-1)

    logprobs = logprobs.squeeze(0).gather(1, targets.squeeze(0).unsqueeze(1)).squeeze() 
    logprobslen = logprobs.shape[0] if len(logprobs.shape) > 0 else 1
    logprobs = logprobs.sum() 

    return logprobs.to('cpu'), cached_states, calc_length_penalty(logprobslen), last_token


def trim_cache(args, kv_cache, max_len):
    if max_len == 0:
        return None
    if kv_cache is None:
        return None

    if max_len == -1:
        return kv_cache
    #print(kv_cache['cache'].shape)
    if kv_cache['cache_lengths'] > max_len:
        bos = kv_cache['cache'][:, :, :, :, 0, :].unsqueeze(-2).clone()
        kv_cache['cache'] = kv_cache['cache'][:, :, :, :, -max_len:, :]

        if not args.length_prediction or args.eosbos:
            kv_cache['cache'] = torch.cat([bos, kv_cache['cache']], dim=-2)

        kv_cache['cache_lengths'] = torch.tensor([kv_cache['cache'].shape[-2]]).to(kv_cache['cache_lengths'].device)
    return kv_cache

'''def calc_score_old(args, hyp, tlm_mean, tlm_std):
    am_prob = hyp['am_score']
    ngram_prob = hyp['second_pass_score'] - am_prob

    lm_prob = (hyp['tlm_prob'] - tlm_mean) / tlm_std
    length_penalty = hyp['length_penalty']
    pen_length = length_penalty * args.length_penalty
    prob = lm_prob - pen_length
    rescore_prob = (am_prob + ngram_prob + prob).item()
    return rescore_prob'''


def calc_score(hyp, hyperparams):
    score = hyp['am_score']
    ngram_prob, ngram_weight = hyp['ngram_lm_score'], hyperparams['ngram_scale']
    score += ngram_prob * ngram_weight
    bpe_lm_score, bpe_lm_weight = hyp['bpe_lm_score'], hyperparams['bpe_lm_weight']
    score += bpe_lm_score * bpe_lm_weight
    tlm_score, tlm_weight = (hyp['tlm_prob'] - hyperparams['tlm_mean']) / hyperparams['tlm_std'], hyperparams['tlm_scale']
    score += tlm_score * tlm_weight
    bpe_length_penalty, bpe_length_penalty_weight = hyp['first_pass_length_penalty'], hyperparams['bpe_length_penalty_weight']
    score += bpe_length_penalty * bpe_length_penalty_weight
    tlm_log_length_penalty, tlm_length_penalty_weight = hyp['length_penalty'], hyperparams['length_penalty']
    score += tlm_log_length_penalty * tlm_length_penalty_weight
    return score


def rescore(args, recording_hyps, hyperparams):
    max_beam = args.stop_at_beam 

    for utt in recording_hyps:
        best_log_p, best_hyp, best_beam = float('-inf'), '', 0
        print(f'Target: {utt["targets"][0]}\n') if args.verbose else None

        utt_data = {k:[] for k in ['am_score', 'ngram_lm_score', 'bpe_lm_score', 'first_pass_length_penalty', 'tlm_prob', 'length_penalty']}
        for idx in utt['beams'][0].keys(): # get all the data for the current utt so calculations can be vectorized
            cur = utt['beams'][0][idx]
            if idx >= max_beam:
                break
            if 'tlm_prob' not in cur:
                continue
            for k in utt_data.keys():
                utt_data[k].append(cur[k])

        for k in utt_data.keys():
            utt_data[k] = torch.tensor(utt_data[k])
        scores = calc_score(utt_data, hyperparams=hyperparams)

        for idx in utt['beams'][0].keys():
            cur = utt['beams'][0][idx]
            if idx >= max_beam:
                break
            if 'tlm_prob' not in cur:
                continue

            hyptext = cur['text']
            recore_prob = scores[idx].item()

            cur['rescore_lp'] = recore_prob
   
            print(f'beam: {idx}, prob: {cur["rescore_lp"]}, hyp: {hyptext}\n') if args.verbose else None

            if cur['rescore_lp'] > best_log_p:
                best_log_p, best_hyp, best_beam = cur['rescore_lp'], hyptext, idx
          
        target_txt, top_hyp = utt['targets'][0], utt['beams'][0][0]['text']
        if args.verbose:
            original_wer, rescored_wer = word_error_rate([target_txt], [top_hyp]), word_error_rate([target_txt], [best_hyp])
            print(f'\n\nOriginal WER: {original_wer}, rescored WER: {rescored_wer}, best beam: {best_beam}\n\n') 
            print(f'{"-"*10} WER Improved! {"-"*10}\n\n') if rescored_wer < original_wer else None 
            print(f'{"-"*10} WER Degradation {"-"*10}\n\n') if rescored_wer > original_wer else None

        utt['best_logp'] = best_log_p
        utt['best_hyp'] = best_hyp
        print(f'best logp: {best_log_p}') if args.verbose else None
    return recording_hyps


def prepare_for_sclite(hypothesis):
    hyps, refs, speakers, utt_durations = [], [], [], []
    for key in hypothesis.keys():
        recording = hypothesis[key]
        for utt in tqdm(recording):
            seg_start, seg_end = utt['meta_data']['timings'].values()
            dur_sec = seg_end - seg_start
            best_hyp = utt['best_hyp']
            speaker = "_".join(utt['meta_data']['speaker'])
            utt_durations.append(dur_sec)
            target = utt['targets'][0]
            speakers.append(speaker)
            hyps.append(best_hyp)
            refs.append(target)
    return hyps, refs, speakers, utt_durations

def get_hyperparameters(args):
    return {
        'tlm_mean': torch.tensor(args.tlm_mean),
        'tlm_std': torch.tensor(args.tlm_std),
        'bpe_lm_weight': torch.tensor(args.bpe_lm_weight),
        'tlm_scale': torch.tensor(args.tlm_scale),
        'ngram_scale': torch.tensor(args.ngram_scale),
        'bpe_length_penalty_weight': torch.tensor(args.bpe_length_penalty_weight),
        'length_penalty': torch.tensor(args.length_penalty),
        }

def get_next_target(args, next_utt, prev_end, tokenizer):
    '''gets first word of next utterance if it is within max_utt_gap of previous utterance'''
    utt = next_utt   
    segment_start, _ = utt['meta_data']['timings'].values()
    if prev_end - segment_start > args.max_utt_gap:
        return None
    top_beam_text = utt['beams'][0][0]['text']
    top_beam_first_word = top_beam_text.split()
    if len(top_beam_first_word) == 0:
        return None
    top_beam_first_word = top_beam_first_word[0]
    return tokenizer.text_to_ids(top_beam_first_word)[0] # first token of first word(


def compute_beam_ppls(args, model, tokenizer, recording_hyps, hyperparameters=None): # remove all the random cloning..
    max_beam = args.stop_at_beam 
    max_history = args.max_history_len
    device = torch.device(args.device)
    prev_end = None # previous utterance end time
    last_token = None # last token of previous utterance
    kv_cache = None
    kvs_to_cache = None


    for i, utt in enumerate(tqdm(recording_hyps)):
        kv_cache = {'cache': kvs_to_cache['cache'].clone(), 'cache_lengths': kvs_to_cache['cache_lengths'].clone()} if kvs_to_cache is not None else None 

        segment_start, segment_end = utt['meta_data']['timings'].values()
        duration = segment_end - segment_start
        duration_data = torch.tensor([duration])[:, None].to(args.device) if args.length_prediction else None
 
        next_target = get_next_target(args, recording_hyps[i+1], segment_end, tokenizer) if i<len(recording_hyps)-1 else None
        next_target = 0 if args.eosbos else next_target # if we are modelling boundaries, we don't need the next word
        
        prev_end = segment_start if prev_end is None else prev_end

        expire_history = True if segment_start - prev_end > args.max_utt_gap else False
        last_token = None if expire_history else last_token
        kv_cache = None if expire_history > args.max_utt_gap else kv_cache
        kv_cache = trim_cache(args, kv_cache, max_history)
        prev_end = segment_end

        if args.use_targets: # uses target hypothesis as history (rather than ASR output)
            target = utt['targets'][0]
            _, cache, _, next_first_token = get_text_probability(args, model, tokenizer, target, cached_states=kv_cache, next_target=next_target)
            kvs_to_cache = {'cache': cache['cache'].clone(), 'cache_lengths': cache['cache_lengths'].clone()}  #debug thing'''''

        has_stats = exists(hyperparameters)
        cached_states = [] 

        for idx in utt['beams'][0].keys():
            cur = utt['beams'][0][idx]
            if idx >= max_beam:
                break
            if args.use_cached_scores and 'tlm_prob' not in cur:
                continue

            hyptext = cur['text']
            am_prob = torch.tensor(cur['am_score'])
            ngram_prob = torch.tensor(cur['second_pass_score']) - am_prob

            tlm_prob, cache, length_penalty, last_token_ = \
                get_text_probability(
                    args, 
                    model, 
                    tokenizer, 
                    hyptext, 
                    cached_states=kv_cache, 
                    next_target=None, # remove
                    first_token=None, # remove
                    duration_data=duration_data
                )

            # do this by default
            if idx == 0 and not has_stats and not args.use_targets: # if not using targets, and there is no hyperpameters to compute a score, then we will use the first beam as the cache
                kvs_to_cache = {'cache': cache['cache'].clone(), 'cache_lengths': cache['cache_lengths'].clone()} if cache is not None else None
                next_first_token = last_token_
             
            
            tlm_prob = ngram_prob if tlm_prob.isnan() or tlm_prob == float('-inf') or tlm_prob == 0 else tlm_prob # replace with ngram_probability if it is nan or inf 
            cur['tlm_prob'] = tlm_prob
            cur['length_penalty'] = length_penalty

            if has_stats and not args.use_targets:
                rescore_prob = calc_score(cur, hyperparameters)
                if exists(cache):
                    cached_states.append((rescore_prob, {'cache': cache['cache'].clone().cpu(), 'cache_lengths': cache['cache_lengths'].clone().cpu()}, last_token_))

        if has_stats and not args.use_targets:
            cached_states = sorted(cached_states, key=lambda x: x[0], reverse=True)
            kvs_to_cache = {k:v.to(device) for k,v in cached_states[0][1].items() if type(v) == torch.Tensor}
            next_first_token = cached_states[0][2]
            del cached_states 
        
        if not args.eosbos:
            last_token = next_first_token 
        

    return recording_hyps


def compute_lm_ppls(args, model, tokenizer, hypothesis, hyperparameters=None):
    for i, key in enumerate(hypothesis.keys()):
        recording = hypothesis[key]
        print(f'Computing perplexities for recording {key}, {i+1}/{len(hypothesis.keys())}')
        hypothesis[key] = compute_beam_ppls(args, model, tokenizer, recording, hyperparameters=hyperparameters)
    return hypothesis

def rescore_speakers(args, hypothesis, hyperparmeters):
    for i, key in enumerate(hypothesis.keys()):
        recording = hypothesis[key]
        #print(f'rescoring for recording {key}, {i+1}/{len(hypothesis.keys())}')
        hypothesis[key] = rescore(args, recording, hyperparmeters)
    return hypothesis

def get_standardisation_stats(hypothesis):
    tlm_scores = []
    for i, key in enumerate(hypothesis.keys()):
        recording = hypothesis[key]
        print(f'getting standardisation stats for recording {key}, {i+1}/{len(hypothesis.keys())}')
        for utt in tqdm(recording):
            cur = utt['beams'][0][0]
            if 'tlm_prob' not in cur:
                continue

            tlm_score = torch.tensor(cur['tlm_prob'])
            assert not tlm_score.isnan() and not tlm_score == float('-inf') and not tlm_score == 0, 'tlm score is nan or inf or zero, smth is wrong )):'
            tlm_scores.append(tlm_score) 
    tlm_scores = torch.stack(tlm_scores)
    tlm_mean, tlm_std = tlm_scores.mean(), tlm_scores.std()
    print(f'TLM mean: {tlm_mean}, TLM std: {tlm_std}')
    return {'tlm_mean': tlm_mean, 'tlm_std': tlm_std}



def save_hyp(args, hyp):
    if args.saveas != '':
        with open(args.saveas, 'wb') as f:
            pkl.dump(hyp, f)
        print(f'Saved hypothesis to {args.saveas}')

def run_random_search(args, model, tokenizer, hypothesis):
    print(f'Running grid search...')
    stop_at_beam = args.stop_at_beam
    #args.use_targets = True # use targets to get initial standardisation stats
    #args.stop_at_beam = 100 # only use top 100 beams to get stats
    if not args.use_cached_scores:
        print(f'Evaluating to get hypothesis... using top {args.stop_at_beam} beams ')
        scored_hypothesis = compute_lm_ppls(args, model, tokenizer, hypothesis)
        save_hyp(args, scored_hypothesis)
    else:
        scored_hypothesis = hypothesis
    #args.stop_at_beam = stop_at_beam
    standardisation_stats = get_standardisation_stats(scored_hypothesis)
    write_to_log(log_file=args.log_path, data=f'TLM mean: {standardisation_stats["tlm_mean"]}, TLM std: {standardisation_stats["tlm_std"]}')
    args.tlm_mean = standardisation_stats['tlm_mean']
    args.tlm_std = standardisation_stats['tlm_std']
    
    bpe_lm_weights_range = [-0.6, 0.6]
    tlm_scales = [25.0, 70.0]
    ngram_scales = [0.0, 1.25]
    length_penalties = [0.0, 0.0] #not used anymore
    bpe_length_penalty_weights = [0.2, 3.75]
    print(f'Running Random search')
   
    scores_v_params = []
    lowest_wer = 1.0
    start_time = time.time()
    try:
        while (time.time() - start_time) < args.random_search_time:
            bpe_lm_weight = random.uniform(bpe_lm_weights_range[0], bpe_lm_weights_range[1])
            tlm_scale = random.uniform(tlm_scales[0], tlm_scales[1])
            ngram_scale = random.uniform(ngram_scales[0], ngram_scales[1])
            length_penalty = random.uniform(length_penalties[0], length_penalties[1])
            bpe_length_penalty_weight = random.uniform(bpe_length_penalty_weights[0], bpe_length_penalty_weights[1])
            
            #print(f'Running grid search with params: bpe_lm_weight: {bpe_lm_weight}, tlm_scale: {tlm_scale}, ngram_scale: {ngram_scale}, length_penalty: {length_penalty}, bpe_length_penalty_weight: {bpe_length_penalty_weight}')
            hyperparameters = {
                'tlm_mean': torch.tensor(args.tlm_mean),
                'tlm_std': torch.tensor(args.tlm_std),
                'bpe_lm_weight': torch.tensor(bpe_lm_weight),
                'tlm_scale': torch.tensor(tlm_scale),
                'ngram_scale': torch.tensor(ngram_scale),
                'bpe_length_penalty_weight': torch.tensor(bpe_length_penalty_weight),
                'length_penalty': torch.tensor(length_penalty),
            }
            hyp = rescore_speakers(
                args=args,
                hypothesis=scored_hypothesis,
                hyperparmeters=hyperparameters,
            )
            score = compute_rescore_wer(hyp, verbose=False)
            scores_v_params.append((score, hyperparameters))
            if score < lowest_wer:
                lowest_wer = score
                print(f'Lowest WER: {score} with params: bpe_lm_weight: {bpe_lm_weight}, tlm_scale: {tlm_scale}, ngram_scale: {ngram_scale}, length_penalty: {length_penalty}, bpe_length_penalty_weight: {bpe_length_penalty_weight}')

    except KeyboardInterrupt:
        print('Keyboard interrupt, stopping search')
    scores_v_params = sorted(scores_v_params, key=lambda x: x[0])
    print(f'Best params: {scores_v_params[0][1]}')
    print(f'Best wer: {scores_v_params[0][0]}')
    return scores_v_params[0][1]

#Lowest WER: 0.09333851880660037 with params: bpe_lm_weight: 0.3, tlm_scale: 30.0, ngram_scale: 0.5, length_penalty: 1.5, bpe_length_penalty_weight: 1.0


def main(args, hypothesis):
    if args.use_cached_scores == False:
        device = torch.device(args.device)
        config = load_config(args.config)
        tokenizer_path = os.path.join(config['model']['tokenizer']['dir'], 'tokenizer.model')
        tokenizer = load_tokenizer(tokenizer_path)
        model = autoload(config=config, tokenizer=tokenizer)
        epoch, val_loss  = load_checkpoint(args=argsclass(**{'checkpoint': args.checkpoint}), model=model, force_cpu=True)
        modeltype = config['model']['modeltype']
        print(f'Loaded model {args.checkpoint} with epoch {epoch} and val_loss {val_loss}\n Model type: {modeltype}')
        model.to(device)
        model.eval()
    else:
        model, tokenizer = None, None
    
    if hypothesis.__class__.__name__ == 'list': # only a list if it hasn't been processed yet   
        assert args.use_cached_scores == False, 'need processed hypothesis lists to use cached scores'
        hypothesis = sort_hypothesis_by_recording(hypothesis)
        hypothesis = order_recordings_by_start_time(hypothesis)

    hyperparameters = get_hyperparameters(args=args)
    if args.run_random_search:
        hyperparameters = run_random_search(args, model, tokenizer, hypothesis)
        if args.use_top_sample:
            hypothesis = compute_lm_ppls(args, model, tokenizer, hypothesis, hyperparameters=hyperparameters)
    elif args.use_cached_scores == False:
        hypothesis = compute_lm_ppls(args, model, tokenizer, hypothesis, hyperparameters=hyperparameters if not args.use_top_sample else None)

    standardise_stats = get_standardisation_stats(hypothesis=hypothesis)
    hyperparameters['tlm_mean'] = torch.tensor(standardise_stats['tlm_mean'])
    hyperparameters['tlm_std'] = torch.tensor(standardise_stats['tlm_std'])
 
    hypothesis = rescore_speakers(args, hypothesis, hyperparmeters=hyperparameters)

    wer = compute_rescore_wer(hypothesis)
  
    print(f'WER: {wer}')

    save_hyp(args, hypothesis)


    if args.eval_with_sclite != '':
        sclite_path = request_env(env_name='SCLITE_PATH', env_path=args.env_file)
        hyps, refs, speakers, utt_durations = prepare_for_sclite(hypothesis)
        refname, hypname = write_trn_files(
            refs = refs,
            hyps = hyps,
            speakers = speakers,
            encoded_lens = utt_durations,
            fname = 'date',
            out_dir = args.eval_with_sclite
        )
        wer = eval_with_sclite(ref=refname, hyp=hypname, SCLITE_PATH=sclite_path)
        print(f'WER (sclite): {wer}')

#TLM mean: -205.09022521972656, TLM std: 114.70083618164062
# 200, 110 ?


#1kw pretrained: Lowest WER: 0.09550530585032502 with params: bpe_lm_weight: 0.36628218945224245, tlm_scale: 34.718944475402765, ngram_scale: 0.5514452793042264, length_penalty: 1.5627193862644981, bpe_length_penalty_weight: 0.643815814552575
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-hyp", "--hyp", type=str, default='./dev_rescored.pkl')
    parser.add_argument('--config', type=str, default='./experiment_configs/lm/decoder_pg19.yaml')
    parser.add_argument('--device', type=str, default='auto')

    parser.add_argument('-log','--log_path', type=str, default='./grid_search.log')
    #parser.add_argument('--tlm_threshold', help='if TLM logp is lower than this threshold TLM won\'t be interpolated', type=float, default=-20)
    parser.add_argument('--checkpoint', type=str, default='./checkpoints/pg19checkpoints_dropout10_nths/pg_19_ft_checkpoint_47_id_91.pt')
    parser.add_argument('--max_utt_gap', type=float, default=10.0)
    parser.add_argument('-save','--saveas', type=str, default='')

    parser.add_argument('-use_targets','--use_targets', action='store_true', help='whether to use targets as history')
    parser.add_argument('-not_use_top_sample','--not_use_top_sample', action='store_true', help='whether to always use top hypothesis as history')

    parser.add_argument('-random_search', '--run_random_search', action='store_true', help='whether to run random search')
    # hyperparameters for rescore
    parser.add_argument('-bpe_lm_weight','--bpe_lm_weight', type=float, default=0.0841)
    parser.add_argument('-bpe_len_pen', '--bpe_length_penalty_weight', type=float, default=2.2083)
    parser.add_argument('-ngram_scale', '--ngram_scale', type=float, default=0.5565) # linearly scale AM logp by this factor')
    parser.add_argument('-length_penalty','--length_penalty', type=float, default=-0.3103) 
    parser.add_argument('-tlm_scale','--tlm_scale', type=float, default=42.5728) # linearly scale TLM logp by this factor
    parser.add_argument('-tlm_mean','--tlm_mean', type=float, default=-168.37120056152344,) # mean of TLM logp
    parser.add_argument('-tlm_std','--tlm_std', type=float, default=105.40515899658203) # std of TLM logp
    # hyperparameters for rescore
    

    parser.add_argument('-random_search_time', '--random_search_time', type=float, default=1000) # time limit for random search

    parser.add_argument('--temperature', type=float, default=0.85) # softmax temperature for TLM (sharpness of distribution, will punish mistakes more)
    parser.add_argument('-history','--max_history_len', type=int, default=-1)
    parser.add_argument('--stop_at_beam', type=int, default=25)
    
    parser.add_argument('-use_cache','--use_cached_scores', action='store_true', help='whether to use cached scores from previous runs rather than recomputing them ')
    parser.add_argument('-v','--verbose', action='store_true', help='whether to print out the rescored hypothesis')
    parser.add_argument('-sclite', '--eval_with_sclite', default='', help='false if blank, path if not. If not blank, will evaluate the rescored hypothesis with sclite and save the results to the specified path')
    
    parser.add_argument('-eosbos','--eosbos', action='store_true', help='whether to model boundary tokens in the TLM')
    parser.add_argument('-length_pred','--length_prediction', action='store_true', help='use length prediction') # not used rlly
    parser.add_argument('-bos','--bos', action='store_true')
    parser.add_argument('-env','--env_file', default='/exp/exp1/acp21rjf/deliberation/speachy/.env', help='path to sclite executable')

    args = parser.parse_args()

    has_stats = exists(args.tlm_mean) and exists(args.tlm_std)
    if not has_stats:
        print("Warning, no TLM stats specified, will use first hypothesis state as history")

    args.use_top_sample = not args.not_use_top_sample

    if args.device == 'auto':
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if args.checkpoint == '':
        print('No checkpoint specified...')
        ckpt = input('Please specify a checkpoint to evaluate: ')
        args.checkpoint = ckpt


    with open(args.hyp, 'rb') as f:
        hyps = pkl.load(f)

    main(args, hyps)


# 500h 1kw ft'd: #Lowest WER: 0.09317184288016 with params: bpe_lm_weight: 0.23888780687924038, tlm_scale: 40.657361448142666, ngram_scale: 0.4422537837665267, length_penalty: 1.791168467081285, bpe_length_penalty_weight: 0.7588113386023086


# zero context pretrained scores:
# - 50w pretrained: Lowest WER: 0.09644980276682037 with params: bpe_lm_weight: 0.24879592263346, tlm_scale: 32.55355768650999, ngram_scale: 0.6457116454439233, length_penalty: 1.9230100697518404, bpe_length_penalty_weight: 0.558161221175397
# - 1kw pretrained: Lowest WER: 0.09550530585032502 with params: bpe_lm_weight: 0.36628218945224245, tlm_scale: 34.718944475402765, ngram_scale: 0.5514452793042264, length_penalty: 1.5627193862644981, bpe_length_penalty_weight: 0.643815814552575
# - 400w pretrained: Lowest WER: 0.09661647869326073 with params: bpe_lm_weight: 0.1521901386453404, tlm_scale: 44.77960158012452, ngram_scale: 0.6522403031850588, length_penalty: 0.7381659348327338, bpe_length_penalty_weight: 0.9636033027905181
