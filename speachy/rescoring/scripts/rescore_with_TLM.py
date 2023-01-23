from tqdm import tqdm
from nemo.collections.asr.metrics.wer import word_error_rate
import argparse 
import pickle as pkl

import tools
from importlib import reload as rl
import non_iid_dataloader as niiddl, lhotse
from tqdm import tqdm
import torch
import torch
from omegaconf.omegaconf import OmegaConf
import lm_utils
import model_utils
from tools import isfalse, istrue, exists
import non_iid_dataloader as niiddl

import os 
from einops import rearrange
from speachy.rescoring.tools import (
        sort_hypothesis_by_recording, 
        order_recordings_by_start_time,
        interpolate
)
import wandb

from speachy.lm.tools.loading import autoload
from speachy.rescoring.scripts.compute_rescore_wer import main as compute_rescore_wer

class argsclass:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

@torch.no_grad()
def get_text_probability(args, model, tokenizer, text, cached_states=None):
    device = torch.device(args.device)
    tokens = tokenizer.text_to_ids(text)
    token_prev = tokens
    tokens = torch.tensor(tokens).unsqueeze(0).long()
  

    add_bos = cached_states is None # problem
  
    token_lens = torch.tensor([tokens.shape[-1]]) + (1 if add_bos else 0)
    if tokens.shape[-1] == 0: # idk what to do here tbh
        return torch.tensor(torch.nan), cached_states, torch.tensor(2).float().log().item()
    #assert cached_states is None, 'FAK'
    tokens, token_lens = tokens.to(device), token_lens.to(device)
    tokens = lm_utils.add_bos(tokens, bos_token_id=0) if add_bos else tokens # don't add bos if we're starting from a cached state (bos is already there)
    targets = tokens.clone()
    targets[:, :-1] = tokens[:, 1:] # shift targets by 1 
    targets = targets[:, :-1] # remove last token (no target for last token)
    #targets = lm_utils.add_eos(targets, eos_id=0, token_lens=token_lens) don't use eos ):
    #targets = lm_utils.mark_padding(targets, lm_utils.token_lens_to_mask(token_lens), pad_id=-100) # probably uncecery since batch of 1 so nooo padding
    
    logits, _, cached_states = model(x=tokens, length=token_lens, cache=cached_states)
 

    # remove first and last token 
    toadd = 1 if add_bos else 0
    logits = logits[:, toadd:-1, :] # no target for last token
    targets = targets[:, toadd:] # 

    # remove eos/bos from probabilities
    logits = logits[:, :, 1:]
    # shift targets by 1 (no more eos/bos)
    targets -= 1

    # temperature
    logits = logits / args.temperature
    logprobs = torch.nn.functional.log_softmax(logits, dim=-1)

    logprobs = logprobs.squeeze(0).gather(1, targets.squeeze(0).unsqueeze(1)).squeeze()
    logprobslen = logprobs.shape[0] if len(logprobs.shape) > 0 else 1
    #logprobslen = 1 if logprobslen == 0 else logprobslen

    length_penalty = torch.tensor(logprobslen+1).float().log().item()

    logprobs = logprobs.sum() 
    
    return logprobs.to('cpu'), cached_states, length_penalty

def remove_multiple_spaces(text):
    return ' '.join(text.split())

def trim_history(history, max_len):
    if max_len == 0:
        return ''
    history = history.split()
    if len(history) > max_len:
        history = history[-max_len:]
    return ' '.join(history)

def trim_cache(kv_cache, max_len):
    if max_len == 0:
        return None
    if kv_cache is None:
        return None

    if max_len == -1:
        return kv_cache
    if kv_cache['cache_lengths'] > max_len:
        bos = kv_cache['cache'][:, :, :, :, 0, :].unsqueeze(-2).clone()
        kv_cache['cache'] = kv_cache['cache'][:, :, :, :, -max_len:, :]
        kv_cache['cache'] = torch.cat([bos, kv_cache['cache']], dim=-2)
        kv_cache['cache_lengths'] = torch.tensor([kv_cache['cache'].shape[-2]]).to(kv_cache['cache_lengths'].device)
    return kv_cache

def rescore(args, recording_hyps, standardise_stats): # disgusting code clean this up robert
    max_beam = args.stop_at_beam 

    for utt in tqdm(recording_hyps):
        best_log_p, best_hyp = float('-inf'), ''
        print(f'Target: {utt["targets"][0]}\n')
        for idx in utt['beams'][0].keys():
            cur = utt['beams'][0][idx]
            if idx >= max_beam:
                break
            if 'tlm_prob' not in cur:
                print('no tlm prob, skipping')
                continue

            hyptext = cur['text']
            AM_prob, NGRAM_prob = torch.tensor(cur['am_score']), torch.tensor(cur['ngram_score'])
            AM_prob = (AM_prob - standardise_stats['am_mean']) / standardise_stats['am_std']

    
            prob = torch.tensor(cur['tlm_prob'])
            prob = (prob - standardise_stats['tlm_mean']) / standardise_stats['tlm_std']
            length_penalty = torch.tensor(cur['length_penalty'])
            penalize_len = length_penalty * args.length_penalty
            prob = prob * args.tlm_scale - penalize_len
            NGRAM_prob = (NGRAM_prob - standardise_stats['ngram_mean']) / standardise_stats['ngram_std']
            NGRAM_prob = NGRAM_prob * args.ngram_scale

            cur['rescore_lp'] = (prob + AM_prob + NGRAM_prob).item()
   
            print(f'beam: {idx}, prob: {cur["rescore_lp"]}, hyp: {hyptext}\n')

            if cur['rescore_lp'] > best_log_p:
                best_log_p = cur['rescore_lp']
                best_hyp = hyptext

        target_txt = utt['targets'][0]
        top_hyp = utt['beams'][0][0]['text']
        original_wer = word_error_rate([target_txt], [top_hyp])
        rescored_wer = word_error_rate([target_txt], [best_hyp])
        print(f'{target_txt} : {top_hyp} : {best_hyp}')
        print(f'\n\nOriginal WER: {original_wer}, rescored WER: {rescored_wer}\n\n')
        if rescored_wer < original_wer:
            print(f'{["-"]*10} WER IMPROVEMENT {["-"]*10}\n\n')
        elif rescored_wer == original_wer:
            print('')
        else:
            print(f'{["-"]*10} WER DEGRADATION {["-"]*10}\n\n')

        utt['best_logp'] = best_log_p
        utt['best_hyp'] = best_hyp
        print(f'best logp: {best_log_p}')
     
    return recording_hyps
        
def compute_beam_ppls(args, model, tokenizer, recording_hyps): # disgusting code clean this up robert
    max_beam = args.stop_at_beam 
    max_history = args.max_history_len
    history_text = ''
    prev_end = None
    kv_cache = None
    kvs_to_cache = None
    for utt in tqdm(recording_hyps):
        kv_cache = {'cache': kvs_to_cache['cache'].clone(), 'cache_lengths': kvs_to_cache['cache_lengths'].clone()} if kvs_to_cache is not None else None # np
        segment_start, segment_end = utt['meta_data']['timings'].values()
        prev_end = segment_start if prev_end is None else prev_end
        
        kv_cache = None if prev_end - segment_start > args.max_utt_gap else kv_cache
        kv_cache = trim_cache(kv_cache, max_history)
        best_log_p, best_hyp = float('-inf'), ''

        for idx in utt['beams'][0].keys():
            cur = utt['beams'][0][idx]
            if idx >= max_beam:
                break
            if args.use_cached_scores and 'tlm_prob' not in cur:
                continue

            hyptext = cur['text']
            AM_prob, NGRAM_prob = torch.tensor(cur['am_score']), torch.tensor(cur['ngram_score'])
            prob, cache, length_penalty = get_text_probability(args, model, tokenizer, hyptext, cached_states=kv_cache)
            if idx == 0 and cache is not None:
                kvs_to_cache = {'cache': cache['cache'].clone(), 'cache_lengths': cache['cache_lengths'].clone()}
        
            prob = NGRAM_prob if prob.isnan() or prob == float('-inf') or prob == 0 else prob # just in case
            cur['tlm_prob'] = prob
            cur['length_penalty'] = length_penalty
    return recording_hyps


def compute_lm_ppls(args, model, tokenizer, hypothesis):
    for i, key in enumerate(hypothesis.keys()):
        recording = hypothesis[key]
        print(f'Computing perplexities for recording {key}, {i+1}/{len(hypothesis.keys())}')
        hypothesis[key] = compute_beam_ppls(args, model, tokenizer, recording)
    return hypothesis

def rescore_speakers(args, hypothesis, stardardise_stats):
    for i, key in enumerate(hypothesis.keys()):
        recording = hypothesis[key]
        print(f'rescoring for recording {key}, {i+1}/{len(hypothesis.keys())}')
        hypothesis[key] = rescore(args, recording, stardardise_stats)
    return hypothesis

def get_standardisation_stats(hypothesis):
    am_scores, tlm_scores, n_gram_scores = [], [], []
    for i, key in enumerate(hypothesis.keys()):
        recording = hypothesis[key]
        print(f'getting standardisation stats for recording {key}, {i+1}/{len(hypothesis.keys())}')
        for utt in tqdm(recording):
            cur = utt['beams'][0][0]
            if 'tlm_prob' not in cur:
                continue
            am_score = torch.tensor(cur['am_score'])
            tlm_score = torch.tensor(cur['tlm_prob'])
            n_gram_score = torch.tensor(cur['ngram_score'])
            am_scores.append(am_score)
            tlm_scores.append(tlm_score)
            n_gram_scores.append(n_gram_score)
    assert len(am_scores) == len(tlm_scores) == len(n_gram_scores) and len(am_scores) > 0
    am_scores = torch.stack(am_scores)
    tlm_scores = torch.stack(tlm_scores)
    n_gram_scores = torch.stack(n_gram_scores)
    am_mean, am_std = am_scores.mean(), am_scores.std()
    tlm_mean, tlm_std = tlm_scores.mean(), tlm_scores.std()
    n_gram_mean, n_gram_std = n_gram_scores.mean(), n_gram_scores.std()
    print(f'AM mean: {am_mean}, AM std: {am_std}')
    print(f'TLM mean: {tlm_mean}, TLM std: {tlm_std}')
    print(f'NGRAM mean: {n_gram_mean}, NGRAM std: {n_gram_std}')
    return {'am_mean': am_mean, 'am_std': am_std, 'tlm_mean': tlm_mean, 'tlm_std': tlm_std, 'ngram_mean': n_gram_mean, 'ngram_std': n_gram_std}


def main(args, hypothesis):
    if args.use_cached_scores == False:
        device = torch.device(args.device)
        config = lm_utils.load_config(args.config)
        tokenizer_path = os.path.join(config['model']['tokenizer']['dir'], 'tokenizer.model')
        tokenizer = tools.load_tokenizer(tokenizer_path)
        model = autoload(config=config, tokenizer=tokenizer)
        epoch, val_loss  = model_utils.load_checkpoint(args=argsclass(**{'checkpoint': args.checkpoint}), model=model, force_cpu=True)
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

    if args.use_cached_scores == False:
        hypothesis = compute_lm_ppls(args, model, tokenizer, hypothesis)
    stardardise_stats = get_standardisation_stats(hypothesis)
    #input('Press enter to continue')
    hypothesis = rescore_speakers(args, hypothesis, stardardise_stats)

    wer = compute_rescore_wer(hypothesis)
    if not args.no_wandb:
        wandb.log({'wer': wer})
    elif args.saveas != '':
        with open(args.saveas, 'wb') as f:
            pkl.dump(hypothesis, f)
    print(f'WER: {wer}')

    
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--hyppkl", type=str, default='./dev_rescored.pkl')
    parser.add_argument('--config', type=str, default='./experiment_configs/lm/decoder_pg19.yaml')
    parser.add_argument('--device', type=str, default='auto')
    #parser.add_argument('--tlm_threshold', help='if TLM logp is lower than this threshold TLM won\'t be interpolated', type=float, default=-20)
    parser.add_argument('--checkpoint', type=str, default='./checkpoints/pg19checkpoints_dropout10_nths/pg_19_ft_checkpoint_47_id_91.pt')
    parser.add_argument('--max_utt_gap', type=float, default=10.0)
    parser.add_argument('--saveas', type=str, default='')

    parser.add_argument('--length_penalty', type=float, default=0.0) 
    parser.add_argument('--stop_at_beam', type=int, default=25)
    parser.add_argument('--tlm_scale', type=float, default=8.0) # linearly scale TLM logp by this factor
    parser.add_argument('--ngram_scale', type=float, default=0.1) # linearly scale AM logp by this factor')
    parser.add_argument('--temperature', type=float, default=0.85) # softmax temperature for TLM (sharpness of distribution, will punish mistakes more)
    parser.add_argument('-use_cache','--use_cached_scores', action='store_true', help='whether to use cached scores from previous runs rather than recomputing them ')

    parser.add_argument('-history','--max_history_len', type=int, default=-1)

    parser.add_argument('--no_wandb', action='store_true')
    args = parser.parse_args()

    if args.device == 'auto':
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if args.checkpoint == '':
        print('No checkpoint specified...')
        ckpt = input('Please specify a checkpoint to evaluate: ')
        args.checkpoint = ckpt

    if not args.no_wandb:
        wandb.init()

    with open(args.hyppkl, 'rb') as f:
        hyps = pkl.load(f)

    main(args, hyps)