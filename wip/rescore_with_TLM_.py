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
import lm_utils
import os 
from einops import rearrange
from speachy.rescoring.tools import (
        sort_hypothesis_by_recording, 
        order_recordings_by_start_time,
        interpolate
)
import wandb
import torch.nn.functional as F
from compute_rescore_wer import main as compute_rescore_wer

class argsclass:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

@torch.no_grad()
def get_text_probability(args, model, tokenizer, text, history, cached_kvs=None, first_token_prob=None):
    '''
    args: argsclass
    model: causal language model
    tokenizer: sentencepiece tokenizer
    history: no longer used :P
    cached_kvs: cached keys and values to speed up inference
    first_token_prob: if there is history this is the probability of the first token in the current sequence, taken from the previous sequence
    '''
    device = torch.device(args.device)
    tokens = tokenizer.text_to_ids(text)
    token_prev = tokens
    tokens = torch.tensor(tokens).unsqueeze(0).long()
  
    toremove = 1 if cached_kvs is None else 0

    tokens = tokens.to(device)
    tokens = lm_utils.add_bos(tokens, bos_token_id=0) if cached_kvs is None else tokens # add bos token if there is no history
    targets = tokens.clone()
    targets[:, :-1] = tokens[:, 1:]
  
    mask = None

    model_args = {'x': tokens, 'mask': mask}
    logits, cached_kvs = model(**model_args, cache_kvs=cached_kvs)
   
    # remove first and last token
    logits, next_token_logits = logits[:, :-1, :], logits[:, -1, :]
    targets = targets[:, toremove:]

    if first_token_prob is not None:
        logits = torch.cat([first_token_prob.unsqueeze(0), logits], dim=1)
  
    

    # remove eos/bos from probabilities
    logits = logits[:, :, 1:]
    # shift targets by 1 (no more eos/bos)
    targets -= 1

    # temperature
    logits = logits / args.temperature
    logprobs = torch.nn.functional.log_softmax(logits, dim=-1)

    # then take the log of the probability of the target
    #print(logprobs.argmax(dim=-1)[0,:20], targets[0,:20])  # DEBUG !!
    #print(tokenizer.ids_to_text(logprobs.argmax(dim=-1)[0,:50].tolist()), '- :SPAACE: -', tokenizer.ids_to_text(targets[0,:50].tolist()))
    logprobs = logprobs.squeeze(0).gather(1, targets.squeeze(0).unsqueeze(1))

    # get total logprob
    logprobs = logprobs.sum() 
    
    return logprobs.to('cpu'), cached_kvs, next_token_logits

def remove_multiple_spaces(text):
    return ' '.join(text.split())

def trim_history(cached_kv, max_len):
    if max_len == 0 or cached_kv is None:
        return cached_kv
    if cached_kv['keys'].shape[-2] > max_len:
        # trim but keep bos token at the beginning
        bos_k, bos_v = cached_kv['keys'][:, :, :, 0, :].unsqueeze(-2), cached_kv['values'][:, :, :, 0, :].unsqueeze(-2)
        cached_kv['keys'] = cached_kv['keys'][:, :, :, -max_len:, :]
        cached_kv['values'] = cached_kv['values'][:, :, :, -max_len:, :]
        print(bos_k.shape, cached_kv['keys'].shape)
        cached_kv['keys'] = torch.cat([bos_k, cached_kv['keys']], dim=-2)
        cached_kv['values'] = torch.cat([bos_v, cached_kv['values']], dim=-2)

    return cached_kv




def compute_beam_ppls(args, model, tokenizer, recording_hyps):
    max_beam = args.stop_at_beam 
    max_history = args.max_history_len
    history_text = ''
    prev_end = None
    cached_kvs = None
    next_token_prob = None
    for utt in tqdm(recording_hyps):
        segment_start, segment_end = utt['meta_data']['timings'].values()
        prev_end = segment_start if prev_end is None else prev_end
        history_text = '' if prev_end - segment_start > args.max_utt_gap else history_text  # reset history if gap between utterances is too large
        cached_kvs = None if history_text == '' else cached_kvs
        next_token_prob = None if history_text == '' else next_token_prob
        cached_kvs = trim_history(cached_kvs, max_history)
        best_log_p, best_hyp = float('-inf'), ''
        print(f'History: {history_text[-250:]}\n')
        print(f'Target: {utt["targets"][0]}\n')
        cached_kvs_tmp, next_token_prob_tmp = None, None
        for idx in utt['beams'][0].keys():
            if idx >= max_beam:
                break
            cur = utt['beams'][0][idx]
            hyptext = cur['text']
            AM_prob = torch.tensor(cur['am_score']) * args.am_scale
            NGRAM_prob = torch.tensor(cur['ngram_score'])

            if hyptext != '':
                prob, cached_kvs_l, next_token_prob_l = get_text_probability(args, model, tokenizer, hyptext, history_text, cached_kvs=cached_kvs, first_token_prob=next_token_prob)
            else:
                prob = F.log_softmax(next_token_prob_l, dim=-1)[0, 0].cpu()

            if idx == 0:
                assert hyptext != '', 'fix this'
                cached_kvs_tmp, next_token_prob_tmp = cached_kvs_l, next_token_prob_l
            prob *= args.tlm_scale

            if prob.isnan() or len(history_text.strip()) == 0:
                prob = AM_prob
            cur['tlm_prob'] = prob
            cur['rescore_lp'] = interpolate(
                am_score=prob,
                ngram_score=prob,
                lm_score=prob,
                alpha=args.interpolation_weight
            )
            #print(f'AM prob: {AM_prob}, LM prob: {prob} (interpolated: {cur["rescore_lp"]})')
            history_len = None if cached_kvs is None else cached_kvs['keys'].shape[-2]
            print(f'beam: {idx}, prob: {cur["rescore_lp"]}, hyp: {hyptext}\n history len: {history_len}\n')

            if cur['rescore_lp'] > best_log_p:
                best_log_p = cur['rescore_lp']
                best_hyp = hyptext
        cached_kvs, next_token_prob = cached_kvs_tmp, next_token_prob_tmp

       
        original_wer = word_error_rate([utt['targets'][0]], [utt['beams'][0][0]['text']])
        rescored_wer = word_error_rate([utt['targets'][0]], [best_hyp])
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
        #history_text += ' ' + best_hyp
        history_text += ' ' + utt['beams'][0][0]['text'] # use original hypothesis as history to prevent context drift
        history_text = remove_multiple_spaces(history_text)
        prev_end = segment_end
    return recording_hyps
        

def compute_all_ppls(args, model, tokenizer, hypothesis):
    for key in hypothesis.keys():
        recording = hypothesis[key]
        print(f'Computing perplexities for recording {key}')
        hypothesis[key] = compute_beam_ppls(args, model, tokenizer, recording)
    return hypothesis

def main(args, hypothesis):
    device = torch.device(args.device)
    config = lm_utils.load_config(args.config)
    tokenizer_path = os.path.join(config['model']['tokenizer']['dir'], 'tokenizer.model')
    tokenizer = tools.load_tokenizer(tokenizer_path)
    
    model = lm_utils.load_model(config, tokenizer, max_len=torch.inf)
    epoch, val_loss  = model_utils.load_checkpoint(args=argsclass(**{'checkpoint': args.checkpoint}), model=model, force_cpu=True)
    modeltype = config['model']['modeltype']
    print(f'Loaded model {args.checkpoint} with epoch {epoch} and val_loss {val_loss}\n Model type: {modeltype}')
    model.to(device)
    model.eval()

    hypothesis = sort_hypothesis_by_recording(hypothesis)
    hypothesis = order_recordings_by_start_time(hypothesis)

    hypothesis = compute_all_ppls(args, model, tokenizer, hypothesis)
    wer = compute_rescore_wer(hypothesis)
    if not args.no_wandb:
        wandb.log({'wer': wer})
    else:
        with open(args.saveas, 'wb') as f:
            pkl.dump(hypothesis, f)
    print(f'WER: {wer}')

    
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--hyppkl", type=str, default='tedlium_hyps.pkl')
    parser.add_argument('--config', type=str, default='./experiment_configs/lm/decoder_pg19.yaml')
    parser.add_argument('--device', type=str, default='auto')
    #parser.add_argument('--tlm_threshold', help='if TLM logp is lower than this threshold TLM won\'t be interpolated', type=float, default=-20)
    parser.add_argument('--checkpoint', type=str, default='./checkpoints/pg19checkpoints_dropout10_nths/pg_19_ft_checkpoint_47_id_91.pt')
    parser.add_argument('--max_utt_gap', type=float, default=10.0)
    parser.add_argument('--saveas', type=str, default='ppls.pkl')

    parser.add_argument('--stop_at_beam', type=int, default=3)
    parser.add_argument('--tlm_scale', type=float, default=0.4) # linearly scale TLM logp by this factor
    parser.add_argument('--am_scale', type=float, default=1.0) # linearly scale AM logp by this factor')
    parser.add_argument('-alpha','--interpolation_weight', type=float, default=0.6) # interpolate TLM and NGRAM logp by this factor (alpha*tlm + (1-alpha)*ngram) 
    parser.add_argument('--temperature', type=float, default=1.0) # softmax temperature for TLM (sharpness of distribution, will punish mistakes more)


    parser.add_argument('--max_history_len', type=int, default=350)
    

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