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

from transformers import GPTNeoXForCausalLM, AutoTokenizer
from transformers import AutoTokenizer, GPT2LMHeadModel



from speachy.rescoring.scripts.compute_rescore_wer import main as compute_rescore_wer

class argsclass:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

@torch.no_grad()
def get_text_probability(args, model, tokenizer, text, history):
    device = torch.device(args.device)
    loss_fct = torch.nn.NLLLoss(ignore_index=-100, reduction='none')
    loss_log_softmax = torch.nn.LogSoftmax(dim=-1)
    if len(history) > 0:
        history = tokenizer.bos_token + history + tokenizer.eos_token
    else:
        text = tokenizer.bos_token + text + tokenizer.eos_token
    tokens, history_tokens = tokenizer(text, return_tensors='pt'), tokenizer(history, return_tensors='pt')
    token_prev = tokens
    history_len = len(history_tokens['input_ids'][0])
    if history_len > 0:
        tokens['input_ids'] = torch.cat((history_tokens['input_ids'].long(), tokens['input_ids'].long()), dim=1)
        tokens['attention_mask'] = torch.cat((history_tokens['attention_mask'].long(), tokens['attention_mask'].long()), dim=1)
    token_lens = torch.tensor([len(tokens['input_ids'][0])])
    tokens['input_ids'], tokens['attention_mask'], token_lens = tokens['input_ids'].to(device), tokens['attention_mask'].to(device), token_lens.to(device)
    outputs = model(**tokens)
    logits = outputs.logits
    labels = tokens['input_ids'].clone()
    if history_len > 0: ## check if this is correct
        labels = labels[:, history_len:]
        logits = logits[:, history_len - 1:, :]
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels.contiguous()
    else:
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
    shift_probs = loss_log_softmax(shift_logits)
    loss = loss_fct(shift_probs.view(-1, shift_probs.size(-1)), shift_labels.view(-1))
    loss = loss.view(shift_labels.size()).squeeze()
    #print(loss.shape, token_lens, tokens['input_ids'].shape, history_len)
    tx_prob = loss.sum().cpu()
    print(tx_prob)
    return tx_prob


def remove_multiple_spaces(text):
    return ' '.join(text.split())

def trim_history(history, max_len):
    if max_len == 0:
        return ''
    history = history.split()
    if len(history) > max_len:
        history = history[-max_len:]
    return ' '.join(history)


def compute_beam_ppls(args, model, tokenizer, recording_hyps):
    max_beam = args.stop_at_beam 
    max_history = args.max_history_len
    history_text = ''
    prev_end = None
    for utt in tqdm(recording_hyps):
        segment_start, segment_end = utt['meta_data']['timings'].values()
        prev_end = segment_start if prev_end is None else prev_end
        history_text = '' if prev_end - segment_start > args.max_utt_gap else history_text  # reset history if gap between utterances is too large
        history_text = trim_history(history_text, max_history)
        best_log_p, best_hyp = float('-inf'), ''
        print(f'History: {history_text}\n')
        print(f'Target: {utt["targets"][0]}\n')
        for idx in utt['beams'][0].keys():
            if idx >= max_beam:
                break
            cur = utt['beams'][0][idx]
            hyptext = cur['text']
            AM_prob = torch.tensor(cur['am_score']) * args.am_scale
            NGRAM_prob = torch.tensor(cur['ngram_score'])
            prob = get_text_probability(args, model, tokenizer, hyptext, history_text) * args.tlm_scale
       
            cur['tlm_prob'] = prob
            cur['rescore_lp'] = interpolate(
                am_score=AM_prob,
                ngram_score=NGRAM_prob,
                lm_score=prob,
                alpha=args.interpolation_weight
            )
            #print(f'AM prob: {AM_prob}, LM prob: {prob} (interpolated: {cur["rescore_lp"]})')
        
            print(f'beam: {idx}, prob: {cur["rescore_lp"]}, hyp: {hyptext}\n history len: {len(history_text.split())}')

            if cur['rescore_lp'] > best_log_p:
                best_log_p = cur['rescore_lp']
                best_hyp = hyptext

       
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
    for i, key in enumerate(hypothesis.keys()):
        recording = hypothesis[key]
        print(f'Computing perplexities for recording {key} - {i+1}/{len(hypothesis.keys())}')
        hypothesis[key] = compute_beam_ppls(args, model, tokenizer, recording)
    return hypothesis

def main(args, hypothesis):
    device = torch.device(args.device)
    '''
    revision="step143000"
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        revision=revision,
    )
    model = GPTNeoXForCausalLM.from_pretrained(
        args.model,
        revision=revision,
    )
    '''
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    print(f'Loaded model {args.model}')
    model.to(device)
    model.eval()

    hypothesis = sort_hypothesis_by_recording(hypothesis)
    hypothesis = order_recordings_by_start_time(hypothesis)

    hypothesis = compute_all_ppls(args, model, tokenizer, hypothesis)
    wer = compute_rescore_wer(hypothesis)
    if not args.no_wandb:
        wandb.log({'wer': wer})
    elif args.saveas != '':
        with open(args.saveas, 'wb') as f:
            pkl.dump(hypothesis, f)
    print(f'WER: {wer}')

    
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--hyppkl", type=str, default='tedlium_hyps.pkl')
    parser.add_argument('--device', type=str, default='auto')
    #parser.add_argument('--tlm_threshold', help='if TLM logp is lower than this threshold TLM won\'t be interpolated', type=float, default=-20)
    parser.add_argument('-model','--model', type=str, default='EleutherAI/pythia-1b')
    parser.add_argument('--max_utt_gap', type=float, default=10.0)
    parser.add_argument('--saveas', type=str, default='')

    parser.add_argument('--stop_at_beam', type=int, default=10)
    parser.add_argument('--tlm_scale', type=float, default=0.4) # linearly scale TLM logp by this factor
    parser.add_argument('--am_scale', type=float, default=1.0) # linearly scale AM logp by this factor')
    parser.add_argument('-alpha','--interpolation_weight', type=float, default=0.9) # interpolate TLM and NGRAM logp by this factor (alpha*tlm + (1-alpha)*ngram) 
    parser.add_argument('--temperature', type=float, default=1.0) # softmax temperature for TLM (sharpness of distribution, will punish mistakes more)


    parser.add_argument('--max_history_len', type=int, default=0)
    

    parser.add_argument('--no_wandb', action='store_true')
    args = parser.parse_args()

    if args.device == 'auto':
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

 

    if not args.no_wandb:
        wandb.init()

    with open(args.hyppkl, 'rb') as f:
        hyps = pkl.load(f)

    main(args, hyps)