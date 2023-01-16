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

from speachy.rescoring.scripts.compute_rescore_wer import main as compute_rescore_wer

class argsclass:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

@torch.no_grad()
def get_text_probability(args, model, tokenizer, text, history):
    device = torch.device(args.device)
    tokens, history_tokens = tokenizer.text_to_ids(text), tokenizer.text_to_ids(history)
    token_prev = tokens
    tokens, history_tokens = torch.tensor(tokens).unsqueeze(0).long(), torch.tensor(history_tokens).unsqueeze(0).long()
  
    history_len = len(history_tokens[0])
    if history_len > 0:
        tokens = torch.cat((history_tokens, tokens), dim=1)

  
    token_lens = torch.tensor([len(tokens[0])]) + 1 # add 1 for the <bos> token
    tokens, token_lens = tokens.to(device), token_lens.to(device)
    tokens = lm_utils.add_bos(tokens, bos_token_id=0)
    targets = tokens.clone()
    targets[:, :-1] = tokens[:, 1:]
    targets = lm_utils.add_eos(targets, eos_id=0, token_lens=token_lens)
    mask = lm_utils.token_lens_to_mask(token_lens)
    targets = lm_utils.mark_padding(targets, mask, pad_id=-100)

    model_args = {'x': tokens, 'mask': mask} if isfalse(callable(getattr(model, 'get_args', False))) \
        else model.get_args(tokens=tokens, mask=mask, lengths=token_lens)

    logits = model(**model_args)
   
    # remove first and last token
    logits = logits[:, history_len + 1:-1, :]
    targets = targets[:, history_len + 1:-1]

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

    #length_penalty = logprobs.shape[0] * (1*args.length_penalty)
    #print('length_penalty', length_penalty)
    # get total logprob
    logprobs = logprobs.sum() #+ length_penalty
    
    return logprobs.to('cpu')

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
            if prob.isnan():
                prob = AM_prob
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

    parser.add_argument('--length_penalty', type=float, default=0.3) # not used
    parser.add_argument('--stop_at_beam', type=int, default=25)
    parser.add_argument('--tlm_scale', type=float, default=0.2) # linearly scale TLM logp by this factor
    parser.add_argument('--am_scale', type=float, default=0.4) # linearly scale AM logp by this factor')
    parser.add_argument('-alpha','--interpolation_weight', type=float, default=0.85) # interpolate TLM and NGRAM logp by this factor (alpha*tlm + (1-alpha)*ngram) 
    parser.add_argument('--temperature', type=float, default=0.85) # softmax temperature for TLM (sharpness of distribution, will punish mistakes more)


    parser.add_argument('-history','--max_history_len', type=int, default=800)
    

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