import argparse
import pickle as pkl
from Levenshtein import distance
from speachy.rescoring.tools import ( sort_hypothesis_by_recording, order_recordings_by_start_time, )
import numpy as np
from functools import reduce
import torch
import os

from speachy.utils.misc import ( add_common_args, get_parameters, load_pkl )

from speachy.utils.general import (
    load_config,
    load_checkpoint,
    load_tokenizer,
    save_checkpoint,
    save_schedular_data,
    write_to_log,
    draw_text,
)

from speachy.utils.general import load_checkpoint
from speachy.lm.tools.loading import autoload
from speachy.utils.helpers import  exists, isfalse, istrue
from speachy.utils.general.training_loop import optimizer, update_schedular

from torch.cuda.amp import GradScaler   

INTERPOLATE = 0.5

def get_edit_distance(hyps, target):
    return list(map(lambda x: distance(x, target) / len(target), hyps))

flatten_nested_list = lambda l: [item for sublist in l for item in sublist]
    
def tokenize_and_pad(utterances, tokenizer):
    tokenized = [tokenizer.text_to_ids(utt) for utt in utterances]
    max_len = max(map(len, tokenized))
    return np.array([utt + [0] * (max_len - len(utt)) for utt in tokenized])

def get_sub_batches(batch, tokenizer):
    proc_utts = lambda utts: tokenize_and_pad(flatten_nested_list(utts), tokenizer)
    sub_batches = []
    max_len = max(map(len, batch))
    for i in range(max_len):
        sub_batches.append({
            'utterances': [],
            'scores': [],
            'lengths': [],
        })
        for el in batch:
            if len(el) > i:
                sub_batches[-1]['utterances'].append(el[i][0])
                sub_batches[-1]['scores'].append(el[i][1])
                sub_batches[-1]['lengths'].append(len(el[i][0]))
            else:
                sub_batches[-1]['utterances'].append(-1)
                sub_batches[-1]['scores'].append(-1)
                sub_batches[-1]['lengths'].append(-1)
    non_empty_indices = np.arange(len(sub_batches[0]['lengths']))

    for i, sub_batch in enumerate(sub_batches):
        sb_utts, sb_scores, sb_lengths = np.array(sub_batch['utterances'], dtype=object), \
            np.array(sub_batch['scores'], dtype=object), np.array(sub_batch['lengths'], dtype=object)
        non_empty = non_empty_indices[sb_lengths != -1]
        # slice based on non empty of previous sub batch
        prev_fetch = None
        if i != 0:
            prev_lengths = sub_batches[i-1]['lengths']
            prev_non_empty = sub_batches[i-1]['non_empty']
            diff_from = ((prev_lengths != -1) == (sb_lengths != -1))[prev_non_empty]
            prev_fetch = np.arange(len(prev_non_empty))[diff_from]
        sub_batches[i] = {
            'utterances': sb_utts,
            'scores': sb_scores,
            'lengths': sb_lengths,
            'non_empty': non_empty,
            'prev_fetch': prev_fetch 
        }
    
    for i, sub_batch in enumerate(sub_batches):
        sub_batches[i] = {
            'utterances': proc_utts(sub_batch['utterances'][sub_batch['non_empty']].tolist()),
            'scores': sub_batch['scores'][sub_batch['non_empty']],
            'lengths': sub_batch['lengths'][sub_batch['non_empty']],
            'prev_fetch': sub_batch['prev_fetch'] # indices to fetch the states from previous sub batch
        }
    return sub_batches


    
def create_samples_from_recording(recording, num_utterances, num_negatives, max_gap=10.0, shuffle=True):
    samples = []
    prev_end = None
    if shuffle == False:
        np.random.seed(42) # deterministic selection of negatives
    for i, utterance in enumerate(recording):
        start_t, end_t = utterance['meta_data']['timings'].values()
        if prev_end is None or (start_t - prev_end) > max_gap:
            samples.append([]) 
        if len(samples[-1]) >= num_utterances:
            samples.append([])
        hyps = utterance['beams'][0]
        hyps = list(map(lambda x: x['text'], list(hyps.values())))
        target = utterance['targets'][0]
        hyps = list(filter(lambda el:el != target, hyps))
        hyps = np.random.choice(hyps, min(num_negatives, len(hyps)), replace=False).tolist()
        examples = [target] + hyps
        error_rates = get_edit_distance(examples, target)
        samples[-1].append((examples, error_rates))
        prev_end = end_t
        
    return samples

def create_dataset_samples(recordings, num_utterances, num_negatives, max_gap=10.0, shuffle=True):
    samples = []
    for recording in recordings.keys():
        samples += create_samples_from_recording(recordings[recording], num_utterances, num_negatives, max_gap, shuffle)
    if shuffle:
        np.random.shuffle(samples) # shuffle samples
    return samples


def sampler(samples, batch_size, tokenizer, shuffle=True):
    sample_indices = np.arange(len(samples))
    np.random.shuffle(sample_indices) if shuffle else None
    for i in range(0, len(samples), batch_size):
        yield get_sub_batches([samples[i] for i in sample_indices[i:i+batch_size]], tokenizer)
      


def train(loader, model):
    for batch in loader:
        for sub_batch in batch:
            pass

'''    train_hyp = load_pkl(args.train_hyp)
    dev_hyp = load_pkl(args.dev_hyp)
    train_data = order_recordings_by_start_time(sort_hypothesis_by_recording(train_hyp))
    dev_data = order_recordings_by_start_time(sort_hypothesis_by_recording(dev_hyp))
    train_samples = create_dataset_samples(train_data, num_utterances=args.utts_per_sample, num_negatives=args.negatives, max_gap=args.max_gap, shuffle=True)
    dev_samples = create_dataset_samples(dev_data, num_utterances=args.utts_per_sample, num_negatives=args.negatives, max_gap=args.max_gap, shuffle=False)'''

def main(args):
    device = torch.device(args.device)
    config = load_config(args.config)
    tokenizer_path = os.path.join(config['model']['tokenizer']['dir'], 'tokenizer.model')
    tokenizer = load_tokenizer(model_path=tokenizer_path)
    model = autoload(config=config, tokenizer=tokenizer)

    train_data = order_recordings_by_start_time(sort_hypothesis_by_recording(load_pkl(args.train_hyp)))
    dev_data = order_recordings_by_start_time(sort_hypothesis_by_recording(load_pkl(args.dev_hyp)))

    epoch_prev = 0
    if args.checkpoint != '':
        epoch_prev, val_loss = load_checkpoint(args=args, model=model, force_cpu=True)
        modeltype = config['model']['modeltype']
        print(f'Loaded model {args.checkpoint} with epoch {epoch_prev} and val_loss {val_loss}\n Model type: {modeltype}')
    model.to(device)

    optim, schedular = optimizer(model, args)
    total_params = get_parameters(model=model, verbose=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_hyp', type=str, required=True)
    parser.add_argument('--dev_hyp', type=str, required=True)
    parser.add_argument('-utts','--utts_per_sample', type=int, default=5)
    parser.add_argument('-negatives','--negatives', type=int, default=5)
    parser.add_argument('-batch','--batch_size', type=int, default=5)
    parser.add_argument('-mgap','--max_allowed_utterance_gap', type=float, default=10.0, help='max allowed gap between utterances in seconds')

    parser.add_argument('--config', type=str, default='./experiment_configs/lm/decoder_pg19.yaml')
    parser.add_argument('--checkpoint', type=str, default='')

    parser.add_argument('--schedular_data', type=str, default='./schedular_data.json')

    parser.add_argument('--max_utt_gap', type=float, default=10.0)

    parser.add_argument('-device', '--device', type=str, default='auto')
    parser.add_argument('--project_name', default='MWER-LM-FINETUNE_INTERSPEECH', type=str)
    parser = add_common_args(parser)

    args = parser.parse_args()

    args.device = 'cuda' if torch.cuda.is_available() else 'cpu' if args.device == 'auto' else args.device 