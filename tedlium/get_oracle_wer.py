from tqdm import tqdm
from nemo.collections.asr.metrics.wer import word_error_rate
import argparse 
import pickle as pkl

def sort_hypothesis_by_recording(hyps):
    recordings = {}
    for hyp in hyps:
        rec = hyp['meta_data']['recording_id']
        if rec not in recordings:
            recordings[rec] = []
        recordings[rec].append(hyp)
    return recordings

def order_recordings_by_start_time(hypothesis): # don't need to do this rlly
    for key in hypothesis.keys():
        hypothesis[key] = sorted(hypothesis[key], key=lambda x: x['meta_data']['timings']['segment_start'])
    return hypothesis

def get_oracle_wer(hypothesis, max_depth=-1):
    hyps, refs = [], []
    for key in hypothesis.keys():
        recording = hypothesis[key]
        for utt in tqdm(recording):
            best_idx, best_score = None, None
            target = utt['targets'][0]
            for idx in utt['beams'][0].keys():
                if max_depth >= 0 and idx >= max_depth:
                    break
                cur = utt['beams'][0][idx]
                hyptext = cur['text']
                hypwer = word_error_rate([hyptext], [target])
                if best_idx is None or hypwer < best_score:
                    best_idx = idx
                    best_score = hypwer
            hyps.append(utt['beams'][0][best_idx]['text'])
            refs.append(target)
    return word_error_rate(hyps, refs)


def main(hypothesis):
    hypothesis = sort_hypothesis_by_recording(hypothesis)
    hypothesis = order_recordings_by_start_time(hypothesis)
    wer = get_oracle_wer(hypothesis, max_depth=args.max_depth)
    print(f"Oracle WER: {wer}")
    return wer

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("-hyp","--hyp", type=str, required=True)
    args.add_argument("--max_depth", type=int, default=-1)
    args = args.parse_args()

    with open(args.hyp, 'rb') as f:
        hyps = pkl.load(f)

    main(hyps)
    
