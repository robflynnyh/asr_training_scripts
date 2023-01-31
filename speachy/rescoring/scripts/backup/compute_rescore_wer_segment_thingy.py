from tqdm import tqdm
from nemo.collections.asr.metrics.wer import word_error_rate
import argparse 
import pickle as pkl
from speachy.utils.helpers import exists

def sort_hypothesis_by_recording(hyps):
    recordings = {}
    for hyp in hyps:
        rec = hyp['meta_data']['recording_id']
        if rec not in recordings:
            recordings[rec] = []
        recordings[rec].append(hyp)
    return recordings



def order_recordings_by_start_time(hypothesis):
    for key in hypothesis.keys():
        hypothesis[key] = sorted(hypothesis[key], key=lambda x: x['meta_data']['timings']['segment_start'])
    return hypothesis


def get_rescore_wer(hypothesis):
    hyps, refs = [], []
    first_p_hyps = []
    for key in hypothesis.keys():
        recording = hypothesis[key]
        best_hyp_text = ''
        first_p_hyp_text = ''
        for utt in tqdm(recording):
            first_p_hyp = utt['beams'][0][0]['text']
            best_hyp = utt['best_hyp']
            first_p_hyp_text += ' ' + first_p_hyp.strip()
            best_hyp_text += ' ' + best_hyp.strip()
            if exists(utt['targets']):
                target = utt['targets'][0]
                hyps.append(best_hyp_text.strip())
                refs.append(target)
                first_p_hyps.append(first_p_hyp_text.strip())
                best_hyp_text, first_p_hyp_text = '', ''
    return word_error_rate(hyps, refs), word_error_rate(first_p_hyps, refs)


def main(hypothesis):
    if hypothesis.__class__.__name__ != 'dict':
        hypothesis = sort_hypothesis_by_recording(hypothesis)
        hypothesis = order_recordings_by_start_time(hypothesis)
        
    wer, prev_wer = get_rescore_wer(hypothesis)
    print(f"Rescored WER: {wer}, First Pass WER: {prev_wer}")
    return wer

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--hypf", type=str, required=True)
    args = args.parse_args()

    with open(args.hypf, 'rb') as f:
        hyps = pkl.load(f)

    main(hyps)
    
