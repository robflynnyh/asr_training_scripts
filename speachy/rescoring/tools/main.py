from tqdm import tqdm
from nemo.collections.asr.metrics.wer import word_error_rate
import argparse 
import torch
import pickle as pkl

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

'''def interpolate(am_score, lm_score, alpha):
    #
    am_score: Acoustic model score (log probability)
    lm_score: Language model score (log probability)
    alpha: Interpolation weight
    we compute this in log space
    #
    log_alpha = torch.log(torch.tensor(alpha))
    log_1_minus_alpha = torch.log(torch.tensor(1-alpha))
    interped = torch.logaddexp(am_score + log_1_minus_alpha, lm_score + log_alpha)
    #print(f'am_score: {am_score}, lm_score: {lm_score}, alpha: {alpha}, interped: {interped}')
    return interped'''

def interpolate(am_score, lm_score, alpha):
    '''
    am_score: Acoustic model score (log probability)
    lm_score: Language model score (log probability)
    alpha: Interpolation weight
  
    '''
    interped = am_score + alpha * lm_score
    return interped