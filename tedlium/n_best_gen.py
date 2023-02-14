import argparse
import torch
from nemo.collections.asr.models.ctc_bpe_models import EncDecCTCModelBPE
from nemo.collections.asr.models.rnnt_bpe_models import EncDecRNNTBPEModel
import tools
import os
from tqdm import tqdm
import numpy as np

import wandb
import kenlm

import multiprocessing
import nemo

from nemo.collections.asr.metrics.wer import word_error_rate
from omegaconf.omegaconf import OmegaConf
from model_utils import load_checkpoint, load_nemo_checkpoint, load_model, load_sc_model, write_to_log

import non_iid_dataloader
import nemo.collections.asr as nemo_asr

import pickle as pkl

from tools import isfalse, istrue, exists, save_json
from nemo.collections.asr.metrics.wer import word_error_rate



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
        s_lprobs = log_probs.softmax(dim=-1).squeeze().cpu().numpy()
        outputs = {
            'meta_data': meta_data,
            'speaker_ids': speaker_ids,
            'targets': targets,
            'batch_num': batch_num,
            'probs': s_lprobs,
        }
        hyp_data.append(outputs)

    return hyp_data


def first_pass_decode(args, bpe_lm, vocab, hyp_data, beam_width, alpha, beta, ids_to_text_func):
    beamer = nemo_asr.modules.BeamSearchDecoderWithLM(
        vocab=vocab,
        beam_width=beam_width,
        alpha=alpha,
        beta=beta,
        lm_path=args.bpe_lm_path,
        input_tensor=False,
        num_cpus=max(os.cpu_count() // 2, 1),
    )
    probs = [el['probs'] for el in hyp_data]

    with nemo.core.typecheck.disable_checks():
        beams_batch = beamer.forward(log_probs=probs, log_probs_length=None,)
    print('Finished decoding')


    new_hyp_data = []
    for hyp, beam_set in zip(hyp_data, beams_batch):
        beams_out = {}
        for cand_idx, cand in enumerate(beam_set):
            enc = [el for el in cand[1]]
            first_pass_score = cand[0]
            enc_text = " ".join(enc)
            bpe_lm_score = bpe_lm.score(enc_text)
            first_pass_length_pen = beta * len(enc)
            am_score = -(bpe_lm_score*alpha - first_pass_score + first_pass_length_pen)
            pred_text = ids_to_text_func([ord(c) - 100 for c in enc])
            beams_out[cand_idx] = {
                'text': pred_text,
                'first_pass_score': cand[0],
                'am_score': am_score,
                'bpe_lm_score': bpe_lm_score,
                'first_pass_length_penalty': first_pass_length_pen,
            }
        new_hyp_data.append({
            'meta_data': hyp['meta_data'],
            'speaker_ids': hyp['speaker_ids'],
            'targets': hyp['targets'],
            'batch_num': hyp['batch_num'],
            'beams': [beams_out],
        })
    return new_hyp_data


def reorder_beams(hyp_data, key):
    '''reorders beams so that the best scoring beam is first using the value at key'''
    for hyp in hyp_data:
        hyp['beams'] = [{i:el[1] for i, el in enumerate(sorted(hyp['beams'][0].items(), key=lambda x: x[1][key], reverse=True))}]

    return hyp_data

def score_text_ngram(ngram, text, unk_offset):
    ngram_lm_score = list(ngram.full_scores(text, bos=False, eos=False))
    non_oov_score = sum([el[0] for el in ngram_lm_score if el[-1] == False])
    oov_score = sum([1 for el in ngram_lm_score if el[-1] == True])
    return non_oov_score, oov_score


def second_pass_rescore(args, ngram_lm, hyp_data, fp_alpha, sp_alpha, beta, unk_offset):
    for hypothesis in hyp_data:
        beams = hypothesis['beams'][0]
        for beam_idx in beams.keys():
            beam = beams[beam_idx]
            text = beam['text']
            text_len = len(text.split())
            length_pen = beta * text_len if not args.log_beta else beta * np.log(text_len)
            if 'ngram_lm_score_non_oov' not in beam or 'ngram_oov_count' not in beam:
                ngram_lm_score_non_oov, oov_score = score_text_ngram(ngram_lm, text, unk_offset)
                beam['ngram_lm_score_non_oov'] = ngram_lm_score_non_oov
                beam['ngram_oov_count'] = oov_score
            else:
                ngram_lm_score_non_oov, oov_score = beam['ngram_lm_score_non_oov'], beam['ngram_oov_count']
            ngram_lm_score = ngram_lm_score_non_oov + oov_score * unk_offset
            beam['ngram_lm_score'] = ngram_lm_score
            am_score = beam['am_score']
            fpass_lm_score = beam['bpe_lm_score']
            fpass_length_pen = beam['first_pass_length_penalty']
            beam['second_pass_length_penalty'] = length_pen
            beam['second_pass_score'] = am_score + sp_alpha * ngram_lm_score + length_pen + fp_alpha * fpass_lm_score + fpass_length_pen
    return reorder_beams(hyp_data, 'second_pass_score')
            

def eval_beams(hyp_data):
    hyps, refs = [], []
    for hyp in hyp_data:
        hyps.append(hyp['beams'][0][0]['text'])
        refs.append(hyp['targets'][0])
    wer = word_error_rate(hyps, refs)
    return wer

def load_pickle(path):
    with open(path, 'rb') as f:
        pkl_data = pkl.load(f)
    return pkl_data['stage'], pkl_data['data']

def save_pickle(path, obj, stage='logits'):
    with open(path, 'wb') as f:
        pkl.dump({'stage':stage, 'data':obj}, f) if stage != 'finished' else pkl.dump(obj, f)

def delete_pickle(path):
    os.remove(path)

def search_first_pass_alpha_beta_search(args, bpe_lm, vocab, hyp_data, ids_to_text_func, beam_width):
    results = []
    alpha_set = [0.9, 1.0, 1.1, 1.2, 1.3, 1.4]
    beta_set = [0.1, 0.2, 0.3, 0.4, 0.5]
    total = len(alpha_set) * len(beta_set)
    current_step = 0
    for alpha in alpha_set:
        for beta in beta_set:
            current_step += 1
            print(f'[{current_step}/{total}]')  
            print(f'alpha: {alpha}, beta: {beta}')
            new_hyp_data = first_pass_decode(args, bpe_lm, vocab, hyp_data, beam_width, alpha, beta, ids_to_text_func)
            wer = eval_beams(new_hyp_data)
            print(f'wer: {wer}')
            results.append({
                'alpha': alpha,
                'beta': beta,
                'wer': wer,
            })
    # get best
    best = sorted(results, key=lambda x: x['wer'])[0]
    print(f'Best alpha: {best["alpha"]}, beta: {best["beta"]}, wer: {best["wer"]}')
    return alpha, beta

def second_pass_alpha_beta_seach(args, ngram_lm, hyp_data):
    results = []
    unk_offsets = np.linspace(np.log10(np.exp(-10)), np.log10(np.exp(-10))*4, 6).tolist()
    for sp_alpha in [0.3, 0.4, 0.5, 0.6]:
        for beta in [0.0, 0.5, 0.6, 0.7]:
            for fp_alpha in [0.8, 0.9, 1.0, 1.1, 1.2]:
                for unk_offset in [-5.5, -8, -10, -14, -16, -18, -20]:
                    print(f'fp_alpha: {fp_alpha}, beta: {beta}, sp_alpha: {sp_alpha}, unk_offset: {unk_offset}')
                    hyp_data = second_pass_rescore(args, ngram_lm, hyp_data, fp_alpha, sp_alpha, beta, unk_offset)
        
                    wer = eval_beams(hyp_data=hyp_data)
                    print(f'wer: {wer}')
                    results.append({
                        'fp_alpha': fp_alpha,
                        'sp_alpha': sp_alpha,
                        'unk_offset': unk_offset,
                        'beta': beta,
                        'wer': wer,
                    })
    # get best
    best = sorted(results, key=lambda x: x['wer'])[0]
    print(f'Best fp_alpha: {best["fp_alpha"]}, beta: {best["beta"]}, sp_alpha: {best["sp_alpha"]}, unk_offset: {best["unk_offset"]}, wer: {best["wer"]}')
    final_hyp_data = second_pass_rescore(args, ngram_lm, hyp_data, best['fp_alpha'], best['sp_alpha'], best['beta'], best['unk_offset'])
    return best['fp_alpha'], best['sp_alpha'], best['beta'], best['unk_offset'], best['wer'], final_hyp_data

def main(args):
    model = load_model(args) if args.self_conditioned == False else load_sc_model(args)
    if args.checkpoint != '':
        load_checkpoint(args, model)
    print('\nTrainable parameters:'+str(sum(p.numel() for p in model.parameters() if p.requires_grad)))
    print(f'Total parameters: {sum(p.numel() for p in model.parameters())}\n')
    corpus_dict = tools.load_corpus()
    tokenizer = tools.load_tokenizer(args.tokenizer_model)
    ids_to_text_func = tokenizer.ids_to_text
    bpe_lm = kenlm.Model(args.bpe_lm_path)
    ngram_lm = kenlm.Model(args.language_model)
    vocab = [chr(idx + 100) for idx in range(len(tokenizer.vocab))]

    
    temp_name_dev = f'dev_{args.load_tmp}' 
    temp_name_test = f'test_{args.load_tmp}'
    dev_stage, test_stage = None, None # stage corresponds to the last step of the pipeline that was completed

    if os.path.exists(os.path.join(args.tmp_dir, temp_name_dev)):
        dev_stage, dev_hyps = load_pickle(os.path.join(args.tmp_dir, temp_name_dev))
    
    
    if dev_stage == None: 
        print(f'Fetching logits for dev set...')
        dev_hyps = get_logits(args, model, corpus_dict['dev'])
        save_pickle(os.path.join(args.tmp_dir, temp_name_dev), dev_hyps, stage='logits')
        dev_stage = 'logits'

    print(f'performing grid search for pass decoding...')
    fp_alpha, fp_beta = search_first_pass_alpha_beta_search(args, bpe_lm, vocab, dev_hyps, ids_to_text_func, 15)
    if args.rundev:
        dev_hyps = first_pass_decode(args, bpe_lm=bpe_lm, vocab=vocab, hyp_data=dev_hyps, beam_width=args.beam_size, alpha=fp_alpha, beta=fp_beta, ids_to_text_func=ids_to_text_func)
        fpass_wer_dev = eval_beams(dev_hyps)
        print(f'First pass dev WER: {fpass_wer_dev}')
        #save_pickle(os.path.join(args.tmp_dir, temp_name_dev), dev_hyps, stage='first_pass')
        #dev_stage = 'first_pass'

        #assert dev_stage == 'first_pass', 'dev stage should be first pass at this point - something went wrong - please delete dev tmp file and try again (sorry)'
        print(f'Performing grid search for second pass decoding...')
        fp_alpha_weight, sp_alpha, sp_beta, unk_offset, dev_wer, dev_hyps = second_pass_alpha_beta_seach(args, ngram_lm, dev_hyps)
        save_pickle(os.path.join(args.tmp_dir, temp_name_dev), dev_hyps, stage='finished')
        print(f'--- Finished dev set -------')
        del dev_hyps # free up memory
    else:
        if os.path.exists(os.path.join(args.tmp_dir, temp_name_test)):
            test_stage, test_hyps = load_pickle(os.path.join(args.tmp_dir, temp_name_test))

        if test_stage == None:
            print(f'Fetching logits for test set...')
            test_hyps = get_logits(args, model, corpus_dict['test'])
            save_pickle(os.path.join(args.tmp_dir, temp_name_test), test_hyps, stage='logits')
            test_stage = 'logits'


        test_hyps = first_pass_decode(args, bpe_lm=bpe_lm, vocab=vocab, hyp_data=test_hyps, beam_width=args.beam_size, alpha=fp_alpha, beta=fp_beta, ids_to_text_func=ids_to_text_func)
        fpass_wer_test = eval_beams(test_hyps)
        print(f'First pass dev WER: {fpass_wer_test}')

        fp_alpha, sp_alpha, beta, unk_offset = args.fp_alpha, args.sp_alpha, args.beta, args.unk_offset
        test_hyps = second_pass_rescore(args, ngram_lm, test_hyps, fp_alpha, sp_alpha, beta, unk_offset)
        test_wer = eval_beams(hyp_data=test_hyps)
        save_pickle(os.path.join(args.tmp_dir, temp_name_test), test_hyps, stage='finished')
        print('FINISHED')
        print(f'first pass alpha: {fp_alpha}, beta: {fp_beta}, second pass alpha: {sp_alpha}, beta: {beta}')
        print(f'Test WER: {test_wer}')
        print('GOODBYE')
    
    


#Best fp_alpha: 0.9, beta: 0.7, sp_alpha: 0.6, unk_offset: -18, wer: 0.09889438302127895 (beam 1000)




if __name__ == '__main__':
    ''''
    Note I've only written this for a batch size of 1 (lazy)
    '''
    parser = argparse.ArgumentParser() 

    parser.add_argument('-rundev', '--rundev', action='store_true', help='whether to run on dev set')
    parser.add_argument('-load_tmp', '--load_tmp', default='tmp.pkl', type=str, help='base name of logit hyp to load (full name = split+_+name')
    parser.add_argument('-tmp_dir','--tmp_dir', type=str, default='./tmp', help='path to tmp dir')

    parser.add_argument('-log_beta', '--log_beta', action='store_true', help='whether to use log scale for beta length penalty')

    parser.add_argument('--load_pretrained', action='store_true')
    parser.add_argument('--pretrained', type=str, default='stt_en_conformer_ctc_small') # stt_en_conformer_ctc_large stt_en_conformer_transducer_large
    parser.add_argument('--model_config', type=str, default='../model_configs/conformer_sc_ctc_bpe_small.yaml') 

    parser.add_argument('--tokenizer_model', type=str, default='./tokenizers/tokenizer_spe_bpe_v128/tokenizer.model', help='path to tokenizer model')
    parser.add_argument('--max_duration', type=float, default=0, help='max duration of audio in seconds')

    parser.add_argument('-sp_alpha', '--sp_alpha', type=float, default=0.6, help='second pass alpha')
    parser.add_argument('-fp_alpha', '--fp_alpha', type=float, default=0.9, help='first pass alpha')
    parser.add_argument('-beta', '--beta', type=float, default=0.7, help='second pass beta')
    parser.add_argument('-unk_offset', '--unk_offset', type=float, default=-18, help='second pass unk offset')

    parser.add_argument('--log_file', type=str, default='eval_log.txt')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints')
    parser.add_argument('--checkpoint', type=str, default='checkpoint_68_id_15.pt')
    

    parser.add_argument('--beam_size', type=int, default=1000)
    parser.add_argument('--bpe_lm_path', type=str, default='./ngrams/binary_bpe/kenlmbpe6.lm.bin')

    parser.add_argument('-lm', '--language_model', type=str, default='./ngrams/cantab_interp_tedlium.arpa', help='arpa n-gram model for decoding')#./ngrams/3gram-6mix.arpa
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument('--alpha', type=float, default=0.5)

    parser.add_argument('-token_skip', '--token_min_logp', default=-5, type=float)
    parser.add_argument('-beam_prune', '--beam_prune_logp', default=-10000, type=float)

    parser.add_argument('-nsc','--not_self_conditioned', action='store_true', help='use for non self-conditioned models')

    parser.add_argument('-mgap','--max_allowed_utterance_gap', type=float, default=3.0, help='max allowed gap between utterances in seconds')


    parser.add_argument('-gap','--gap', default=0.1, type=float, help='gap between utterances when concatenating')

    parser.add_argument('--single_speaker_with_gaps', action='store_true', help='if set, utterances will contain 1 speaker and additional gaps of speaker_gap will be added if there is a speaker change between two utternces of the same speaker')
    parser.add_argument('--speaker_gap', type=float, default=1.0, help='for use with single_speaker_with_gaps, will add this many seconds of silence between utterances of the same speaker when there is a speaker change in between them')

    parser.add_argument('--split_speakers', action='store_true', help='if set, wont concat samples from different speakers, (concat_samples must be enabled)')

    parser.add_argument('-psl','--pass_segment_lengths', action='store_true', help='if set, will pass segment lens to the model, used with concat_samples for multi segment models')
    parser.add_argument('-save','--save_outputs', default='', type=str, help='save outputs to file')
    parser.add_argument('-dropout', '--dropout_rate', help='dropout at inference', default=0.0, type=float)

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
