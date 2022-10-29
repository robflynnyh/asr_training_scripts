from collections import OrderedDict
from lhotse import RecordingSet, Recording, AudioSource, SupervisionSegment, SupervisionSet
from typing import List, Dict, Tuple, Union, Any
from lhotse import CutSet
import os
from nemo.collections import nlp as nemo_nlp
import torch
import numpy as np
from lhotse.dataset.collation import collate_audio
from lhotse.dataset.sampling.dynamic_bucketing import DynamicBucketingSampler
from lhotse.dataset.sampling.simple import SimpleCutSampler
from tqdm import tqdm
import json
import datetime
from os.path import join
import subprocess
import re

AMI_DEFAULT = "/store/store1/data/ami/"
SCLITE_PATH = '/exp/exp1/acp21rjf/SCTK/bin/sclite'

RANDOM_WORDS = ['Swimming', 'Popcorn', 'Dinosoar', 'Rectangle', 'WuhWuh', 'Handle', 'Infiltration', 'Spring', 'Bee', 'Boop', 'Beep','Boat', 'Bicycle', 'Car', 'Cat', 'Dog', 'Elephant', 'Fish', 'Giraffe', 'Trigger','Horse', 'Lion', 'Star','Monkey', 'Pig', 'Bond', 'Rabbit', 'Dime', 'Protect', 'Sheep', 'Tiger', 'Train', 'Truck', 'Brain','Whale', 'Zebra', 'Studio', 'Dough', 'Probably', 'Horizantal', 'Tough', 'Huge', 'Tiny', 'Diseased', 'Knees', 'Clown', 'Blough', 'Woop','Skrrt', 'Skrrt', 'High', 'Low', 'Blow', 'Preaching', 'Street', 'Crazy', 'Hazy', 'Lazy', 'Striking', 'Dragon', 'Boom', 'Abdomen', 'Chips', 'Nation', 'Lord', 'Drop', 'HmmHmm', 'Lava', 'Rhymes']

# helper funcs
def isfalse(val:Any) -> bool:
    return val == False

def istrue(val:Any) -> bool:
    return val == True

def exists(val:Any) -> bool:
    return val is not None
    
def save_json(obj:Dict, path:str):
    with open(path, 'w') as f:
        json.dump(obj, f)

def load_json(path:str) -> Dict:
    with open(path, 'r') as f:
        return json.load(f)

def run_cmd(cmd:str):
    print(f'Running {cmd}')
    subprocess.run(cmd, shell=True, check=True)

def get_date():
    return str(datetime.datetime.now()).replace(' ', '_').replace(':', '-').replace('.', '-')

def read_text(filename:str) -> List[str]:
    with open(filename, 'r') as f:
        return f.read().split('\n')

def check_exists(path:str):
    assert os.path.exists(path), f'{path} does not exist'

def unpack_nested(nested:List[List[Any]]) -> List[Any]:
    return [item for sublist in nested for item in sublist]

def remove_multiple_spaces(text:str) -> str:
    return re.sub(' +', ' ', text)

def ifexists_call(var:Any, fn:callable, *args, **kwargs) -> Any:
    return fn(var, *args, **kwargs) if exists(var) else None

def iftrue_call(var:Any, fn:callable, *args, **kwargs) -> Any:
    return fn(*args, **kwargs) if var else None

def write_trn_files(refs:List[str], hyps:List[str], speakers:List[str]=[], encoded_lens:List[int]=[], fname:str='date', out_dir:str='./'):
    print(f'Writing trn files to {out_dir}')
    assert len(refs) == len(hyps), 'refs and hyps must be the same length'
    if len(speakers) != len(refs):
        speakers = ['any'] * len(refs)
        print('Speaker not provided or not the same length as refs and hyps. Using "any" for all.')
    if len(encoded_lens) != len(refs):
        encoded_lens = [-1] * len(refs)
        print('Encoded lens not provided or not the same length as refs and hyps. Using -1 for all.')

    if fname == 'date':
        fname = get_date()
        print(f'No fname provided. Using {fname} (date) for fname.')
    fname = fname if fname.endswith('.trn') else fname + '.trn'
    
    refname = join(out_dir, 'ref_' + fname)
    hypname = join(out_dir, 'hyp_' + fname)
    print(f'Writing {refname} and {hypname}')
    for i, (ref, hyp, speaker, encoded_len) in enumerate(zip(refs, hyps, speakers, encoded_lens)):
        with open(refname, 'a') as f:
            f.write(f';;len: {encoded_len}\n{ref} ({speaker}_{i})\n')
        with open(hypname, 'a') as f:
            f.write(f';;len: {encoded_len}\n{hyp} ({speaker}_{i})\n')
    print('All Done')
    return refname, hypname


def eval_with_sclite(ref, hyp, mode='dtl all'):
    list(map(check_exists, [ref, hyp]))
    cmd = f'{SCLITE_PATH} -r {ref} -h {hyp} -i rm -o {mode} stdout > {hyp}.out'
    run_cmd(cmd)
    outf = read_text(f'{hyp}.out')
    wer = [el for el in outf if 'Percent Total Error' in el][0]
    print(f'Saved output to {hyp}.out')
    return wer


def random_word_generator(num_words:int) -> str:
    ''' (use for run ids)
    Generates a random word from the list of random words
    num_words: number of words to generate
    '''
    words = []
    for i in range(num_words):
        words.append(RANDOM_WORDS[np.random.randint(0, len(RANDOM_WORDS))])

    print(len(RANDOM_WORDS))
    return ''.join(words)

def get_OOV_words(test:str, train:str):
    '''
    Returns percent of OOV words in test set
    test: test set path
    train: train set path
    '''
    with open(test) as f:
        test_lines = f.read().splitlines()
    with open(train) as f:
        train_lines = f.read().splitlines()
    test_words = []
    for line in test_lines:
        test_words.extend(line.split(' '))
    train_words = []
    for line in train_lines:
        train_words.extend(line.split(' '))
    # filter out empty strings
    test_words = [word for word in test_words if word != '' or word != '\n']
    test_words_set = set(test_words)
    train_words_ser = set(train_words)
    oov_words_unqiue = test_words_set - train_words_ser
    print(f'num unique test words: {len(test_words_set)}')
    print(f'num unique train words: {len(train_words_ser)}')
    print(f'num unique oov words: {len(oov_words_unqiue)}')

    oov_words = [word for word in test_words if word in oov_words_unqiue]
    print(f'num oov words: {len(oov_words)}')
    print(f'percent of oov words: {len(oov_words)/len(test_words):.2%}')

    

def list_checkpoint_val_losses(checkpoint_dir:str, verbose:bool=True, return_data:bool=False) -> Dict[str, float]:
    checkpoints = {}
    for file in os.listdir(checkpoint_dir):
        if file.endswith('.pt'):
            checkpoints[file] = None
    for file in checkpoints.keys():
        path_ = os.path.join(checkpoint_dir, file)
        checkpoint = torch.load(path_, map_location='cpu')
        checkpoints[file] = checkpoint
        if verbose:
            print(f'{file}: {checkpoints[file]["val_loss"]}')
    print('\n') if verbose else None
    if return_data:
        return checkpoints

def merge_top_checkpoints(checkpoint_dir:str, top_n:int, target:str):
    '''
    Merges the top n checkpoints in a directory into a single checkpoint
    checkpoint_dir: directory containing checkpoints
    top_n: number of checkpoints to merge
    target: path to save (including filename)
    '''
    checkpoints = list_checkpoint_val_losses(checkpoint_dir, verbose=False, return_data=True)
    checkpoints = sorted(checkpoints.items(), key=lambda x: x[1]['val_loss'])

    checkpoints = checkpoints[:top_n]

    checkpoint_weights = [checkpoints[i][1]['model_state_dict'] for i in range(top_n)]
    
    new_model_weights = OrderedDict() 
    for key in checkpoint_weights[0].keys():
        new_model_weights[key] = None
        for i in range(top_n):
            weights_to_add = checkpoint_weights[i][key] / top_n
            if new_model_weights[key] is None:
                new_model_weights[key] = weights_to_add
            else:
                new_model_weights[key] += weights_to_add
    torch.save({'model_state_dict': new_model_weights}, target)



def load_corpus(target_folder:str=AMI_DEFAULT+"ami_dataset", prefix_path=AMI_DEFAULT, mic='ihm') -> Dict[str, CutSet]:
    ami_ds = {}
    for split in ['train', 'dev', 'test']:
        cuts = CutSet.from_file(os.path.join(target_folder, f'ami-{mic}_cuts_{split}.jsonl.gz'))
        ami_ds[split] = cuts.with_recording_path_prefix(prefix_path)
    return ami_ds

def get_corpus_duration(split:CutSet):
    '''Returns the total duration of the corpus in hours (duh)
       split: lhotse cutset split from load_corpus
    '''
    dur = 0
    for entry in tqdm(split):
        dur += entry.supervisions[0].duration
    print(f'Corpus duration: {dur/60/60:.2f} hours')

def load_tokenizer(model_path:str="tokenizer_spe_bpe_v128/tokenizer.model"):
    tokenizer_spe = nemo_nlp.modules.get_tokenizer(tokenizer_name="sentencepiece", tokenizer_model=model_path)
    return tokenizer_spe

def convert_lhotse_to_manifest(split:CutSet, target:str):
    '''
    Converts a lhotse cutset to a nvidia nemo manifest file
    split: lhotse cutset split from load_corpus
    target: path to save (including filename)
    '''
    manifest = []
    for entry in tqdm(split):
        manifest.append({
            'text': entry.supervisions[0].text,
            'audio_path': entry.recording.sources[0].source,
            'duration': entry.duration
        })
    with open(target, 'w') as f:
        for line in manifest:
            f.write(json.dumps(line) + '\n')
    print(f'Saved manifest to {target}')

class TokenizerCollator:
    def __init__(
        self,
        tokenizer,
        pad_id=0,
    ):
        self.pad_id = pad_id
       
        self.tokenizer = tokenizer
        self.token2idx = self.tokenizer.text_to_ids
        self.idx2token = self.tokenizer.ids_to_text

    def __call__(self, cuts: CutSet) -> Tuple[torch.Tensor, torch.Tensor]:
        token_sequences = [
            " ".join(supervision.text for supervision in cut.supervisions) ##
            for cut in cuts
        ]
        max_len = max(len(token_sequence) for token_sequence in token_sequences) + 1 

        seqs = []
        unpadded_lens = []
        for sequence in token_sequences:
            seq = self.tokenizer.text_to_ids(sequence)
            seqs.append(seq + [self.pad_id] * (max_len - len(seq)))
            unpadded_lens.append(len(seq))
        
        tokens_batch = torch.from_numpy(np.array(seqs, dtype=np.int64))
 
        tokens_lens = torch.IntTensor([seq for seq in unpadded_lens])

        return tokens_batch, tokens_lens

class MinimalDataset(torch.utils.data.Dataset):
  def __init__(self, tokenizer: TokenizerCollator):
    self.tokenizer = tokenizer

  def __getitem__(self, cuts: CutSet) -> dict:
    cuts = cuts.sort_by_duration()
    audios, audio_lens = collate_audio(cuts)
    tokens, token_lens = self.tokenizer(cuts)
    return {
        "audio": audios,
        "audio_lens": audio_lens,
        "tokens": tokens,
        "token_lens": token_lens,
    }

class EvaluationDataset(torch.utils.data.Dataset):
    '''
    Dataset for use when evaluating the model
    Therefore we need to return audio, 
    and text but no tokens!
    '''
    @staticmethod
    def __getitem__(cuts: CutSet) -> dict:
        cuts = cuts.sort_by_duration()
        audios, audio_lens = collate_audio(cuts)
        return {
            "audio": audios,
            "audio_lens": audio_lens,
            "text": [" ".join(supervision.text.strip() for supervision in cut.supervisions if supervision.text.strip() != "") for cut in cuts],
        }

    
def eval_dataloader(cuts, batch_size:int, shuffle:bool=False):
    dataset = EvaluationDataset()
    sampler = SimpleCutSampler(cuts, shuffle=shuffle, max_cuts=batch_size)
    dataloader = torch.utils.data.DataLoader(dataset, sampler=sampler)
    return dataloader

def load_dataloader(cuts, tokenizer, max_duration:int, shuffle:bool=True):
    '''
    Example usage:
    - Obtain corpus
    \n
    ami_dict = load_corpus()
    train_dl = load_dataloader(ami_dict['train'], tokenizer, max_duration=360, shuffle=True)
    \n
    - tokenizer should be a sentencepiece tokenizer
    '''
    collator = TokenizerCollator(tokenizer)
    dataset = MinimalDataset(collator)
    sampler = SimpleCutSampler(cuts, max_duration=max_duration, shuffle=shuffle)
    dataloader = torch.utils.data.DataLoader(dataset, sampler=sampler)
    
    return dataloader
