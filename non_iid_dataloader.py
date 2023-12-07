import tools
import numpy as np
import torch
import lhotse
from lhotse.dataset.collation import collate_audio
from lhotse.dataset.cut_transforms import plain_concat, individual_speaker_concat
from lhotse import CutSet

from tools import TokenizerCollator, isfalse, istrue, exists

def prepare_partition(split):
    meetings = {}
    for entry in split:
        r_id = entry.supervisions[0].recording_id
        if r_id not in meetings:
            meetings[r_id] = []
        if entry.supervisions[0].custom != None: # = None for some silent segments that lost time information, need to repartition to get this back )::::
            assert 'segment_start' in entry.supervisions[0].custom, "custom field must contain segment_start"
            meetings[r_id].append(entry)
   
    for r_id in meetings.keys():
        meetings[r_id] = sorted(meetings[r_id], key=lambda x: x.supervisions[0].custom['segment_start'])
    
    return meetings

def get_duration_per_partition(split, verbose:bool=True):
    partition = prepare_partition(split)
    to_hours = lambda x: x / 3600
    durations = {}
    for r_id in partition.keys():
        durations[r_id] = to_hours(sum([el.duration for el in partition[r_id]]))
        if verbose:
            print(f'{r_id}: {durations[r_id]} hours')
    return durations




def prepare_samples(
        meetings, 
        max_duration=20, 
        concat_samples=False, 
        split_speakers=False, 
        gap=0.1, 
        speaker_gap=1.0, 
        single_speaker_with_gaps=False,
        max_allowed_utterance_gap=-1.0
    ):
    '''
    meetings: dictionary of arrays corresponding to a given interview, elements in the array are utterances from the meeting
    utterances are given in a contiguous manner, i.e. the first utterance in the array is the first utterance in the meeting\n
    
    max_duration: maximum duration of a sample in seconds
    
    concat_samples: if True, concatenate samples into a single utterance
    
    single_speaker_with_gaps: if True, cuts will be made with 1 speaker per sample if there is a speaker change the next instance of the same speaker will be be added to the sample
    
    speaker_gap: only used with single_speaker_with_gap, when a sample is added to a previous sample that had a speaker change in-between, an gap of speaker_gap seconds will be added to the sample
    we will try find the size with zero remainder, or the largest remainder if there is no size with zero remainder withing the given range
    
    max_allowed_utterance_gap: if > 0, then we will not allow utterances to be added to a sample if the gap between the current utterance and the previous utterance is greater than max_allowed_utterance_gap
    (note this isn't used with single_speaker_with_gaps)
    '''
    def get_speakers(cut):
        return [el.supervisions[0].speaker for el in cut]

    if single_speaker_with_gaps:
        assert max_allowed_utterance_gap < 0, "max_allowed_utterance_gap is not used with single_speaker_with_gaps, set to -1 to disable"
        assert speaker_gap > gap, "speaker_gap is smaller than gap, this is unintended behaviour"

    samples = []
    #print('num_meeetings', len(meetings))
    for key in meetings.keys():
        meeting = meetings[key].copy()
        
        samples.extend(
            plain_concat(
                cuts = meeting,
                gap = gap, 
                max_duration = max_duration,
                concat_cuts = concat_samples, 
                seperate_speakers = split_speakers,
                speaker_list = get_speakers(meeting) if split_speakers else [],
                max_allowed_utterance_gap = max_allowed_utterance_gap,
            ) \
                if isfalse(single_speaker_with_gaps) else individual_speaker_concat(
                    cuts = meeting,
                    gap = gap,
                    max_duration = max_duration,
                    concat_cuts = concat_samples,
                    speaker_gap = speaker_gap,
                    speaker_list = get_speakers(meeting)
                )
        )

    return samples


def get_text(cutlist):
    all_text = [""]
    for i in range(len(cutlist)):
        meeting = cutlist[i]
        for z in range(len(meeting)):
            utt = meeting[z]
            for supervision in utt.supervisions:
                all_text[-1] += supervision.text.strip() if all_text[-1] == "" else " " + supervision.text.strip()
        all_text.append("") if i < len(cutlist) - 1 else None 
    
    all_text = [tools.transform_txt(el) for el in all_text]
    all_text = [el.strip() for el in all_text]

    return all_text


def get_text_shuffle(cutlist):
    '''
    same as above except the text within each sample set is shuffled
    i.e if sample is utterances within a meeting chunked up to x seconds
    '''
    all_text = []
    for i in range(len(cutlist)):
        meeting = cutlist[i]
        curs = []
        for z in range(len(meeting)):
            utt = meeting[z]
            for supervision in utt.supervisions:
                cur = supervision.text.strip()
                curs.append(cur)
        np.random.shuffle(curs)
        all_text.append(" ".join(curs).strip())
 
    return all_text

    
class ___Minimal_IID_Dataset(torch.utils.data.Dataset):
  def __init__(self, tokenizer: TokenizerCollator):
    self.tokenizer = tokenizer

  def __getitem__(self, cuts: CutSet) -> dict:
    audios, audio_lens = collate_audio(cuts)
    tokens, token_lens = self.tokenizer(cuts)
    return {
        "audio": audios,
        "audio_lens": audio_lens,
        "tokens": tokens,
        "token_lens": token_lens,
    }


class Minimal_IID_Dataset(torch.utils.data.Dataset):
  def __init__(self, tokenizer: TokenizerCollator, all_cuts, text_only=False):
    self.tokenizer = tokenizer
    self.all_cuts = all_cuts
    self.text_only = text_only
    
  def __len__(self):
    return len(self.all_cuts)

  def __getitem__(self, idx) -> dict:
    text_only = self.text_only
    cuts = self.all_cuts[idx]

    audios, audio_lens = collate_audio(cuts) if isfalse(text_only) else (None, None)
    #print(len(cuts))
    tokens, token_lens = self.tokenizer(cuts=cuts) if isfalse(text_only) else self.tokenizer(cuts=None, text=cuts)
    return {
        "audio": audios,
        "audio_lens": audio_lens,
        "tokens": tokens,
        "token_lens": token_lens,
    } if isfalse(text_only) else {
        "tokens": tokens,
        "token_lens": token_lens,
        "text": cuts
    }

class Minimal_Evaluation_IID_Dataset(torch.utils.data.Dataset):
  def __init__(self, all_cuts, return_speaker=False):
    self.all_cuts = all_cuts
    self.return_speaker = return_speaker
    
  def __len__(self):
    return len(self.all_cuts)

  def __getitem__(self, idx) -> dict:
    cuts = self.all_cuts[idx]
    
    audios, audio_lens = collate_audio(cuts)
    out = {
        "audio": audios,
        "audio_lens": audio_lens,
        "text": [tools.remove_multiple_spaces(" ".join(supervision.text for supervision in cut.supervisions)) for cut in cuts],
    }

    if self.return_speaker:
        out["speakers"] = [[el.speaker for el in cut.supervisions] for cut in cuts]

    return out

def get_eval_dataloader(
    split,
    max_duration=25,
    return_speaker=False,
    batch_size=1,
    concat_samples=False,
    split_speakers=False,
    text_only=False,
    gap=0.1,
    speaker_gap=1.0,
    single_speaker_with_gaps=False,
    max_allowed_utterance_gap=-1,
    shuffle=False,
    ):
    assert isfalse(split_speakers) or concat_samples, "concat_samples must be True if split_speakers is True"
    assert text_only, "Not here use standard dataloader"

    meetings = prepare_partition(split)
    samples = prepare_samples(
        meetings = meetings, 
        max_duration = max_duration, 
        concat_samples=concat_samples, 
        split_speakers=split_speakers, 
        gap=gap, 
        speaker_gap=speaker_gap,
        single_speaker_with_gaps=single_speaker_with_gaps,
        max_allowed_utterance_gap=max_allowed_utterance_gap,
    )
    return torch.utils.data.DataLoader(
        Minimal_Evaluation_IID_Dataset(samples, return_speaker=return_speaker),
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_batch_fn_eval
    )


def collate_batch_fn_eval(batch):
    #raise NotImplementedError()

    max_len = max(el['audio'].shape[1] for el in batch)
    # pad audio to max length
    for el in batch:
        el['audio'] = torch.nn.functional.pad(el['audio'], (0, max_len - el['audio'].shape[1]))
        el['segment_lens'] = torch.LongTensor([el['audio'].shape[0]])
    # concatenate everything
    collated = {}
    for key in batch[0].keys():
        if key == 'text' or key == 'speakers':
            collated[key] = [[sub_el] for el in batch for sub_el in el[key]]
        else:
            collated[key] = torch.cat([el[key] for el in batch], dim=0)

    return collated


def collate_batch_handler(text_only=False):
    def collate_batch_fn(batch):
        #raise NotImplementedError()
        
        max_len = max(el['audio'].shape[1] for el in batch) if isfalse(text_only) else None
        max_len_tokens = max(max(el['token_lens']) for el in batch)
        # pad audio to max length
        for el in batch:
            if isfalse(text_only):
                el['audio'] = torch.nn.functional.pad(el['audio'], (0, max_len - el['audio'].shape[1]))
                el['segment_lens'] = torch.LongTensor([el['audio'].shape[0]])
            el['tokens'] = torch.nn.functional.pad(el['tokens'], (0, max_len_tokens - el['tokens'].shape[1]))
            
        # concatenate everything
        collated = {}
        for key in batch[0].keys():
            collated[key] = torch.cat([el[key] for el in batch], dim=0) if key != 'text' else [el[key] for el in batch]
            

        return collated
    return collate_batch_fn
       



def get_data_loader(
        split, 
        tokenizer=None, 
        shuffle=True, 
        max_duration=16,
        num_workers=4, 
        pinned_memory=False,
        batch_size=1,
        concat_samples=False,
        split_speakers=False,
        gap=0.1,
        speaker_gap=1.0,
        single_speaker_with_gaps=False,
        text_only=False,
        max_allowed_utterance_gap=-1,
        pad_id=0,
    ):
    '''
    split: {train, dev, test} load using tools.load_corpus
    max_duration: maximum duration of a sample in seconds
    we will try find the size with zero remainder, or the largest remainder
    '''
    meetings = prepare_partition(split)
    
    samples = prepare_samples(
        meetings=meetings, 
        max_duration=max_duration, 
        concat_samples=concat_samples,
        split_speakers=split_speakers,
        gap=gap,
        speaker_gap=speaker_gap,
        single_speaker_with_gaps=single_speaker_with_gaps,
        max_allowed_utterance_gap=max_allowed_utterance_gap,
    )

    if text_only:
        samples = get_text(samples)

    if tokenizer == None:
        tokenizer = tools.load_tokenizer()

    tokencollator = TokenizerCollator(tokenizer, text_only=text_only, pad_id=pad_id)

    dataset = Minimal_IID_Dataset(tokencollator, samples, text_only=text_only)

    return torch.utils.data.DataLoader(dataset, 
        shuffle=shuffle, 
        num_workers=num_workers, 
        pin_memory=pinned_memory,
        batch_size=batch_size,
        collate_fn=collate_batch_handler(text_only=text_only) 
    )

