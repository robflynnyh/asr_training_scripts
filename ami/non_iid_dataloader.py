import tools
import numpy as np
import torch

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


def prepare_samples(meetings, max_duration=20, concat_samples=False, split_speakers=False, gap=0.1, speaker_gap=1.0, single_speaker_with_gaps=False):
    '''
    meetings: dictionary of arrays corresponding to a given interview, elements in the array are utterances from the meeting
    utterances are given in a contiguous manner, i.e. the first utterance in the array is the first utterance in the meeting\n
    max_duration: maximum duration of a sample in seconds
    concat_samples: if True, concatenate samples into a single utterance
    single_speaker_with_gaps: if True, cuts will be made with 1 speaker per sample if there is a speaker change the next instance of the same speaker will be be added to the sample
    speaker_gap: only used with single_speaker_with_gap, when a sample is added to a previous sample that had a speaker change in-between, an gap of speaker_gap seconds will be added to the sample
    we will try find the size with zero remainder, or the largest remainder if there is no size with zero remainder withing the given range
    '''
    def get_speakers(cut):
        return [el.supervisions[0].speaker for el in cut]

    if single_speaker_with_gaps:
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
                speaker_list = get_speakers(meeting) if split_speakers else []
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
  def __init__(self, tokenizer: TokenizerCollator, all_cuts):
    self.tokenizer = tokenizer
    self.all_cuts = all_cuts
    
  def __len__(self):
    return len(self.all_cuts)

  def __getitem__(self, idx) -> dict:
    cuts = self.all_cuts[idx]
    audios, audio_lens = collate_audio(cuts)
    tokens, token_lens = self.tokenizer(cuts)
    return {
        "audio": audios,
        "audio_lens": audio_lens,
        "tokens": tokens,
        "token_lens": token_lens,
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
    gap=0.1,
    speaker_gap=1.0,
    single_speaker_with_gaps=False,
    ):
    assert isfalse(split_speakers) or concat_samples, "concat_samples must be True if split_speakers is True"
    meetings = prepare_partition(split)
    samples = prepare_samples(
        meetings = meetings, 
        max_duration = max_duration, 
        concat_samples=concat_samples, 
        split_speakers=split_speakers, 
        gap=gap, 
        speaker_gap=speaker_gap,
        single_speaker_with_gaps=single_speaker_with_gaps
    )
    return torch.utils.data.DataLoader(
        Minimal_Evaluation_IID_Dataset(samples, return_speaker=return_speaker),
        batch_size=batch_size,
        shuffle=False,
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

def collate_batch_fn(batch):
    #raise NotImplementedError()
    
    max_len = max(el['audio'].shape[1] for el in batch)
    max_len_tokens = max(max(el['token_lens']) for el in batch)
    # pad audio to max length
    for el in batch:
        el['audio'] = torch.nn.functional.pad(el['audio'], (0, max_len - el['audio'].shape[1]))
        el['tokens'] = torch.nn.functional.pad(el['tokens'], (0, max_len_tokens - el['tokens'].shape[1]))
        el['segment_lens'] = torch.LongTensor([el['audio'].shape[0]])
    # concatenate everything
    collated = {}
    for key in batch[0].keys():
        collated[key] = torch.cat([el[key] for el in batch], dim=0)

    return collated
       


def get_data_loader(
        split, 
        tokenizer=None, 
        shuffle=True, 
        max_duration=16,
        num_workers=4, 
        pinned_memory=True,
        batch_size=1,
        concat_samples=False,
        split_speakers=False,
        gap=0.1,
        speaker_gap=1.0,
        single_speaker_with_gaps=False
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
        single_speaker_with_gaps=single_speaker_with_gaps
    )

    if tokenizer == None:
        tokenizer = tools.load_tokenizer()
    tokencollator = TokenizerCollator(tokenizer)
    dataset = Minimal_IID_Dataset(tokencollator, samples)

    return torch.utils.data.DataLoader(dataset, 
        shuffle=shuffle, 
        num_workers=num_workers, 
        pin_memory=pinned_memory,
        batch_size=batch_size,
        collate_fn=collate_batch_fn 
    )

