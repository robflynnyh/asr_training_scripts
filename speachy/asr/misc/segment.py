'''
Uses WebRTC ( https://github.com/wiseman/py-webrtcvad ) to segment audio into chunks based on voice activity detection.
https://github.com/wiseman/py-webrtcvad/blob/master/example.py
'''

import collections
import contextlib
import sys
import wave
import librosa
import webrtcvad
import numpy as np

def read_wave(path):
    """Reads a .wav file.
    Takes the path, and returns (PCM audio data, sample rate).
    """
    with contextlib.closing(wave.open(path, 'rb')) as wf:
        num_channels = wf.getnchannels()
        assert num_channels == 1
        sample_width = wf.getsampwidth()
        assert sample_width == 2
        sample_rate = wf.getframerate()
        assert sample_rate == 16000 # only 16000 is supported by downstream 
        pcm_data = wf.readframes(wf.getnframes())
        return pcm_data, sample_rate


class Frame(object):
    """Represents a "frame" of audio data."""
    def __init__(self, bytes, timestamp, duration):
        self.bytes = bytes
        self.timestamp = timestamp
        self.duration = duration


def frame_generator(frame_duration_ms, audio, sample_rate):
    """Generates audio frames from PCM audio data.
    Takes the desired frame duration in milliseconds, the PCM data, and
    the sample rate.
    Yields Frames of the requested duration.
    """
    n = int(sample_rate * (frame_duration_ms / 1000.0) * 2)
    offset = 0
    timestamp = 0.0
    duration = (float(n) / sample_rate) / 2.0
    while offset + n < len(audio):
        yield Frame(audio[offset:offset + n], timestamp, duration)
        timestamp += duration
        offset += n


def vad_collector(sample_rate, frame_duration_ms,
                  padding_duration_ms, vad, frames, min_len_sec=25):
    """Filters out non-voiced audio frames.
    Given a webrtcvad.Vad and a source of audio frames, yields only
    the voiced audio.
    Uses a padded, sliding window algorithm over the audio frames.
    When more than 90% of the frames in the window are voiced (as
    reported by the VAD), the collector triggers and begins yielding
    audio frames. Then the collector waits until 90% of the frames in
    the window are unvoiced to detrigger.
    The window is padded at the front and back to provide a small
    amount of silence or the beginnings/endings of speech around the
    voiced frames.
    Arguments:
    sample_rate - The audio sample rate, in Hz.
    frame_duration_ms - The frame duration in milliseconds.
    padding_duration_ms - The amount to pad the window, in milliseconds.
    vad - An instance of webrtcvad.Vad.
    frames - a source of audio frames (sequence or generator).
    Returns: A generator that yields PCM audio data.
    """
    num_padding_frames = int(padding_duration_ms / frame_duration_ms)
    # We use a deque for our sliding window/ring buffer.
    ring_buffer = collections.deque(maxlen=num_padding_frames)
    # We have two states: TRIGGERED and NOTTRIGGERED. We start in the
    # NOTTRIGGERED state.
    triggered = False
    min_frames = int(min_len_sec / (frame_duration_ms / 1000))

    voiced_frames = []
    for i, frame in enumerate(frames):
        is_speech = vad.is_speech(frame.bytes, sample_rate)
        if not triggered:
            ring_buffer.append((i, frame, is_speech))
            num_voiced = len([f for ix, f, speech in ring_buffer if speech])
            # If we're NOTTRIGGERED and more than 90% of the frames in
            # the ring buffer are voiced frames, then enter the
            # TRIGGERED state.
            if num_voiced > 0.9 * ring_buffer.maxlen:
                triggered = True
                # We want to yield all the audio we see from now until
                # we are NOTTRIGGERED, but we have to start with the
                # audio that's already in the ring buffer.
                for ix, f, s in ring_buffer:
                    voiced_frames.append((ix, f))
                ring_buffer.clear()
        else:
            # We're in the TRIGGERED state, so collect the audio data
            # and add it to the ring buffer.
            voiced_frames.append((i, frame))
            ring_buffer.append((i, frame, is_speech))
            num_unvoiced = len([f for ix, f, speech in ring_buffer if not speech])
            # If more than 90% of the frames in the ring buffer are
            # unvoiced, then enter NOTTRIGGERED and yield whatever
            # audio we've collected.
            # also check if the length of the voiced frames is greater than the minimum length
            if num_unvoiced > 0.9 * ring_buffer.maxlen and len(voiced_frames) > min_frames:
                triggered = False
                yield b''.join([f.bytes for ix, f in voiced_frames]), [ix for ix, f in voiced_frames]
                ring_buffer.clear()
                voiced_frames = []
    # If we have any leftover voiced audio when we run out of input,
    # yield it.
    if voiced_frames:
        yield b''.join([f.bytes for ix, f in voiced_frames]), [ix for ix, f in voiced_frames]


def float_to_pcm16(audio):
    ints = (audio * 32767).astype(np.int16)
    little_endian = ints.astype('<u2')
    buf = little_endian.tostring()
    return buf

def load_audio(audio_path, sr=16000):
    return librosa.load(audio_path, sr=sr)[0]


def process(audio, sr, vad_mode:int, min_len_seconds:int=25, padding:int=250, frame_size:int=30) -> list:
    '''
    audio_path: str = path to wav file \n
    padding: int = time in ms to pad chunks \n
    vad_mode: int {0-3} = aggressiveness at filtering out non-speech 3 = most aggressive \n
    frame_size: int {10,20,30} = frame size in ms to process audio
    '''
    audio = float_to_pcm16(audio) # convert to pcm16
    vad = webrtcvad.Vad(vad_mode)
    frames = frame_generator(frame_size, audio, sr)
    frames = list(frames)
    segments = vad_collector(sr, frame_size, padding, vad, frames, min_len_seconds)
    chunks = []
    timeidx = []
    for segment, indexes in segments:
        floatseg = librosa.util.buf_to_float(segment) # convert to float
        '''
        #implement this to work properly to store idx bins for each chunk after concatenation with timeidx for each bin
        if len(chunks) != 0 and ( len(chunks[-1]) + len(floatseg) ) < max_seg_len:
            chunks[-1] = np.concatenate((chunks[-1], floatseg))
            timeidx[-1].append({
                'start': indexes[0]*frame_size / 1000,
                'end': indexes[-1]*frame_size / 1000
            })
        else:
            chunks.append(floatseg)
            timeidx.append([{
                'start': (indexes[0]*frame_size) / 1000,
                'end': (indexes[-1]*frame_size) / 1000
            }])
        
        '''
        chunks.append(floatseg)
        timeidx.append({
            'start': (indexes[0]*frame_size) / 1000, # in seconds
            'end': (indexes[-1]*frame_size) / 1000
        })
        
    return chunks, timeidx
