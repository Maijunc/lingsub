from typing import List, Dict, Any, Union

import tiktoken
import torch
import re
from pathlib import Path
import filetype
import ffmpeg
import spacy
import time
import audioread
import langcodes
from lingua import LanguageDetectorBuilder

from lingsub.defaults import supported_languages_lingua
from lingsub.logger import logger

def extend_filename(filename: Path, extend: str) -> Path:
    """Extend a filename with some string."""
    return filename.with_stem(filename.stem + extend)

def format_timestamp(seconds: float, fmt: str = 'lrc') -> str:
    """
    Converts a timestamp in seconds into a string in the specified format.

    Args:
        seconds (float): Timestamp in seconds.
        fmt (str): Format of the output string. Supported values are:
            - 'lrc' for LRC format, e.g., '1:23.45'
            - 'srt' for SRT format, e.g., '01:23:45,678'

    Returns:
        str: A string representation of the timestamp in the specified format.
    """
    # Ensure that the timestamp is non-negative.
    # assert seconds >= 0, "non-negative timestamp expected"
    if seconds < 0:
        logger.warning(f"Negative timestamp: {seconds}")
        if fmt == 'lrc':
            return '0:00.00'
        elif fmt == 'srt':
            return '00:00:00,000'
        else:
            raise ValueError(f"Unsupported timestamp format: {fmt}")

    # Convert seconds into milliseconds.
    milliseconds = round(seconds * 1000.0)

    # Extract hours, minutes, seconds, and milliseconds from milliseconds.
    hours = milliseconds // 3600000
    milliseconds %= 3600000
    minutes = milliseconds // 60000
    milliseconds %= 60000
    seconds = milliseconds // 1000
    milliseconds %= 1000

    # Return the timestamp in the specified format.
    if fmt == 'lrc':
        return f"{minutes:02d}:{seconds:02d}.{milliseconds // 10:02d}"
    elif fmt == 'srt':
        return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"
    else:
        raise ValueError(f"Unsupported timestamp format: {fmt}")

def get_spacy_lib(lang):
    special_case = {
        'core_web': ['zh', 'en'],
        'ent_wiki': ['xx']
    }

    mid_str = 'core_news'
    for k, v in special_case.items():
        if lang in v:
            mid_str = k

    return f'{lang}_{mid_str}_sm'

def spacy_load(lang) -> spacy.Language:
    lib_name = get_spacy_lib(lang)
    try:
        nlp = spacy.load(lib_name)
    except (IOError, ImportError, OSError):
        logger.warning(f'Spacy model {lib_name} missed, downloading')
        spacy.cli.download(lib_name)
        nlp = spacy.load(lib_name)

    return nlp

def get_audio_duration(path: Union[str, Path]) -> float:
    with audioread.audio_open(str(path)) as audio:
        return audio.duration

class Timer:
    def __init__(self, task=""):
        self._start = None
        self._stop = None
        self.task = task

    def start(self):
        if self.task:
            logger.info(f'Start {self.task}')
        self._start = time.perf_counter()

    def stop(self):
        self._stop = time.perf_counter()
        logger.info(f'{self.task} Elapsed: {self._elapsed:.2f}s')

    @property
    def _elapsed(self):
        return self._stop - self._start

    @property
    def duration(self):
        return time.perf_counter() - self._start

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()

def release_memory(model: torch.nn.Module) -> None:
    # gc.collect()
    torch.cuda.empty_cache()
    del model

def get_file_type(path: Path) -> str:
    if path.suffix == '.ts':
        return 'video'

    try:
        file_type = filetype.guess(path).mime.split('/')[0]
    except (TypeError, AttributeError) as e:
        raise RuntimeError(f'File {path} is not a valid file.') from e

    if file_type not in ['audio', 'video']:
        raise RuntimeError(f'File {path} is not a valid file. Should be audio or video file.')

    return file_type

# 提取音频
def extract_audio(path: Path) -> Path:
    """
    Extract audio from video.
    :return: Audio path
    """
    file_type = get_file_type(path)
    if file_type == 'audio':
        return path

    probe = ffmpeg.probe(path)
    audio_streams = next((stream for stream in probe['streams'] if stream['codec_type'] == 'audio'), None)
    sample_rate = audio_streams['sample_rate']
    logger.info(f'File {path}: Audio sample rate: {sample_rate}')

    audio, err = (
        ffmpeg.input(path).
        output("pipe:", format='wav', acodec='pcm_s16le', ar=sample_rate, loglevel='quiet').
        run(capture_stdout=True)
    )

    if err:
        raise RuntimeError(f'ffmpeg error: {err}')

    audio_path = path.with_suffix('.wav')
    with open(audio_path, 'wb') as f:
        f.write(audio)

    return audio_path

def parse_timestamp(time_stamp: str, fmt: str) -> float:
    """
    Parse a timestamp from a subtitle file and convert it to seconds.

    Args:
        time_stamp (str): The timestamp to parse.
        fmt (str): The format of `time_stamp`. Supported values are:
            - 'lrc' for LRC format, e.g., '1:23.45'
            - 'srt' for SRT format, e.g., '01:23:45,678'

    Returns:
        float: The timestamp in seconds.

    Raises:
        ValueError: If `time_stamp` does not match the expected format for the specified `fmt`.
    """

    if fmt == 'lrc':
        if not re.match(r'^\d+:\d+\.\d+$', time_stamp):
            raise ValueError(f"Invalid timestamp format for LRC: {time_stamp}")
        minutes, seconds = time_stamp.split(':')
        seconds, hundredths_of_sec = seconds.split('.')
        return int(minutes) * 60 + int(seconds) + int(hundredths_of_sec) / 100.0
    elif fmt == 'srt':
        if not re.match(r'^\d+:\d+:\d+,\d+$', time_stamp):
            raise ValueError(f"Invalid timestamp format for SRT: {time_stamp}")
        hours, minutes, seconds = time_stamp.split(':')
        seconds, milliseconds = seconds.split(',')
        return int(hours) * 3600 + int(minutes) * 60 + int(seconds) + int(milliseconds) / 1000.0
    else:
        raise ValueError(f"Unsupported timestamp format: {fmt}")

def detect_lang(text):
    detector = LanguageDetectorBuilder.from_languages(*supported_languages_lingua).build()
    name = detector.detect_language_of(text).name.lower()
    lang_code = langcodes.Language.find(name).language
    return lang_code

def get_messages_token_number(messages: List[Dict[str, Any]], model: str = "gpt-3.5-turbo") -> int:
    total = sum([get_text_token_number(element['content'], model=model) for element in messages])

    return total

def get_text_token_number(text: str, model: str = "gpt-3.5-turbo") -> int:
    tokens = tiktoken.encoding_for_model(model).encode(text)

    return len(tokens)