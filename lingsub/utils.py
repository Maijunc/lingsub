import torch
from pathlib import Path
import filetype
import ffmpeg
from lingsub.logger import logger

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