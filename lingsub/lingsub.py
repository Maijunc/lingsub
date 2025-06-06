from typing import List, Union, Optional
from pathlib import Path
from pprint import pformat
from queue import Queue
import shutil
from faster_whisper.transcribe import Segment
import json

from lingsub.defaults import default_preprocess_options, default_asr_options, default_vad_options
from lingsub.preprocess import Preprocessor
from lingsub.logger import logger
from lingsub.transcribe import Transcriber

from lingsub.utils import get_file_type, extract_audio, extend_filename, Timer, format_timestamp, get_audio_duration

class SRCer:
    def __init__(self, whisper_model: str = 'large-v3', compute_type: str = 'float16', device: str = 'cuda',
                 asr_options: Optional[dict] = None, vad_options: Optional[dict] = None,
                 preprocess_options: Optional[dict] = None):

        self.from_video = set()

        # Merge default options with provided options
        self.asr_options = {**default_asr_options, **(asr_options or {})}
        self.vad_options = {**default_vad_options, **(vad_options or {})}
        self.preprocess_options = {**default_preprocess_options, **(preprocess_options or {})}

        self.transcriber = Transcriber(model_name=whisper_model, compute_type=compute_type, device=device,
                                       asr_options=self.asr_options, vad_options=self.vad_options)

        self.transcribed_paths = []

    def clear_temp_files(self, paths):
        """
        Clear the temporary files generated during the transcription and translation process.

        Args:
            paths (List[Path]): List of paths to the processed audio files.

        This method removes temporary folders and generated wave files from video processing.
        """
        temp_folders = set([path.parent for path in paths])
        for folder in temp_folders:
            assert folder.name == 'preprocessed', f'Not a temporary folder: {folder}'

            shutil.rmtree(folder)
            logger.debug(f'Removed {folder}')

        for input_video_path in self.from_video:
            generated_wave = input_video_path.with_suffix('.wav')
            if generated_wave.exists():
                generated_wave.unlink()
                logger.debug(f'Removed generated wav (from video): {generated_wave}')

    def produce_transcriptions(self, transcription_queue, audio_paths, src_lang):
        """
        Sequentially produce transcriptions for given audio paths and put them in the queue.

        Args:
            transcription_queue (Queue): Queue to store transcribed paths.
            audio_paths (List[Path]): List of audio file paths to transcribe.
            src_lang (str): Source language for transcription. If None, language will be auto-detected.

        This method processes each audio file sequentially, transcribing it if necessary,
        and puts the path of the transcribed JSON file into the queue.
        """
        for audio_path in audio_paths:
            transcribed_path = extend_filename(audio_path, '_transcribed').with_suffix('.json')
            if not transcribed_path.exists():
                with Timer('Transcription process'):
                    logger.info(
                        f'Audio length: {audio_path}: {format_timestamp(get_audio_duration(audio_path), fmt="srt")}')
                    segments, info = self.transcriber.transcribe(audio_path, language=src_lang)
                    logger.info(f'Detected language: {info.language}')

                    # [Segment(start, end, text, words=[Word(start, end, word, probability)])]

                # Save the transcribed json
                self.to_json(segments, name=transcribed_path, lang=info.language)  # xxx_transcribed.json
            else:
                logger.info(f'Found transcribed json file: {transcribed_path}')
            transcription_queue.put(transcribed_path)
            # logger.info(f'Put transcription: {transcribed_path}')

        transcription_queue.put(None)
        logger.info('Transcription producer finished.')

    def run(self, paths: Union[str, Path, List[Union[str, Path]]], src_lang: Optional[str] = None,
            noise_suppress=False, vocal_separate=False,
            clear_temp=False):
        '''
        Run the entire transcription and translation process.
        Args:
            noise_suppress:
            vocal_separate:

        Returns:

        '''
        self.transcribed_paths = []

        if not paths:
            logger.warning('No audio/video file given. Skip SRCer.run()')
            return []

        if isinstance(paths, str) or isinstance(paths, Path):
            paths = [paths]

        paths = list(map(Path, paths))

        audio_paths = self.pre_process(paths, noise_suppress=noise_suppress, vocal_separate=vocal_separate)

        logger.info(f'Working on {len(audio_paths)} audio files: {pformat(audio_paths)}')

        transcription_queue = Queue()

        self.produce_transcriptions(transcription_queue, audio_paths, src_lang)

        if clear_temp:
            logger.info('Clearing temporary folder...')
            self.clear_temp_files(audio_paths)

    @staticmethod
    def to_json(segments: List[Segment], name, lang):
        """
        Convert transcription segments to JSON format and save to file.

        Args:
            segments (List[Segment]): List of transcription segments.
            name (str): Name of the output JSON file.
            lang (str): Language of the transcription.

        Returns:
            dict: The JSON representation of the transcription.

        This method creates a JSON structure from the transcription segments and saves it to a file.
        """
        result = {
            'language': lang,
            'segments': []
        }

        if not segments:
            result['segments'].append({
                'start': 0.0,
                'end': 5.0,
                'text': "no speech found"
            })
        else:
            for segment in segments:
                result['segments'].append({
                    'start': segment.start,
                    'end': segment.end,
                    'text': segment.text
                })

        with open(name, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=4)

        logger.info(f'File saved to {name}')

        return result

    def pre_process(self, paths, noise_suppress=False, vocal_separate=False):
        """
        Preprocess input audio/video files.

        Args:
            paths (List[Path]): Input file paths
            noise_suppress (bool): Apply noise suppression if True
            vocal_separate (bool): Apply vocal separation if True

        Returns:
            List[Path]: Preprocessed audio file paths
        """
        paths = [Path(p) for p in set(paths)]

        for i, path in enumerate(paths):
            if not path.is_file():
                raise FileNotFoundError(f'File not found: {path}')

            if get_file_type(path) == 'video':
                self.from_video.add(path.with_suffix(''))
                audio_path = path.with_suffix('.wav')
                if not audio_path.exists():
                    extract_audio(path)
                paths[i] = audio_path

        return Preprocessor(paths, options=self.preprocess_options).run(noise_suppress=noise_suppress,
                                                                        vocal_separate=vocal_separate)