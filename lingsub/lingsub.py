from typing import List, Union, Optional
from pathlib import Path
from pprint import pformat
import shutil

from lingsub.defaults import default_preprocess_options
from lingsub.utils import get_file_type, extract_audio
from lingsub.preprocess import Preprocessor
from lingsub.logger import logger

class SRCer:
    def __init__(self,preprocess_options: Optional[dict] = None):

        self.from_video = set()

        # Merge default options with provided options
        self.preprocess_options = {**default_preprocess_options, **(preprocess_options or {})}

        self.transcribed_paths = []

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

        return Preprocessor(paths, options=self.preprocess_options).run(noise_suppress=noise_suppress, vocal_separate=vocal_separate)

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

    def run(self, paths: Union[str, Path, List[Union[str, Path]]], noise_suppress=False, vocal_separate=False,
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

        if clear_temp:
            logger.info('Clearing temporary folder...')
            self.clear_temp_files(audio_paths)

