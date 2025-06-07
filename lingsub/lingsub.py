import concurrent.futures
import traceback
from typing import List, Union, Optional
from pathlib import Path
from pprint import pformat
from queue import Queue
import shutil
from faster_whisper.transcribe import Segment
import json
from copy import deepcopy
from threading import Lock

from lingsub.context import TranslateInfo
from lingsub.defaults import default_preprocess_options, default_asr_options, default_vad_options
from lingsub.opt import SubtitleOptimizer
from lingsub.preprocess import Preprocessor
from lingsub.logger import logger
from lingsub.subtitle import Subtitle, BilingualSubtitle
from lingsub.transcribe import Transcriber
from lingsub.models import ModelConfig
from lingsub.translate import LLMTranslator

from lingsub.utils import get_file_type, extract_audio, extend_filename, Timer, format_timestamp, get_audio_duration

class SRCer:
    def __init__(self, whisper_model: str = 'large-v3', compute_type: str = 'float16', device: str = 'cuda',
                 chatbot_model: Union[str, ModelConfig] = 'deepseek-chat', fee_limit: float = 0.8,
                 consumer_thread: int = 4,
                 base_url_config: Optional[dict] = None, glossary: Optional[Union[dict, str, Path]] = None,
                 asr_options: Optional[dict] = None, vad_options: Optional[dict] = None,
                 preprocess_options: Optional[dict] = None, proxy: Optional[str] = None,
                 retry_model: Optional[Union[str, ModelConfig]] = None, is_force_glossary_used: bool = False):

        self.chatbot_model = chatbot_model
        self.fee_limit = fee_limit
        self.base_url_config = base_url_config
        self.retry_model = retry_model
        self.is_force_glossary_used = is_force_glossary_used
        self.glossary = self.parse_glossary(glossary)

        self.from_video = set()
        self.proxy = proxy

        self.api_fee = 0  # Can be updated in different thread, operation should be thread-safe
        self._lock = Lock()
        self.exception = None
        self.consumer_thread = consumer_thread

        # Merge default options with provided options
        self.asr_options = {**default_asr_options, **(asr_options or {})}
        self.vad_options = {**default_vad_options, **(vad_options or {})}
        self.preprocess_options = {**default_preprocess_options, **(preprocess_options or {})}

        self.transcriber = Transcriber(model_name=whisper_model, compute_type=compute_type, device=device,
                                       asr_options=self.asr_options, vad_options=self.vad_options)

        self.transcribed_paths = []

    @staticmethod
    def parse_glossary(glossary: Union[dict, str, Path]):
        if not glossary:
            return None

        if isinstance(glossary, dict):
            return glossary

        glossary_path = Path(glossary)
        if not glossary_path.exists():
            logger.warning('Glossary file not found.')
            return None

        with open(glossary_path, 'r', encoding='utf-8') as f:
            glossary = json.load(f)

        return glossary

    def translation_worker(self, transcription_queue, target_lang, skip_trans, bilingual_sub):
        """
        Worker function for parallel translation and subtitle processing.

        Args:
            transcription_queue (Queue): Queue containing paths of transcribed files.
            target_lang (str): Target language for translation.
            skip_trans (bool): Whether to skip the translation process.
            bilingual_sub (bool): Whether to generate bilingual subtitles.

        This method continuously processes transcriptions from the queue, handling translation,
        subtitle generation, and bilingual subtitle creation if required.
        """

        def process_translation(base_name, target_lang, transcribed_opt_sub, skip_trans):
            translated_path = extend_filename(transcribed_opt_sub.filename, '_translated')
            final_json_path = translated_path.with_name(f'{base_name}.json')

            if final_json_path.exists():
                return Subtitle.from_json(final_json_path)

            if skip_trans:
                shutil.copy(transcribed_opt_sub.filename, final_json_path)
                transcribed_opt_sub.filename = final_json_path
                return transcribed_opt_sub

            try:
                with Timer('Translation process'):
                    return self._translate(base_name, target_lang, transcribed_opt_sub, translated_path)
            except Exception as e:
                self.exception = e
                return None

        def generate_subtitle_files(subtitle, base_name, subtitle_format):
            subtitle_path = getattr(subtitle, f'to_{subtitle_format}')()
            result_path = subtitle_path.parent.parent / f'{base_name}.{subtitle_format}'
            shutil.move(subtitle_path, result_path)
            self.transcribed_paths.append(result_path)

        def handle_bilingual_subtitles(transcribed_path, base_name, transcribed_opt_sub, subtitle_format):
            bilingual_subtitle = BilingualSubtitle.from_preprocessed(transcribed_path.parent, base_name)
            bilingual_optimizer = SubtitleOptimizer(bilingual_subtitle)
            bilingual_optimizer.extend_time()

            bilingual_path = getattr(bilingual_subtitle, f'to_{subtitle_format}')()
            shutil.move(bilingual_path, bilingual_path.parent.parent / bilingual_path.name)

            non_translated_subtitle = transcribed_opt_sub
            optimizer = SubtitleOptimizer(non_translated_subtitle)
            optimizer.extend_time()
            non_translated_path = getattr(non_translated_subtitle, f'to_{subtitle_format}')()
            shutil.move(
                non_translated_path,
                non_translated_path.parent.parent / f'{base_name}_nontrans.{subtitle_format}'
            )

        while True:
            logger.debug('Translation worker waiting transcription...')
            transcribed_path = transcription_queue.get()

            if transcribed_path is None:
                transcription_queue.put(None)
                logger.debug('Translation worker finished.')
                return

            logger.info(f'Got transcription: {transcribed_path}')

            # Extract base name and determine subtitle format
            base_name = transcribed_path.stem.replace('_preprocessed_transcribed', '')
            subtitle_format = 'srt' if transcribed_path.parent.parent / base_name in self.from_video else 'lrc'

            # Process transcription
            transcribed_sub = Subtitle.from_json(transcribed_path)
            transcribed_opt_sub = self.post_process(transcribed_sub, update_name=True)

            # Handle translation
            final_subtitle = process_translation(base_name, target_lang, transcribed_opt_sub, skip_trans)

            # Generate and move subtitle files
            generate_subtitle_files(final_subtitle, base_name, subtitle_format)

            # Handle bilingual subtitles if needed
            if not skip_trans and bilingual_sub:
                handle_bilingual_subtitles(transcribed_path, base_name, transcribed_opt_sub, subtitle_format)

            logger.info(f'Translation fee til now: {self.api_fee:.4f} USD')

    def _translate(self, audio_name, target_lang, transcribed_opt_sub, translated_path):
        """
        Perform translation of transcribed subtitles.

        Args:
            audio_name (str): Name of the audio file.
            target_lang (str): Target language for translation.
            transcribed_opt_sub (Subtitle): Optimized transcribed subtitle object.
            translated_path (Path): Path to save the translated subtitle.

        Returns:
            Subtitle: Final translated and post-processed subtitle object.

        This method handles the translation process, including context preparation,
        actual translation, and post-processing of the translated subtitles.
        """
        context = TranslateInfo(title=audio_name, audio_type='Movie', glossary=self.glossary,
                                forced_glossary=self.is_force_glossary_used)

        json_filename = Path(translated_path.parent / (audio_name + '.json'))
        compare_path = Path(translated_path.parent, f'{audio_name}_compare.json')
        if not translated_path.exists():
            # Translate the transcribed json
            translator = LLMTranslator(chatbot_model=self.chatbot_model, fee_limit=self.fee_limit,
                                       proxy=self.proxy, base_url_config=self.base_url_config,
                                       retry_model=self.retry_model)

            target_texts = translator.translate(
                transcribed_opt_sub.texts,
                src_lang=transcribed_opt_sub.lang,
                target_lang=target_lang,
                info=context,
                compare_path=compare_path
            )

            with self._lock:
                self.api_fee += translator.api_fee  # Ensure thread-safe

            translated_sub = deepcopy(transcribed_opt_sub)
            translated_sub.set_texts(target_texts, lang=target_lang)

            # xxx_transcribed_optimized_translated.json
            translated_sub.save(translated_path, update_name=True)
        else:
            logger.info(f'Found translated json file: {translated_path}')
        translated_sub = Subtitle.from_json(translated_path)

        final_subtitle = self.post_process(translated_sub, output_name=json_filename, update_name=True,
                                           extend_time=True)  # xxx.json

        return final_subtitle

    def consume_transcriptions(self, transcription_queue, target_lang, skip_trans, bilingual_sub):
        """
        Consume transcriptions from the queue using multiple threads for parallel processing.

        Args:
            transcription_queue (Queue): Queue containing paths of transcribed files.
            target_lang (str): Target language for translation.
            skip_trans (bool): Whether to skip the translation process.
            bilingual_sub (bool): Whether to generate bilingual subtitles.

        This method creates multiple worker threads to process transcriptions in parallel,
        handling translation and subtitle generation.
        """
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(self.translation_worker, transcription_queue, target_lang, skip_trans,
                                       bilingual_sub)
                       for _ in range(self.consumer_thread)]
            concurrent.futures.wait(futures)
        logger.info('Transcription consumer finished.')

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

    def run(self, paths: Union[str, Path, List[Union[str, Path]]], src_lang: Optional[str] = None, target_lang='zh-cn',
            skip_trans=False, noise_suppress=False, vocal_separate=False, bilingual_sub=False, clear_temp=False) -> List[str]:
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

        with Timer('Transcription (Producer) and Translation (Consumer) process'):
            consumer = concurrent.futures.ThreadPoolExecutor(thread_name_prefix='Consumer') \
                .submit(self.consume_transcriptions, transcription_queue, target_lang, skip_trans, bilingual_sub)
            producer = concurrent.futures.ThreadPoolExecutor(thread_name_prefix='Producer') \
                .submit(self.produce_transcriptions, transcription_queue, audio_paths, src_lang)

            producer.result()
            consumer.result()

            if self.exception:
                traceback.print_exception(type(self.exception), self.exception, self.exception.__traceback__)
                raise self.exception

        logger.info(f'Total API fee used: {self.api_fee:.4f} USD')

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

    @staticmethod
    def post_process(transcribed_sub: Path, output_name: Path = None, remove_files: List[Path] = None,
                     update_name=False, extend_time=False):
        """
        Post-process the transcribed subtitles.

        Args:
            transcribed_sub (Path): Path to the transcribed subtitle file.
            output_name (Path, optional): Path for the output file.
            remove_files (List[Path], optional): List of files to remove after processing.
            update_name (bool): Whether to update the subtitle name.
            extend_time (bool): Whether to extend the time of subtitles.

        Returns:
            Subtitle: The post-processed subtitle object.

        This method applies various optimizations to the transcribed subtitles and saves the result.
        """
        optimizer = SubtitleOptimizer(transcribed_sub)
        optimizer.perform_all(extend_time=extend_time)
        optimizer.save(output_name, update_name=update_name)

        # Remove intermediate files
        if remove_files:
            _ = [file.unlink() for file in remove_files if file.is_file()]

        return optimizer.subtitle