import logging
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import List, Union

import torch
import torchaudio
from demucs.apply import apply_model
from demucs.pretrained import get_model
from df.enhance import enhance, init_df, load_audio, save_audio
from ffmpeg_normalize import FFmpegNormalize
from tqdm import tqdm

from lingsub.defaults import default_preprocess_options
from lingsub.logger import logger
from lingsub.utils import release_memory


def loudness_norm_single(audio_path: Path, ln_path: Path):
    """
    Normalize the loudness of a single audio file using FFmpegNormalize.

    Args:
        audio_path (Path): The path to the input audio file.
        ln_path (Path): The path to save the normalized audio file.
    """
    normalizer = FFmpegNormalize(output_format='wav', sample_rate=48000, progress=logger.level <= logging.DEBUG,
                                 keep_lra_above_loudness_range_target=True)

    if not ln_path.exists():
        normalizer.add_media_file(str(audio_path), str(ln_path))
        normalizer.run_normalization()


class Preprocessor:
    """
    Preprocess audio to make it clear and normalized.
    """

    def __init__(self, audio_paths: Union[str, Path, List[str], List[Path]], output_folder='preprocessed',
                 options: dict = default_preprocess_options):
        if not isinstance(audio_paths, list):
            audio_paths = [audio_paths]
        self.audio_paths = [Path(p) for p in audio_paths]
        self.output_paths = [p.parent / output_folder for p in self.audio_paths]
        self.options = options

        for path in self.output_paths:
            if not path.exists():
                path.mkdir()

    def noise_suppression(self, audio_paths: Union[str, Path, List[str], List[Path]], atten_lim_db=15):
        """
        Supress noise in audio.
        """
        if not audio_paths:
            return []

        if 'atten_lim_db' in self.options.keys():
            atten_lim_db = self.options['atten_lim_db']

        model, df_state, _ = init_df()
        chunk_size = 180  # 3 min

        ns_audio_paths = []
        for audio_path, output_path in zip(audio_paths, self.output_paths):
            audio_name = audio_path.stem
            ns_path = output_path / f'{audio_name}_ns.wav'

            if not ns_path.exists():
                audio, info = load_audio(audio_path, sr=df_state.sr())

                # Split audio into 3 min chunks
                audio_chunks = [audio[:, i:i + chunk_size * info.sample_rate]
                                for i in range(0, audio.shape[1], chunk_size * info.sample_rate)]

                enhanced_chunks = []
                for ac in tqdm(audio_chunks, desc=f'Noise suppressing for {audio_name}'):
                    enhanced_chunks.append(enhance(model, df_state, ac, atten_lim_db=atten_lim_db))

                enhanced = torch.cat(enhanced_chunks, dim=1)

                assert enhanced.shape == audio.shape, f'Enhanced audio shape does not match original audio shape: {enhanced.shape} != {audio.shape}'

                save_audio(ns_path, enhanced, sr=df_state.sr())

            ns_audio_paths.append(ns_path)

        release_memory(model)

        return ns_audio_paths

    def loudness_normalization(self, audio_paths: Union[str, Path, List[str], List[Path]]):
        """
        Normalize loudness of audio.
        """
        logger.info('Loudness normalizing...')

        args = []
        ln_audio_paths = []
        for audio_path, output_path in zip(audio_paths, self.output_paths):
            ln_path = output_path / f'{audio_path.stem}_ln.wav'
            args.append((audio_path, ln_path))
            ln_audio_paths.append(ln_path)

        # Multi-processing
        with ProcessPoolExecutor() as executor:
            results = [executor.submit(loudness_norm_single, *arg) for arg in args]

            exceptions = [res.exception() for res in results]
            if any(exceptions):
                # Get the first not None exception
                exception = next(filter(None, exceptions))

                logger.error(f'Loudness normalization failed, exception: {exception}')
                raise exception

        return ln_audio_paths

    def vocal_separation(self, audio_paths: Union[str, Path, List[str], List[Path]]):
        """
        Separate vocals from audio using Demucs.

        Args:
            audio_paths: Input audio file paths.

        Returns:
            list of Path: A list of paths to the separated vocal audio files.
        """
        if not audio_paths:
            return []

        # Get options with defaults
        model_name = self.options.get('vocal_separation_model', 'htdemucs')
        device = self.options.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')

        # Load Demucs model
        model = get_model(model_name)
        model.to(device)

        vs_audio_paths = []

        for audio_path, output_path in zip(audio_paths, self.output_paths):
            audio_name = audio_path.stem
            vs_path = output_path / f'{audio_name}_vs.wav'

            if not vs_path.exists():
                # Load audio
                waveform, sample_rate = torchaudio.load(audio_path)

                # Convert to expected format (batch, channels, time)
                if waveform.dim() == 2:
                    waveform = waveform.unsqueeze(0)  # Add batch dimension

                # Move to device
                waveform = waveform.to(device)

                # Apply model and extract vocals
                with torch.no_grad():
                    sources = apply_model(model, waveform, device=device)
                    vocals_idx = model.sources.index('vocals')
                    vocals = sources[:, vocals_idx]

                # Save the vocals
                vocals = vocals.cpu()
                torchaudio.save(vs_path, vocals[0], sample_rate)

                logger.info(f"Saved separated vocals to {vs_path}")

            vs_audio_paths.append(vs_path)

        return vs_audio_paths

    def run(self, noise_suppress=False, vocal_separate=False):
        """
        Args:
            noise_suppress (bool, optional): A boolean flag indicating whether to perform noise suppression.
                Defaults to False.
            vocal_separate (bool, optional): A boolean flag indicating whether to perform vocal separation.
                Defaults to False.

        Returns:
            list of Path: A list of Path objects representing the final processed audio paths.
        """
        # Check if the preprocessed audio already exists.
        need_process = []
        final_processed_audios = []
        for audio_path, output_path in zip(self.audio_paths, self.output_paths):
            audio_name = audio_path.stem
            preprocessed_path = output_path / f'{audio_name}_preprocessed.wav'
            final_processed_audios.append(preprocessed_path)
            if preprocessed_path.exists():
                logger.info(f'Preprocessed audio already exists in {preprocessed_path}')
                continue
            else:
                need_process.append(audio_path)

        # Apply noise suppression if requested
        processed_paths = need_process
        if noise_suppress:
            processed_paths = self.noise_suppression(processed_paths)
            logger.info('Noise suppression completed')

        # Apply vocal separation if requested
        if vocal_separate:
            processed_paths = self.vocal_separation(processed_paths)
            logger.info('Vocal separation completed')

        # Apply loudness normalization
        ln_paths: list[Path] = self.loudness_normalization(processed_paths)

        for path, audio_path in zip(ln_paths, need_process):
            audio_name = audio_path.stem

            path = path.rename(path.parent / f'{audio_name}_preprocessed.wav')
            logger.info(f'Preprocessed audio saved to {path}')

        return final_processed_audios
