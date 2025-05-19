"""Audio processing module for loading and preprocessing audio files."""

import numpy as np
import librosa
import soundfile as sf
from typing import Tuple
import logging

from .config import DEFAULT_SAMPLE_RATE

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_audio(file_path: str, target_sr: int = DEFAULT_SAMPLE_RATE) -> Tuple[np.ndarray, float]:
    """Load and preprocess an audio file.
    
    Args:
        file_path: Path to the audio file
        target_sr: Target sample rate (default: from config)
        
    Returns:
        Tuple of (audio_data, sample_rate)
        
    Raises:
        FileNotFoundError: If the audio file doesn't exist
        ValueError: If the audio file can't be loaded
    """
    logger.info(f"Loading audio file: {file_path}")
    
    try:
        # First try loading with librosa
        logger.info("Attempting to load with librosa...")
        audio_data, sr = librosa.load(
            file_path,
            sr=target_sr,
            mono=True,
            duration=None,  # Load entire file
            offset=0.0,     # Start from beginning
            res_type='kaiser_best'  # Use high-quality resampling
        )
        logger.info(f"Successfully loaded with librosa. Duration: {len(audio_data)/sr:.2f}s, Sample rate: {sr}Hz")
        return audio_data, sr
    except Exception as e:
        logger.warning(f"Librosa loading failed: {str(e)}")
        try:
            # Fallback to soundfile if librosa fails
            logger.info("Attempting to load with soundfile...")
            audio_data, sr = sf.read(file_path)
            if len(audio_data.shape) > 1:  # Convert to mono if stereo
                audio_data = audio_data.mean(axis=1)
            if sr != target_sr:  # Resample if needed
                audio_data = librosa.resample(
                    audio_data,
                    orig_sr=sr,
                    target_sr=target_sr,
                    res_type='kaiser_best'
                )
                sr = target_sr
            logger.info(f"Successfully loaded with soundfile. Duration: {len(audio_data)/sr:.2f}s, Sample rate: {sr}Hz")
            return audio_data, sr
        except Exception as sf_error:
            error_msg = f"Error loading audio file: {str(e)} (librosa) / {str(sf_error)} (soundfile)"
            logger.error(error_msg)
            raise ValueError(error_msg)


def normalize_audio(audio_data: np.ndarray) -> np.ndarray:
    """Normalize audio data to have a maximum absolute value of 1.0.
    
    Args:
        audio_data: Input audio data as numpy array
        
    Returns:
        Normalized audio data
    """
    logger.info(f"Normalizing audio data. Shape: {audio_data.shape}, Max: {np.max(np.abs(audio_data)):.3f}")
    normalized = librosa.util.normalize(audio_data)
    logger.info(f"Normalized audio data. New max: {np.max(np.abs(normalized)):.3f}")
    return normalized


def get_audio_duration(audio_data: np.ndarray, sr: float) -> float:
    """Calculate the duration of the audio in seconds.
    
    Args:
        audio_data: Input audio data as numpy array
        sr: Sample rate in Hz
        
    Returns:
        Duration in seconds
    """
    duration = len(audio_data) / sr
    logger.info(f"Audio duration: {duration:.2f} seconds")
    return duration 