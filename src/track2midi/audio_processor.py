"""Audio processing module for loading and preprocessing audio files."""

import os
import numpy as np
import soundfile as sf
import logging
from typing import Tuple, Optional

from .config import DEFAULT_SAMPLE_RATE

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_audio(
    file_path: str,
    target_sr: int = DEFAULT_SAMPLE_RATE,
    duration: Optional[float] = None,
    offset: float = 0.0,
) -> Tuple[np.ndarray, int]:
    """Load audio file and convert to mono with specified sample rate.
    
    Args:
        file_path: Path to the audio file
        target_sr: Target sample rate in Hz
        duration: Duration to load in seconds (None for full file)
        offset: Start time offset in seconds
        
    Returns:
        Tuple of (audio data as numpy array, sample rate)
        
    Raises:
        FileNotFoundError: If the audio file does not exist
        RuntimeError: If the audio file cannot be loaded
    """
    logger.info(f"Loading audio file: {file_path}")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Audio file not found: {file_path}")
    
    try:
        # Load audio using soundfile
        logger.info("Loading with soundfile...")
        audio_data, sr = sf.read(file_path)
        
        # Get duration for logging
        duration_sec = len(audio_data) / sr
        logger.info(f"Successfully loaded with soundfile. Duration: {duration_sec:.2f}s, Sample rate: {sr}Hz")
        
        # Convert stereo to mono if needed
        if len(audio_data.shape) > 1 and audio_data.shape[1] > 1:
            audio_data = np.mean(audio_data, axis=1)
        
        # Resample if necessary
        if sr != target_sr:
            # Simple resampling with linear interpolation
            # For better quality, consider using a dedicated resampling library
            orig_len = len(audio_data)
            resampled_len = int(orig_len * target_sr / sr)
            resampling_indices = np.linspace(0, orig_len - 1, resampled_len)
            audio_data = np.interp(resampling_indices, np.arange(orig_len), audio_data)
            sr = target_sr
            logger.info(f"Resampled audio from {orig_len} samples at {sr}Hz to {resampled_len} samples at {target_sr}Hz")
        
        return audio_data, sr
        
    except Exception as e:
        raise RuntimeError(f"Failed to load audio file: {str(e)}")


def normalize_audio(audio_data: np.ndarray) -> np.ndarray:
    """Normalize audio data to the range [-1.0, 1.0].
    
    Args:
        audio_data: Input audio data
        
    Returns:
        Normalized audio data
    """
    logger.info(f"Normalizing audio data. Shape: {audio_data.shape}, Max: {np.max(np.abs(audio_data)):.3f}")
    
    # Find maximum absolute amplitude
    max_amp = np.max(np.abs(audio_data))
    
    # Avoid division by zero
    if max_amp > 0:
        normalized_data = audio_data / max_amp
    else:
        normalized_data = audio_data
    
    logger.info(f"Normalized audio data. New max: {np.max(np.abs(normalized_data)):.3f}")
    return normalized_data


def get_audio_duration(file_path: str) -> float:
    """Get the duration of an audio file in seconds.
    
    Args:
        file_path: Path to the audio file
        
    Returns:
        Duration in seconds
        
    Raises:
        FileNotFoundError: If the audio file does not exist
        RuntimeError: If the audio file cannot be opened
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Audio file not found: {file_path}")
    
    try:
        info = sf.info(file_path)
        return info.duration
    except Exception as e:
        raise RuntimeError(f"Failed to get audio duration: {str(e)}") 