"""Utilities for collecting and preparing real training data."""

import os
import glob
import logging
import numpy as np
from typing import Dict, List, Optional
import soundfile as sf

from .ml_classifier import DrumSample
from .analysis_engine import extract_features_for_onset

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def process_drum_sample(
    file_path: str,
    label: str,
    sample_rate: int = 22050,
) -> Optional[DrumSample]:
    """Process a single drum sample audio file.
    
    Args:
        file_path: Path to audio file containing a single drum hit
        label: Drum type label (e.g., "kick", "snare")
        sample_rate: Target sample rate for processing
        
    Returns:
        DrumSample object or None if processing failed
    """
    try:
        # Load audio file
        audio_data, sr = sf.read(file_path)
        
        # Convert to mono if stereo
        if len(audio_data.shape) > 1 and audio_data.shape[1] == 2:
            audio_data = np.mean(audio_data, axis=1)
        
        # Resample if needed
        if sr != sample_rate:
            # Use resampy or librosa for resampling if available
            # For simplicity, we're skipping this step in this implementation
            # as most drum samples will likely be at standard sample rates
            pass
        
        # Normalize audio
        audio_data = audio_data / np.max(np.abs(audio_data))
        
        # Find the peak (assuming drum hit is the loudest part)
        peak_index = np.argmax(np.abs(audio_data))
        peak_time = peak_index / sr
        
        # Extract features - using the same function as for real audio analysis
        features = extract_features_for_onset(audio_data, sr, peak_time)
        
        # Create drum sample
        return DrumSample(features=features, label=label)
    
    except Exception as e:
        logger.error(f"Failed to process {file_path}: {str(e)}")
        return None


def collect_samples_from_directory(
    base_dir: str,
    sample_rate: int = 22050,
) -> List[DrumSample]:
    """Collect training samples from a directory structure.
    
    The directory structure should be:
    base_dir/
      kick/
        sample1.wav
        sample2.wav
        ...
      snare/
        sample1.wav
        ...
      hihat_closed/
        ...
      ...
    
    Args:
        base_dir: Base directory containing subdirectories for each drum type
        sample_rate: Target sample rate for processing
        
    Returns:
        List of DrumSample objects
    """
    samples = []
    
    # Get subdirectories, each representing a drum type
    drum_types = [d for d in os.listdir(base_dir) 
                 if os.path.isdir(os.path.join(base_dir, d))]
    
    logger.info(f"Found drum types: {drum_types}")
    
    for drum_type in drum_types:
        drum_dir = os.path.join(base_dir, drum_type)
        
        # Get all audio files in this directory
        file_patterns = ["*.wav", "*.WAV", "*.mp3", "*.MP3", "*.ogg", "*.OGG"]
        audio_files = []
        
        for pattern in file_patterns:
            audio_files.extend(glob.glob(os.path.join(drum_dir, pattern)))
        
        logger.info(f"Found {len(audio_files)} audio samples for {drum_type}")
        
        # Process each audio file
        for file_path in audio_files:
            sample = process_drum_sample(file_path, drum_type, sample_rate)
            if sample is not None:
                samples.append(sample)
    
    logger.info(f"Collected {len(samples)} valid samples across {len(drum_types)} drum types")
    return samples


def collect_samples_from_list(
    sample_list: List[Dict[str, str]],
    sample_rate: int = 22050,
) -> List[DrumSample]:
    """Collect training samples from a list of file paths and labels.
    
    Args:
        sample_list: List of dictionaries with 'path' and 'label' keys
        sample_rate: Target sample rate for processing
        
    Returns:
        List of DrumSample objects
    """
    samples = []
    
    for item in sample_list:
        file_path = item['path']
        label = item['label']
        
        sample = process_drum_sample(file_path, label, sample_rate)
        if sample is not None:
            samples.append(sample)
    
    logger.info(f"Collected {len(samples)} valid samples from file list")
    return samples


def mix_synthetic_and_real_data(
    real_samples: List[DrumSample],
    num_synthetic: int = 1000,
) -> List[DrumSample]:
    """Mix real samples with synthetic data for improved training.
    
    Args:
        real_samples: List of real drum samples
        num_synthetic: Number of synthetic samples to generate
        
    Returns:
        Combined list of samples
    """
    from .ml_classifier import generate_synthetic_training_data
    
    # Generate synthetic data
    synthetic_samples = generate_synthetic_training_data(num_synthetic)
    
    # Combine and shuffle
    combined_samples = real_samples + synthetic_samples
    np.random.shuffle(combined_samples)
    
    logger.info(f"Combined dataset: {len(real_samples)} real samples + "
               f"{len(synthetic_samples)} synthetic samples = "
               f"{len(combined_samples)} total samples")
    
    return combined_samples 