"""Real-world data collection module for training ML models with actual audio examples."""

import os
import numpy as np
import soundfile as sf
import mido
from typing import List, Dict, Tuple, Optional, Set
import logging
from dataclasses import dataclass
from .ml_classifier import DrumSample

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
DEFAULT_WINDOW_SIZE_MS = 50  # Analysis window size in milliseconds
DEFAULT_SAMPLE_RATE = 44100  # Default sample rate in Hz
ALIGNMENT_THRESHOLD_MS = 40  # Max time difference to align MIDI and audio events

# MIDI drum mapping
DRUM_MAPPING = {
    35: "kick",           # Acoustic Bass Drum
    36: "kick",           # Bass Drum
    38: "snare",          # Acoustic Snare
    40: "snare_electric", # Electric Snare
    42: "hihat_closed",   # Closed Hi-hat
    46: "hihat_open",     # Open Hi-hat
    49: "crash",          # Crash Cymbal
    51: "ride",           # Ride Cymbal
    43: "tom_floor",      # High Floor Tom
    45: "tom_low",        # Low Tom
}

@dataclass
class MidiEvent:
    """Represents a MIDI note event with time and type information."""
    time_seconds: float
    note: int
    drum_type: str

def extract_midi_events(midi_path: str) -> List[MidiEvent]:
    """Extract drum events from a MIDI file.
    
    Args:
        midi_path: Path to the MIDI file
        
    Returns:
        List of MidiEvent objects
    """
    logger.info(f"Extracting MIDI events from {midi_path}")
    
    try:
        midi_file = mido.MidiFile(midi_path)
        
        # Extract timing information
        tempo = 500000  # Default tempo (120 BPM)
        for track in midi_file.tracks:
            for msg in track:
                if msg.type == 'set_tempo':
                    tempo = msg.tempo
                    break
        
        # Get tick to time conversion factor
        tick_to_seconds = tempo / (midi_file.ticks_per_beat * 1000000)
        
        # Extract note events
        events = []
        for track in midi_file.tracks:
            absolute_time_ticks = 0
            
            for msg in track:
                # Update absolute time
                absolute_time_ticks += msg.time
                
                # Process note-on events with velocity > 0
                if msg.type == 'note_on' and hasattr(msg, 'velocity') and msg.velocity > 0:
                    # Convert time to seconds
                    time_seconds = absolute_time_ticks * tick_to_seconds
                    
                    # Map note to drum type
                    drum_type = DRUM_MAPPING.get(msg.note, f"unknown_{msg.note}")
                    
                    # Create event
                    event = MidiEvent(
                        time_seconds=time_seconds,
                        note=msg.note,
                        drum_type=drum_type
                    )
                    events.append(event)
        
        # Sort events by time
        events.sort(key=lambda e: e.time_seconds)
        
        logger.info(f"Extracted {len(events)} MIDI events")
        return events
        
    except Exception as e:
        logger.error(f"Error extracting MIDI events: {str(e)}")
        return []

def extract_features_for_point(
    audio_data: np.ndarray,
    sr: int,
    time_seconds: float,
    window_size_ms: int = DEFAULT_WINDOW_SIZE_MS
) -> Dict[str, float]:
    """Extract audio features for a specific time point.
    
    Args:
        audio_data: Audio data array
        sr: Sample rate in Hz
        time_seconds: Time point in seconds
        window_size_ms: Size of the analysis window in milliseconds
        
    Returns:
        Dictionary of extracted features
    """
    # Convert time to sample index
    center_sample = int(time_seconds * sr)
    window_samples = int(window_size_ms * sr / 1000)
    
    # Extract window around time point
    start_sample = max(0, center_sample - window_samples // 2)
    end_sample = min(len(audio_data), center_sample + window_samples // 2)
    window = audio_data[start_sample:end_sample]
    
    # Handle stereo audio
    if len(window.shape) > 1 and window.shape[1] > 1:
        window = window.mean(axis=1)  # Convert to mono
    
    # If window is too small, pad with zeros
    if len(window) < window_samples:
        padding = np.zeros(window_samples - len(window))
        window = np.concatenate([window, padding])
    
    # Calculate FFT
    fft = np.abs(np.fft.rfft(window))
    freqs = np.fft.rfftfreq(len(window), 1/sr)
    
    # Calculate features
    # Spectral centroid
    centroid = np.sum(freqs * fft) / (np.sum(fft) + 1e-10)
    
    # Spectral bandwidth
    bandwidth = np.sqrt(np.sum((freqs - centroid)**2 * fft) / (np.sum(fft) + 1e-10))
    
    # Spectral rolloff (95th percentile)
    cumsum = np.cumsum(fft)
    rolloff_point = np.where(cumsum >= 0.95 * cumsum[-1])[0]
    rolloff = freqs[rolloff_point[0]] if len(rolloff_point) > 0 else 0
    
    # Zero crossing rate
    zcr = np.sum(np.abs(np.diff(np.signbit(window)))) / len(window)
    
    # RMS energy
    rms = np.sqrt(np.mean(window**2))
    
    # Spectral flatness
    flatness = np.exp(np.mean(np.log(fft + 1e-10))) / (np.mean(fft) + 1e-10)
    
    # Spectral spread
    spread = np.sqrt(np.sum((freqs - centroid)**2 * fft) / (np.sum(fft) + 1e-10))
    
    # Scale RMS to a more usable range (0.1 to 1.0)
    max_amplitude = np.max(np.abs(audio_data))
    scaled_rms = 0.1 + 0.9 * (rms / (max_amplitude + 1e-10))
    
    features = {
        "spectral_centroid": float(centroid),
        "spectral_bandwidth": float(bandwidth),
        "spectral_rolloff": float(rolloff),
        "zero_crossing_rate": float(zcr),
        "spectral_flatness": float(flatness),
        "spectral_spread": float(spread),
        "rms": float(scaled_rms),
    }
    
    return features

def collect_samples_from_file_pair(
    mp3_path: str,
    midi_path: str,
    window_size_ms: int = DEFAULT_WINDOW_SIZE_MS
) -> List[DrumSample]:
    """Collect training samples from an MP3/MIDI file pair.
    
    Args:
        mp3_path: Path to the MP3 file
        midi_path: Path to the MIDI file
        window_size_ms: Size of the analysis window in milliseconds
        
    Returns:
        List of DrumSample objects
    """
    logger.info(f"Collecting samples from {mp3_path} / {midi_path}")
    
    try:
        # Load audio file
        audio_data, sr = sf.read(mp3_path)
        
        # Extract MIDI events
        midi_events = extract_midi_events(midi_path)
        
        # For each MIDI event, extract features
        samples = []
        for event in midi_events:
            # Extract features at the event time
            features = extract_features_for_point(
                audio_data, sr, event.time_seconds, window_size_ms
            )
            
            # Create sample
            sample = DrumSample(
                features=features,
                label=event.drum_type
            )
            samples.append(sample)
        
        logger.info(f"Collected {len(samples)} samples")
        return samples
        
    except Exception as e:
        logger.error(f"Error collecting samples: {str(e)}")
        return []

def get_available_training_pairs(
    mp3_dir: str, 
    midi_dir: str
) -> Set[str]:
    """Get the set of files that have both MP3 and MIDI versions available.
    
    Args:
        mp3_dir: Directory containing MP3 files
        midi_dir: Directory containing MIDI files
        
    Returns:
        Set of basenames that have both MP3 and MIDI versions
    """
    # Get a list of MP3 and MIDI files
    mp3_files = {f.replace(".mp3", "") for f in os.listdir(mp3_dir) 
                if f.endswith(".mp3") and not f.endswith(".mp3:Zone.Identifier")}
    
    midi_files = {f.replace(".mid", "") for f in os.listdir(midi_dir) 
                 if f.endswith(".mid")}
    
    # Find files that have both MP3 and MIDI versions
    common_files = mp3_files.intersection(midi_files)
    return common_files

def collect_samples_from_directory(
    mp3_dir: str,
    midi_dir: str,
    max_files: Optional[int] = None,
    selected_files: Optional[List[str]] = None
) -> List[DrumSample]:
    """Collect training samples from directories of MP3 and MIDI files.
    
    Args:
        mp3_dir: Directory containing MP3 files
        midi_dir: Directory containing MIDI files
        max_files: Maximum number of files to process (None for all)
        selected_files: Specific files to process (None for all available pairs)
        
    Returns:
        List of DrumSample objects
    """
    logger.info(f"Collecting samples from directories: {mp3_dir} and {midi_dir}")
    
    # Get files with both MP3 and MIDI versions
    common_files = get_available_training_pairs(mp3_dir, midi_dir)
    file_count = len(common_files)
    logger.info(f"Found {file_count} files with both MP3 and MIDI versions")
    
    # Filter to selected files if specified
    if selected_files is not None:
        common_files = {f for f in common_files if f in selected_files}
        logger.info(f"Filtered to {len(common_files)}/{file_count} selected files")
    
    # Limit the number of files if requested
    if max_files is not None:
        common_files = list(common_files)[:max_files]
        logger.info(f"Limited to first {len(common_files)} files")
    
    # Collect samples from each file pair
    all_samples = []
    processed_count = 0
    
    for basename in common_files:
        mp3_path = os.path.join(mp3_dir, f"{basename}.mp3")
        midi_path = os.path.join(midi_dir, f"{basename}.mid")
        
        samples = collect_samples_from_file_pair(mp3_path, midi_path)
        all_samples.extend(samples)
        
        processed_count += 1
        logger.info(f"Processed {processed_count}/{len(common_files)} files, collected {len(samples)} samples from {basename}")
    
    logger.info(f"Collected a total of {len(all_samples)} samples from {processed_count} files")
    
    # Print statistics on collected samples
    class_counts = {}
    for sample in all_samples:
        class_counts[sample.label] = class_counts.get(sample.label, 0) + 1
    
    logger.info("Sample distribution by class:")
    for label, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
        logger.info(f"  {label}: {count} samples ({count / len(all_samples) * 100:.1f}%)")
    
    return all_samples

def collect_focused_samples(
    mp3_dir: str,
    midi_dir: str,
    target_classes: List[str],
    max_files: Optional[int] = None
) -> List[DrumSample]:
    """Collect training samples with a focus on specific drum classes.
    
    Args:
        mp3_dir: Directory containing MP3 files
        midi_dir: Directory containing MIDI files
        target_classes: List of drum classes to focus on (e.g., ["snare", "snare_electric"])
        max_files: Maximum number of files to process (None for all)
        
    Returns:
        List of DrumSample objects
    """
    logger.info(f"Collecting samples focused on classes: {target_classes}")
    
    # First, scan MIDI files to find those with the target classes
    common_files = get_available_training_pairs(mp3_dir, midi_dir)
    
    # Prioritize files with the target classes
    prioritized_files = []
    for basename in common_files:
        midi_path = os.path.join(midi_dir, f"{basename}.mid")
        events = extract_midi_events(midi_path)
        
        # Check if the file contains any of the target classes
        contains_target = False
        target_count = 0
        
        for event in events:
            if event.drum_type in target_classes:
                contains_target = True
                target_count += 1
        
        if contains_target:
            # Add to prioritized list with count of target notes
            prioritized_files.append((basename, target_count))
    
    # Sort by number of target notes (descending)
    prioritized_files.sort(key=lambda x: x[1], reverse=True)
    
    # Get basenames of prioritized files
    selected_files = [basename for basename, _ in prioritized_files]
    
    # Limit to max_files if specified
    if max_files is not None:
        selected_files = selected_files[:max_files]
    
    # Collect samples from prioritized files
    logger.info(f"Found {len(selected_files)} files containing target classes")
    return collect_samples_from_directory(mp3_dir, midi_dir, selected_files=selected_files) 