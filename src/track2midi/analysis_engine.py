"""Analysis engine for detecting and classifying drum hits in audio."""

import numpy as np
import soundfile as sf
from typing import Dict, List, Tuple, TypedDict
from dataclasses import dataclass
import logging
import random
import os

from .config import (
    DEFAULT_SENSITIVITY,
    MIN_ONSET_DISTANCE_MS,
    ONSET_WINDOW_DURATION_MS,
    DRUM_TEMPLATES,
)
from .ml_classifier import load_or_create_classifier, get_default_model_path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global ML classifier instance (loaded on first use)
_ml_classifier = None
_using_ml_classification = False

@dataclass
class DetectedDrumEvent:
    """Represents a detected drum hit with its properties."""
    time_seconds: float
    label: str
    amplitude: float


def detect_onsets(
    audio_data: np.ndarray,
    sr: float,
    sensitivity_factor: float = DEFAULT_SENSITIVITY,
) -> np.ndarray:
    """Detect onset times in the audio using energy-based detection.
    
    Args:
        audio_data: Input audio data
        sr: Sample rate in Hz
        sensitivity_factor: Multiplier for onset detection threshold
        
    Returns:
        Array of onset times in seconds
    """
    logger.info("Starting onset detection...")
    
    # Calculate frame size and hop size
    frame_size = int(0.02 * sr)  # 20ms frames
    hop_size = int(0.01 * sr)    # 10ms hop
    
    # Calculate energy for each frame
    energy = []
    for i in range(0, len(audio_data) - frame_size, hop_size):
        frame = audio_data[i:i + frame_size]
        energy.append(np.sum(frame ** 2))
    energy = np.array(energy)
    
    # Calculate energy difference
    energy_diff = np.diff(energy)
    energy_diff = np.append(0, energy_diff)  # Pad with 0 to match length
    
    # Normalize energy difference
    energy_diff = (energy_diff - np.mean(energy_diff)) / np.std(energy_diff)
    
    # Find peaks in energy difference
    min_distance = int(MIN_ONSET_DISTANCE_MS * sr / (1000 * hop_size))
    threshold = np.mean(energy_diff) + sensitivity_factor * np.std(energy_diff)
    
    logger.info(f"Onset detection threshold: {threshold:.3f}")
    
    # Find peaks
    peaks = []
    for i in range(1, len(energy_diff) - 1):
        if (energy_diff[i] > threshold and 
            energy_diff[i] > energy_diff[i-1] and 
            energy_diff[i] > energy_diff[i+1]):
            if not peaks or (i - peaks[-1]) >= min_distance:
                peaks.append(i)
    
    # Convert peak indices to time
    onset_times = np.array(peaks) * hop_size / sr
    
    logger.info(f"Detected {len(onset_times)} onsets")
    if len(onset_times) > 0:
        logger.info(f"First onset at {onset_times[0]:.3f}s, last onset at {onset_times[-1]:.3f}s")
    
    return onset_times


def estimate_tempo(onset_times: np.ndarray, default_tempo: float = 120.0) -> float:
    """Estimate tempo based on median inter-onset intervals.
    
    Args:
        onset_times: Array of onset times in seconds.
        default_tempo: Default tempo to return if estimation is not possible.
        
    Returns:
        Estimated tempo in BPM.
    """
    if len(onset_times) > 1:
        intervals = np.diff(onset_times)
        # Filter for reasonable intervals (0.3s to 1.5s, i.e., 40 BPM to 200 BPM)
        reasonable_intervals = intervals[(intervals >= 0.3) & (intervals <= 1.5)]
        
        if len(reasonable_intervals) > 0:
            median_interval = np.median(reasonable_intervals)
            tempo = 60.0 / median_interval
            # Clamp tempo to a reasonable range
            tempo = min(200.0, max(60.0, tempo))
            logger.info(f"Estimated tempo from onsets: {tempo:.1f} BPM")
            return tempo
        else:
            logger.info(f"No reasonable intervals found for tempo estimation, using default: {default_tempo} BPM")
            return default_tempo
    else:
        logger.info(f"Not enough onsets to estimate tempo, using default: {default_tempo} BPM")
        return default_tempo


def extract_features_for_onset(
    audio_data: np.ndarray,
    sr: float,
    onset_time: float,
    window_duration_ms: int = ONSET_WINDOW_DURATION_MS,
) -> Dict[str, float]:
    """Extract audio features for a detected onset.
    
    Args:
        audio_data: Input audio data
        sr: Sample rate in Hz
        onset_time: Time of the onset in seconds
        window_duration_ms: Duration of the analysis window in milliseconds
        
    Returns:
        Dictionary of extracted features
    """
    # Convert onset time to sample index
    onset_sample = int(onset_time * sr)
    window_samples = int(window_duration_ms * sr / 1000)
    
    # Extract window around onset
    start_sample = max(0, onset_sample - window_samples // 2)
    end_sample = min(len(audio_data), onset_sample + window_samples // 2)
    window = audio_data[start_sample:end_sample]
    
    # Calculate FFT
    fft = np.abs(np.fft.rfft(window))
    freqs = np.fft.rfftfreq(len(window), 1/sr)
    
    # Calculate features
    # Spectral centroid
    centroid = np.sum(freqs * fft) / np.sum(fft)
    
    # Spectral bandwidth
    bandwidth = np.sqrt(np.sum((freqs - centroid)**2 * fft) / np.sum(fft))
    
    # Spectral rolloff (95th percentile)
    rolloff = freqs[np.where(np.cumsum(fft) >= 0.95 * np.sum(fft))[0][0]]
    
    # Zero crossing rate
    zcr = np.sum(np.abs(np.diff(np.signbit(window)))) / len(window)
    
    # RMS energy
    rms = np.sqrt(np.mean(window**2))
    
    # Spectral flatness
    flatness = np.exp(np.mean(np.log(fft + 1e-10))) / np.mean(fft)
    
    # Spectral spread
    spread = np.sqrt(np.sum((freqs - centroid)**2 * fft) / np.sum(fft))
    
    # Scale RMS to a more usable range (0.1 to 1.0)
    scaled_rms = 0.1 + 0.9 * (rms / np.max(audio_data))
    
    features = {
        "spectral_centroid": float(centroid),
        "spectral_bandwidth": float(bandwidth),
        "spectral_rolloff": float(rolloff),
        "zero_crossing_rate": float(zcr),
        "spectral_flatness": float(flatness),
        "spectral_spread": float(spread),
        "rms": float(scaled_rms),
    }
    
    logger.debug(f"Features for onset at {onset_time:.3f}s: {features}")
    
    return features


def classify_drum_hit(features: Dict[str, float]) -> str:
    """Classify a drum hit based on its features using ML classifier if available,
    falling back to rule-based approach if no model is loaded.
    
    Args:
        features: Dictionary of extracted features
        
    Returns:
        Drum type label (e.g., "kick", "snare", "hihat_closed")
    """
    global _ml_classifier, _using_ml_classification
    
    # Log feature values for debugging
    logger.debug(f"Classifying hit with centroid={features['spectral_centroid']:.1f}, "
                f"bandwidth={features['spectral_bandwidth']:.1f}, "
                f"rolloff={features['spectral_rolloff']:.2f}, "
                f"zcr={features['zero_crossing_rate']:.2f}, "
                f"flatness={features['spectral_flatness']:.2f}, "
                f"spread={features['spectral_spread']:.2f}")
    
    # Check if model file exists first
    model_path = get_default_model_path()
    model_exists = os.path.exists(model_path)
    
    # Try to use ML classifier if available
    if model_exists:
        try:
            # Load classifier on first use
            if _ml_classifier is None:
                _ml_classifier = load_or_create_classifier()
                _using_ml_classification = True
                logger.info("Using ML-based classification")
            
            # If model is trained, use it for classification
            if _ml_classifier.model is not None:
                predicted_class, class_probs = _ml_classifier.classify(features)
                
                # Log the top 3 probabilities for debugging
                top_classes = sorted(class_probs.items(), key=lambda x: x[1], reverse=True)[:3]
                top_classes_str = ", ".join([f"{cls}: {prob:.3f}" for cls, prob in top_classes])
                logger.debug(f"ML classification: {predicted_class} (probabilities: {top_classes_str})")
                
                return predicted_class
            else:
                logger.warning("ML classifier instance exists but model is None, falling back to rule-based classification")
                _using_ml_classification = False
                # Fall back to rule-based method below
        except Exception as e:
            logger.error(f"Error using ML classifier: {str(e)}, falling back to rule-based classification")
            _using_ml_classification = False
            # Fall back to rule-based method below
    else:
        if not _using_ml_classification:  # Only log once
            logger.info("No trained model found at path: {}. Using rule-based classification".format(model_path))
            _using_ml_classification = False
    
    # === Rule-based method (fallback) ===
    
    # Extract features
    centroid = features["spectral_centroid"]
    bandwidth = features["spectral_bandwidth"]
    # rolloff = features["spectral_rolloff"] # Not used in current DRUM_TEMPLATES
    # zcr = features["zero_crossing_rate"] # Not used in current DRUM_TEMPLATES
    # flatness = features["spectral_flatness"] # Not used in current DRUM_TEMPLATES
    # spread = features["spectral_spread"] # Not used in current DRUM_TEMPLATES
    rms = features["rms"] # RMS might be useful for thresholding or tie-breaking

    best_match = "unknown"
    min_distance = float('inf')

    for drum_type, template in DRUM_TEMPLATES.items():
        # Calculate a simple distance (e.g., weighted Euclidean or Mahalanobis-like)
        # For MVP, let's use normalized distance from mean, scaled by std if available
        dist = 0
        
        # Spectral Centroid
        if "spectral_centroid_mean" in template and "spectral_centroid_std" in template:
            dist += ((centroid - template["spectral_centroid_mean"]) / template["spectral_centroid_std"])**2
        elif "spectral_centroid_mean" in template: # If only mean is present
            dist += ((centroid - template["spectral_centroid_mean"]) / (template["spectral_centroid_mean"] * 0.1))**2 # Assume 10% std if not specified

        # Spectral Bandwidth
        if "spectral_bandwidth_mean" in template and "spectral_bandwidth_std" in template:
            dist += ((bandwidth - template["spectral_bandwidth_mean"]) / template["spectral_bandwidth_std"])**2
        elif "spectral_bandwidth_mean" in template:
            dist += ((bandwidth - template["spectral_bandwidth_mean"]) / (template["spectral_bandwidth_mean"] * 0.1))**2

        # Example for RMS (if we add it to templates, e.g. "rms_min_threshold")
        # if "rms_min_threshold" in template and rms < template["rms_min_threshold"]:
        #     dist += float('inf') # Penalize if below RMS threshold

        if dist < min_distance:
            min_distance = dist
            best_match = drum_type
            
    # Basic threshold for matching quality, if min_distance is too high, it's "unknown"
    # This threshold is arbitrary and needs tuning.
    # A distance based on sum of squared normalized differences (like chi-squared per degree of freedom)
    # of 2-3 per feature might be a starting point. For 2 features, maybe threshold of 5-10.
    MAX_ACCEPTABLE_DISTANCE = 10.0 # Needs tuning
    if min_distance > MAX_ACCEPTABLE_DISTANCE:
        logger.debug(f"Rule-based: No close match. Min distance {min_distance:.2f} for {best_match}. Classified as unknown.")
        return "unknown"
    
    logger.debug(f"Rule-based classification: {best_match} (distance: {min_distance:.2f})")
    return best_match


def is_using_ml_classification() -> bool:
    """Check if ML classification is being used.
    
    Returns:
        True if ML classification is active, False otherwise
    """
    global _using_ml_classification
    return _using_ml_classification


def force_load_ml_classifier() -> bool:
    """Explicitly loads the ML classifier and activates ML classification.
    
    Returns:
        True if ML classifier was successfully loaded, False otherwise
    """
    global _ml_classifier, _using_ml_classification
    
    # Check if model file exists
    model_path = get_default_model_path()
    if not os.path.exists(model_path):
        logger.warning(f"No trained model found at path: {model_path}")
        return False
    
    try:
        # Load or reload the ML classifier
        _ml_classifier = load_or_create_classifier()
        
        # Check if model was loaded successfully
        if _ml_classifier.model is not None:
            _using_ml_classification = True
            logger.info("ML classifier loaded and activated successfully")
            return True
        else:
            logger.warning("ML classifier instance created but model is None")
            return False
    
    except Exception as e:
        logger.error(f"Error loading ML classifier: {str(e)}")
        return False


def process_audio(
    audio_data: np.ndarray,
    sr: float,
    sensitivity_factor: float = DEFAULT_SENSITIVITY,
) -> Tuple[List[DetectedDrumEvent], float]:
    """Process audio to detect and classify drum hits.
    
    Args:
        audio_data: Input audio data
        sr: Sample rate in Hz
        sensitivity_factor: Multiplier for onset detection threshold
        
    Returns:
        Tuple of (list of detected drum events, estimated tempo)
    """
    logger.info("Starting audio processing...")
    
    # Check if ML classification will be used by loading model first
    global _ml_classifier
    model_path = get_default_model_path()
    if os.path.exists(model_path) and _ml_classifier is None:
        _ml_classifier = load_or_create_classifier()
        
    # Report which classification method will be used
    logger.info(f"Will use {'ML-based' if is_using_ml_classification() else 'rule-based'} classification")
    
    # Detect onsets
    onset_times = detect_onsets(audio_data, sr, sensitivity_factor)
    logger.info(f"Detected {len(onset_times)} onsets")
    
    # Estimate tempo
    tempo = estimate_tempo(onset_times) # Call the new function
    # logger.info(f"Estimated tempo: {tempo:.1f} BPM") # Logging is now inside estimate_tempo
    
    # Process each onset
    drum_events = []
    for i, onset_time in enumerate(onset_times):
        # Extract features
        features = extract_features_for_onset(audio_data, sr, onset_time)
        
        # Classify drum hit
        label = classify_drum_hit(features)
        
        # Create drum event
        event = DetectedDrumEvent(
            time_seconds=float(onset_time),
            label=label,
            amplitude=float(features["rms"]),
        )
        drum_events.append(event)
        
        # Log every 10th event
        if i % 10 == 0:
            logger.info(f"Processed {i+1}/{len(onset_times)} events")
    
    # Log final results
    drum_counts = {}
    for event in drum_events:
        drum_counts[event.label] = drum_counts.get(event.label, 0) + 1
    
    logger.info("Drum hit classification results:")
    for label, count in drum_counts.items():
        logger.info(f"  {label}: {count} hits")
    
    return drum_events, tempo 