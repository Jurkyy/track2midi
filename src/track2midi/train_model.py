"""Train the ML drum classifier with synthetic and real-world data."""

import os
import logging
import numpy as np
import argparse
from pathlib import Path
from typing import List, Dict, Optional
import json

from .ml_classifier import (
    DrumClassifierML, 
    DrumSample, 
    generate_synthetic_training_data,
    get_default_model_path
)
from .real_data_collector import collect_samples_from_directory, collect_focused_samples

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Default paths
DEFAULT_EXAMPLES_DIR = "examples"
DEFAULT_MP3_DIR = os.path.join(DEFAULT_EXAMPLES_DIR, "mp3")
DEFAULT_MIDI_DIR = os.path.join(DEFAULT_EXAMPLES_DIR, "midi")

def train_model(
    num_samples: int = 2000,
    use_real_data: bool = True,
    real_data_weight: float = 0.6,  # Weight given to real-world data (0-1)
    save_path: Optional[str] = None,
    mp3_dir: str = DEFAULT_MP3_DIR,
    midi_dir: str = DEFAULT_MIDI_DIR,
    test_size: float = 0.2,
    focus_on_classes: Optional[List[str]] = None
) -> Dict:
    """Train the drum classifier model using both synthetic and real-world data.
    
    Args:
        num_samples: Number of synthetic samples to generate
        use_real_data: Whether to include real-world data
        real_data_weight: Weight given to real data vs synthetic (0-1)
        save_path: Path to save the model, or None for default
        mp3_dir: Directory containing MP3 files
        midi_dir: Directory containing MIDI files
        test_size: Proportion of samples to use for testing
        focus_on_classes: List of drum classes to focus on (e.g., ["snare", "snare_electric"])
        
    Returns:
        Dictionary with training results
    """
    logger.info(f"Starting model training with {num_samples} synthetic samples")
    
    # Initialize classifier
    classifier = DrumClassifierML()
    
    # Generate synthetic training data
    logger.info("Generating synthetic training data...")
    synthetic_samples = generate_synthetic_training_data(num_samples=num_samples)
    
    # Collect real-world samples if requested
    real_samples = []
    if use_real_data:
        logger.info("Collecting real-world training data...")
        
        # Use focused sampling if specific classes are requested
        if focus_on_classes:
            logger.info(f"Using focused sampling for classes: {focus_on_classes}")
            real_samples = collect_focused_samples(
                mp3_dir=mp3_dir, 
                midi_dir=midi_dir,
                target_classes=focus_on_classes,
                max_files=5  # Limit to 5 files for faster training
            )
        else:
            # Standard collection from all available pairs
            real_samples = collect_samples_from_directory(
                mp3_dir=mp3_dir,
                midi_dir=midi_dir,
                max_files=3  # Limit to 3 files for faster training
            )
    
    # Combine synthetic and real-world data with proper weighting
    all_samples = []
    
    if use_real_data and real_samples:
        # Calculate sample weights to achieve desired balance
        total_samples = num_samples + len(real_samples)
        target_real_count = int(total_samples * real_data_weight)
        target_synthetic_count = total_samples - target_real_count
        
        # Determine sampling rate for each source
        if len(real_samples) > target_real_count:
            # Need to subsample real data
            real_sample_rate = target_real_count / len(real_samples)
            synthetic_sample_rate = 1.0
        else:
            # Use all real data
            real_sample_rate = 1.0
            # May need to subsample synthetic data
            remaining_space = target_synthetic_count
            synthetic_sample_rate = min(1.0, remaining_space / len(synthetic_samples))
        
        # Apply sampling rates
        logger.info(f"Balancing dataset - real data weight: {real_data_weight}")
        logger.info(f"Real sample rate: {real_sample_rate}, synthetic sample rate: {synthetic_sample_rate}")
        
        # Add real samples
        np.random.seed(42)  # For reproducibility
        for sample in real_samples:
            if np.random.random() < real_sample_rate:
                all_samples.append(sample)
        
        # Add synthetic samples
        for sample in synthetic_samples:
            if np.random.random() < synthetic_sample_rate:
                all_samples.append(sample)
    else:
        # Just use synthetic data
        all_samples = synthetic_samples
    
    # Shuffle samples
    np.random.shuffle(all_samples)
    
    # Log dataset composition
    real_count = sum(1 for s in all_samples if s in real_samples)
    synthetic_count = len(all_samples) - real_count
    logger.info(f"Final dataset composition: {real_count} real samples, {synthetic_count} synthetic samples")
    
    # Train the model
    logger.info(f"Training classifier on {len(all_samples)} samples...")
    results = classifier.train(all_samples, test_size=test_size)
    
    # Determine save path
    if save_path is None:
        save_path = get_default_model_path()
    
    # Ensure the directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Save the model
    classifier.save(save_path)
    logger.info(f"Model saved to {save_path}")
    
    # Save training metrics
    metrics_path = os.path.splitext(save_path)[0] + "_metrics.json"
    
    # Convert numpy arrays to lists for JSON serialization
    sanitized_results = {}
    for key, value in results.items():
        if isinstance(value, dict):
            sanitized_results[key] = {k: v if not isinstance(v, np.ndarray) else v.tolist() 
                                     for k, v in value.items()}
        elif isinstance(value, np.ndarray):
            sanitized_results[key] = value.tolist()
        else:
            sanitized_results[key] = value
    
    # Add dataset info
    sanitized_results["dataset_info"] = {
        "total_samples": len(all_samples),
        "real_samples": real_count,
        "synthetic_samples": synthetic_count,
        "real_data_weight": real_data_weight
    }
    
    with open(metrics_path, 'w') as f:
        json.dump(sanitized_results, f, indent=2)
    
    logger.info(f"Training metrics saved to {metrics_path}")
    logger.info(f"Model trained with accuracy: {results['accuracy']:.4f}")
    
    return results


def main() -> None:
    """Command-line entry point for training the model."""
    parser = argparse.ArgumentParser(description="Train the drum classification model")
    parser.add_argument(
        "--samples", type=int, default=2000,
        help="Number of synthetic samples to generate (default: 2000)"
    )
    parser.add_argument(
        "--real-data", action="store_true", default=True,
        help="Use real-world data from examples directory (default: True)"
    )
    parser.add_argument(
        "--no-real-data", action="store_false", dest="real_data",
        help="Don't use real-world data, only synthetic"
    )
    parser.add_argument(
        "--real-weight", type=float, default=0.6,
        help="Weight to give to real-world data vs synthetic (0-1, default: 0.6)"
    )
    parser.add_argument(
        "--mp3-dir", type=str, default=DEFAULT_MP3_DIR,
        help=f"Directory containing MP3 files (default: {DEFAULT_MP3_DIR})"
    )
    parser.add_argument(
        "--midi-dir", type=str, default=DEFAULT_MIDI_DIR,
        help=f"Directory containing MIDI files (default: {DEFAULT_MIDI_DIR})"
    )
    parser.add_argument(
        "--test-size", type=float, default=0.2,
        help="Proportion of data to use for testing (default: 0.2)"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help=f"Path to save the model (default: {get_default_model_path()})"
    )
    
    args = parser.parse_args()
    
    # Train the model
    train_model(
        num_samples=args.samples,
        use_real_data=args.real_data,
        real_data_weight=args.real_weight,
        save_path=args.output,
        mp3_dir=args.mp3_dir,
        midi_dir=args.midi_dir,
        test_size=args.test_size
    )


if __name__ == "__main__":
    main() 