"""Train the ML drum classifier with real-world data."""

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
    use_real_data: bool = True,
    save_path: Optional[str] = None,
    mp3_dir: str = DEFAULT_MP3_DIR,
    midi_dir: str = DEFAULT_MIDI_DIR,
    test_size: float = 0.2,
    focus_on_classes: Optional[List[str]] = None
) -> tuple[float, str, str]:
    """Train the drum classifier model using real-world data.
    
    Args:
        use_real_data: Whether to include real-world data (will be true by default effectively)
        save_path: Path to save the model, or None for default
        mp3_dir: Directory containing MP3 files
        midi_dir: Directory containing MIDI files
        test_size: Proportion of samples to use for testing
        focus_on_classes: List of drum classes to focus on
        
    Returns:
        Tuple of (accuracy, classification_report_string, model_save_path)
    """
    logger.info("Starting model training using exclusively real-world data.")
    
    classifier = DrumClassifierML()
    
    real_samples = []
    if use_real_data:
        logger.info("Collecting real-world training data...")
        if focus_on_classes:
            logger.info(f"Using focused sampling for classes: {focus_on_classes}")
            real_samples = collect_focused_samples(
                mp3_dir=mp3_dir, 
                midi_dir=midi_dir,
                target_classes=focus_on_classes,
            )
        else:
            real_samples = collect_samples_from_directory(
                mp3_dir=mp3_dir,
                midi_dir=midi_dir,
            )
    
    if not real_samples:
        logger.error("No real samples collected. Cannot train model.")
        # Return a structure consistent with successful training but indicating no data
        return 0.0, "No samples collected. Training cannot proceed.", get_default_model_path()

    all_samples = real_samples
    logger.info(f"Using {len(all_samples)} real samples for training.")
    
    # Shuffle samples
    np.random.shuffle(all_samples)
    
    # Log dataset composition (real data only now)
    real_count = len(all_samples)
    logger.info(f"Final dataset composition: {real_count} real samples")
    
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
    
    sanitized_results = {}
    for key, value in results.items():
        if isinstance(value, dict):
            sanitized_results[key] = {k: v if not isinstance(v, np.ndarray) else v.tolist() 
                                     for k, v in value.items()}
        elif isinstance(value, np.ndarray):
            sanitized_results[key] = value.tolist()
        else:
            sanitized_results[key] = value
    
    sanitized_results["dataset_info"] = {
        "total_samples": len(all_samples),
        "real_samples": real_count,
        # "synthetic_samples": 0, # Removed
        # "used_synthetic_data_flag": False # Removed
    }
    
    with open(metrics_path, 'w') as f:
        json.dump(sanitized_results, f, indent=2)
    
    logger.info(f"Training metrics saved to {metrics_path}")
    logger.info(f"Model trained with accuracy: {results['accuracy']:.4f}")
    
    # Extract report for return value, ensure it's a string
    report_for_return = results.get("classification_report", {})
    if isinstance(report_for_return, dict):
        report_str_for_return = classification_report_to_str(report_for_return)
    else:
        report_str_for_return = str(report_for_return)

    return results['accuracy'], report_str_for_return, save_path # Updated return value

def classification_report_to_str(report_dict: Dict) -> str:
    """Converts a classification report dictionary to a formatted string."""
    lines = []
    # Check if it's the full report dict or just the accuracy score etc.
    if not isinstance(report_dict, dict) or not any(isinstance(v, dict) for v in report_dict.values()):
        return str(report_dict) # Not a full report, just str it

    for label, metrics in report_dict.items():
        if label in ["accuracy", "macro avg", "weighted avg"]:
            if isinstance(metrics, dict): # e.g. weighted avg can be a dict
                line = f"{label}: " + " ".join([f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}" for k,v in metrics.items() if k!= 'support'])
                lines.append(line)
                if 'support' in metrics:
                    lines.append(f"    support: {metrics['support']}")
            elif isinstance(metrics, float): # accuracy is a float
                 lines.append(f"{label}: {metrics:.4f}")
            else:
                lines.append(f"{label}: {metrics}") # Other top-level items like macro avg fields
        elif isinstance(metrics, dict): # Per-class metrics
            lines.append(f"{label}:")
            metric_items = [f"{metric_name}: {value:.4f}" if isinstance(value, float) else f"{metric_name}: {value}" 
                            for metric_name, value in metrics.items()]
            lines.append("  " + " | ".join(metric_items))
        else: # Fallback for unexpected structure
            lines.append(f"{label}: {metrics}")
    return "\n".join(lines)

def main() -> None:
    """Command-line entry point for training the model."""
    parser = argparse.ArgumentParser(description="Train the drum classification model (uses real data only)")
    parser.add_argument(
        "--real-data", action="store_true", default=True,
        help="Use real-world data from examples directory (default: True, always active now)"
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
    
    train_model(
        use_real_data=True,
        save_path=args.output,
        mp3_dir=args.mp3_dir,
        midi_dir=args.midi_dir,
        test_size=args.test_size
    )

if __name__ == "__main__":
    main() 