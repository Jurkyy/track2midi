"""Script to train the drum classifier model."""

import os
import argparse
import logging
from typing import List, Optional

from .ml_classifier import (
    DrumClassifierML, 
    generate_synthetic_training_data,
    get_default_model_path
)
from .config import MODELS_DIR

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def train_model(
    num_samples: int = 2000,
    test_size: float = 0.2,
    output_path: Optional[str] = None
) -> None:
    """Train a new drum classification model.
    
    Args:
        num_samples: Number of synthetic samples to generate
        test_size: Proportion of data to use for testing
        output_path: Path to save the model, or None for default
    """
    logger.info(f"Starting model training with {num_samples} synthetic samples")
    
    # Generate synthetic training data
    samples = generate_synthetic_training_data(num_samples)
    logger.info(f"Generated {len(samples)} balanced training samples")
    
    # Create and train the classifier
    classifier = DrumClassifierML()
    metrics = classifier.train(samples, test_size=test_size)
    
    # Print metrics
    logger.info(f"Model accuracy: {metrics['accuracy']:.4f}")
    logger.info("Classification report:")
    for class_name, stats in metrics['classification_report'].items():
        if isinstance(stats, dict):  # Skip aggregate metrics
            precision = stats.get('precision', 0)
            recall = stats.get('recall', 0)
            f1 = stats.get('f1-score', 0)
            support = stats.get('support', 0)
            logger.info(f"  {class_name}: precision={precision:.3f}, recall={recall:.3f}, "
                        f"f1={f1:.3f}, support={support}")
    
    # Save model
    model_path = output_path or get_default_model_path()
    classifier.save(model_path)
    logger.info(f"Model saved to {model_path}")


def main() -> None:
    """Command-line entry point for training the model."""
    parser = argparse.ArgumentParser(description="Train the drum classification model")
    parser.add_argument(
        "--samples", type=int, default=2000,
        help="Number of synthetic samples to generate (default: 2000)"
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
        num_samples=args.samples,
        test_size=args.test_size,
        output_path=args.output
    )


if __name__ == "__main__":
    main() 