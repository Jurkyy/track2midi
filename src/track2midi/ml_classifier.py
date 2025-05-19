"""Machine learning based drum hit classifier."""

import os
import pickle
import logging
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

from .config import MODELS_DIR

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DrumSample:
    """Represents a labeled drum sample for training."""
    features: Dict[str, float]
    label: str


class DrumClassifierML:
    """Machine learning based drum hit classifier."""
    
    def __init__(self) -> None:
        """Initialize the classifier."""
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = [
            "spectral_centroid", 
            "spectral_bandwidth",
            "spectral_rolloff",
            "zero_crossing_rate",
            "spectral_flatness",
            "spectral_spread",
            # RMS is used for amplitude, not for classification
        ]
        
    def _features_to_vector(self, features: Dict[str, float]) -> np.ndarray:
        """Convert feature dictionary to feature vector.
        
        Args:
            features: Dictionary of features
            
        Returns:
            Feature vector as numpy array
        """
        return np.array([features[name] for name in self.feature_names]).reshape(1, -1)
    
    def train(self, samples: List[DrumSample], test_size: float = 0.2) -> Dict[str, Any]:
        """Train the classifier on drum samples.
        
        Args:
            samples: List of labeled drum samples
            test_size: Proportion of samples to use for testing
            
        Returns:
            Dictionary with training results and metrics
        """
        logger.info(f"Training drum classifier on {len(samples)} samples")
        
        # Extract features and labels
        X = np.array([[sample.features[name] for name in self.feature_names] 
                     for sample in samples])
        y = np.array([sample.label for sample in samples])
        
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train Random Forest model (good balance of accuracy and interpretability)
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            class_weight='balanced'
        )
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate model
        y_pred = self.model.predict(X_test_scaled)
        
        # Calculate metrics
        class_report = classification_report(y_test, y_pred, output_dict=True)
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        # Extract feature importance
        feature_importance = dict(zip(self.feature_names, 
                                    self.model.feature_importances_))
        
        # Log results
        logger.info(f"Model trained with accuracy: {class_report['accuracy']:.4f}")
        logger.info(f"Feature importance: {feature_importance}")
        
        return {
            "accuracy": class_report['accuracy'],
            "classification_report": class_report,
            "confusion_matrix": conf_matrix,
            "feature_importance": feature_importance
        }
    
    def classify(self, features: Dict[str, float]) -> Tuple[str, Dict[str, float]]:
        """Classify a drum hit based on its features.
        
        Args:
            features: Dictionary of extracted features
            
        Returns:
            Tuple of (predicted class, class probabilities dict)
        """
        if self.model is None:
            raise ValueError("Model not trained or loaded. Call train() or load() first.")
        
        # Extract and scale feature vector
        X = self._features_to_vector(features)
        X_scaled = self.scaler.transform(X)
        
        # Predict class and probabilities
        predicted_class = self.model.predict(X_scaled)[0]
        class_probs = dict(zip(self.model.classes_, 
                             self.model.predict_proba(X_scaled)[0]))
        
        return predicted_class, class_probs
    
    def save(self, model_path: str) -> None:
        """Save the trained model and scaler to disk.
        
        Args:
            model_path: Path to save the model
        """
        if self.model is None:
            raise ValueError("No model to save. Train a model first.")
        
        # Create model directory if it doesn't exist
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # Save model and scaler
        with open(model_path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'scaler': self.scaler,
                'feature_names': self.feature_names
            }, f)
        
        logger.info(f"Model saved to {model_path}")
    
    def load(self, model_path: str) -> bool:
        """Load a trained model and scaler from disk.
        
        Args:
            model_path: Path to the saved model
            
        Returns:
            True if model was loaded successfully, False otherwise
        """
        try:
            with open(model_path, 'rb') as f:
                data = pickle.load(f)
                self.model = data['model']
                self.scaler = data['scaler']
                self.feature_names = data['feature_names']
            
            logger.info(f"Model loaded from {model_path}")
            return True
            
        except (FileNotFoundError, KeyError) as e:
            logger.error(f"Error loading model: {str(e)}")
            return False


def get_default_model_path() -> str:
    """Get the default path for the drum classification model.
    
    Returns:
        Path to the default model file
    """
    return os.path.join(MODELS_DIR, "drum_classifier.pkl")


def load_or_create_classifier() -> DrumClassifierML:
    """Load the classifier from disk or create a new one if not found.
    
    Returns:
        Drum classifier instance
    """
    model_path = get_default_model_path()
    classifier = DrumClassifierML()
    
    # Try to load existing model
    if os.path.exists(model_path):
        if classifier.load(model_path):
            return classifier
    
    # Return untrained classifier if model couldn't be loaded
    logger.warning("No trained model found. Using untrained classifier.")
    return classifier


def generate_synthetic_training_data(num_samples: int = 1000) -> List[DrumSample]:
    """Generate synthetic training data based on typical drum characteristics.
    
    Args:
        num_samples: Number of samples to generate
        
    Returns:
        List of synthetic drum samples
    """
    # Typical feature distributions for different drum types
    drum_templates = {
        "kick": {
            "spectral_centroid": (100, 300),     # (mean, std)
            "spectral_bandwidth": (150, 100),
            "spectral_rolloff": (500, 200),
            "zero_crossing_rate": (0.05, 0.02),
            "spectral_flatness": (0.1, 0.05),
            "spectral_spread": (200, 100),
        },
        "snare": {
            "spectral_centroid": (1500, 400),
            "spectral_bandwidth": (1200, 300),
            "spectral_rolloff": (3000, 800),
            "zero_crossing_rate": (0.3, 0.1),
            "spectral_flatness": (0.2, 0.1),
            "spectral_spread": (1500, 500),
        },
        "snare_electric": {
            "spectral_centroid": (2000, 500),
            "spectral_bandwidth": (1500, 400),
            "spectral_rolloff": (4000, 1000),
            "zero_crossing_rate": (0.4, 0.15),
            "spectral_flatness": (0.3, 0.15),
            "spectral_spread": (2000, 600),
        },
        "hihat_closed": {
            "spectral_centroid": (3500, 800),
            "spectral_bandwidth": (2500, 600),
            "spectral_rolloff": (7000, 1500),
            "zero_crossing_rate": (0.7, 0.2),
            "spectral_flatness": (0.6, 0.2),
            "spectral_spread": (3000, 800),
        },
        "hihat_open": {
            "spectral_centroid": (4000, 900),
            "spectral_bandwidth": (3000, 700),
            "spectral_rolloff": (8000, 1800),
            "zero_crossing_rate": (0.8, 0.2),
            "spectral_flatness": (0.7, 0.2),
            "spectral_spread": (3500, 900),
        },
        "tom_floor": {
            "spectral_centroid": (300, 100),
            "spectral_bandwidth": (250, 100),
            "spectral_rolloff": (800, 300),
            "zero_crossing_rate": (0.1, 0.05),
            "spectral_flatness": (0.15, 0.05),
            "spectral_spread": (300, 100),
        },
        "tom_low": {
            "spectral_centroid": (500, 150),
            "spectral_bandwidth": (400, 150),
            "spectral_rolloff": (1200, 400),
            "zero_crossing_rate": (0.15, 0.05),
            "spectral_flatness": (0.2, 0.08),
            "spectral_spread": (500, 150),
        },
        "crash": {
            "spectral_centroid": (4500, 1000),
            "spectral_bandwidth": (3500, 800),
            "spectral_rolloff": (9000, 2000),
            "zero_crossing_rate": (0.9, 0.2),
            "spectral_flatness": (0.6, 0.2),
            "spectral_spread": (4000, 1000),
        },
        "ride": {
            "spectral_centroid": (5000, 1200),
            "spectral_bandwidth": (4000, 900),
            "spectral_rolloff": (10000, 2200),
            "zero_crossing_rate": (0.85, 0.25),
            "spectral_flatness": (0.65, 0.2),
            "spectral_spread": (4500, 1100),
        },
    }
    
    # Generate samples with balanced distribution
    samples = []
    labels = list(drum_templates.keys())
    samples_per_class = num_samples // len(labels)
    
    for label in labels:
        template = drum_templates[label]
        
        for _ in range(samples_per_class):
            # Generate features with random variation around mean values
            features = {}
            for feature_name, (mean, std) in template.items():
                features[feature_name] = float(np.random.normal(mean, std))
            
            # Add RMS energy (not used for classification but needed for completeness)
            features["rms"] = float(np.random.uniform(0.1, 1.0))
            
            # Create sample
            samples.append(DrumSample(features=features, label=label))
    
    # Shuffle samples
    np.random.shuffle(samples)
    
    return samples 