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
            n_estimators=200,          # More trees for better accuracy (was 100)
            max_depth=15,              # Deeper trees for more complex patterns (was 10)
            min_samples_split=4,       # Slightly more aggressive splitting (was 5)
            min_samples_leaf=1,        # Allow smaller leaf nodes for more detail (was 2)
            max_features='sqrt',       # Standard practice for random forests
            bootstrap=True,            # Use bootstrapping
            random_state=42,
            class_weight='balanced',
            n_jobs=-1                  # Use all available CPU cores
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
            "spectral_centroid": (130, 280),     # (mean, std) - Adjusted for more realism
            "spectral_bandwidth": (180, 90),
            "spectral_rolloff": (450, 180),
            "zero_crossing_rate": (0.05, 0.02),
            "spectral_flatness": (0.1, 0.04),
            "spectral_spread": (220, 90),
        },
        "snare": {
            "spectral_centroid": (1600, 350),
            "spectral_bandwidth": (1300, 280),
            "spectral_rolloff": (3200, 750),
            "zero_crossing_rate": (0.35, 0.08),
            "spectral_flatness": (0.22, 0.09),
            "spectral_spread": (1600, 420),
        },
        "snare_electric": {
            "spectral_centroid": (2200, 480),
            "spectral_bandwidth": (1600, 350),
            "spectral_rolloff": (4200, 950),
            "zero_crossing_rate": (0.45, 0.12),
            "spectral_flatness": (0.32, 0.12),
            "spectral_spread": (2100, 580),
        },
        "hihat_closed": {
            "spectral_centroid": (3700, 750),
            "spectral_bandwidth": (2700, 550),
            "spectral_rolloff": (7500, 1400),
            "zero_crossing_rate": (0.75, 0.18),
            "spectral_flatness": (0.65, 0.18),
            "spectral_spread": (3200, 750),
        },
        "hihat_open": {
            "spectral_centroid": (4200, 850),
            "spectral_bandwidth": (3200, 650),
            "spectral_rolloff": (8300, 1700),
            "zero_crossing_rate": (0.82, 0.18),
            "spectral_flatness": (0.72, 0.18),
            "spectral_spread": (3700, 850),
        },
        "tom_floor": {
            "spectral_centroid": (320, 90),
            "spectral_bandwidth": (270, 90),
            "spectral_rolloff": (850, 280),
            "zero_crossing_rate": (0.12, 0.04),
            "spectral_flatness": (0.17, 0.04),
            "spectral_spread": (320, 90),
        },
        "tom_low": {
            "spectral_centroid": (520, 130),
            "spectral_bandwidth": (420, 130),
            "spectral_rolloff": (1250, 380),
            "zero_crossing_rate": (0.17, 0.04),
            "spectral_flatness": (0.22, 0.07),
            "spectral_spread": (520, 130),
        },
        "crash": {
            "spectral_centroid": (4800, 950),
            "spectral_bandwidth": (3700, 750),
            "spectral_rolloff": (9500, 1900),
            "zero_crossing_rate": (0.92, 0.18),
            "spectral_flatness": (0.62, 0.18),
            "spectral_spread": (4200, 950),
        },
        "ride": {
            "spectral_centroid": (5300, 1100),
            "spectral_bandwidth": (4200, 850),
            "spectral_rolloff": (10500, 2100),
            "zero_crossing_rate": (0.87, 0.22),
            "spectral_flatness": (0.67, 0.18),
            "spectral_spread": (4700, 1050),
        },
    }
    
    # Class weights to balance the dataset - generate more samples for harder-to-classify classes
    class_weights = {
        "kick": 1.0,
        "snare": 1.0,
        "snare_electric": 1.0,
        "hihat_closed": 1.5,    # Generate 50% more hi-hat samples
        "hihat_open": 1.5,      # Generate 50% more hi-hat samples
        "tom_floor": 1.0,
        "tom_low": 1.0,
        "crash": 1.5,           # Generate 50% more crash samples
        "ride": 1.5,            # Generate 50% more ride samples
    }
    
    # Generate samples with balanced distribution
    samples = []
    labels = list(drum_templates.keys())
    base_samples_per_class = num_samples // len(labels)
    
    for label in labels:
        template = drum_templates[label]
        # Apply weight to determine number of samples
        weight = class_weights.get(label, 1.0)
        samples_for_class = int(base_samples_per_class * weight)
        
        logger.debug(f"Generating {samples_for_class} samples for {label}")
        
        for _ in range(samples_for_class):
            # Generate features with random variation around mean values
            features = {}
            
            # Add some cross-correlation between features to make it more realistic
            # E.g., if centroid is higher than avg, bandwidth might also be higher
            centroid_factor = np.random.normal(0, 1)  # Random factor for correlated variation
            
            for feature_name, (mean, std) in template.items():
                # Base value with normal distribution
                base_value = np.random.normal(mean, std)
                
                # Add correlation factor for more realism (different for each feature)
                if feature_name == "spectral_centroid":
                    # This is our base feature, just use the random value
                    value = base_value
                elif feature_name == "spectral_bandwidth":
                    # Bandwidth correlates with centroid
                    value = base_value + (centroid_factor * std * 0.3)  # 30% correlation
                elif feature_name == "spectral_rolloff":
                    # Rolloff correlates with centroid
                    value = base_value + (centroid_factor * std * 0.4)  # 40% correlation
                elif feature_name == "zero_crossing_rate":
                    # ZCR correlates with centroid
                    value = base_value + (centroid_factor * std * 0.25)  # 25% correlation
                else:
                    # Other features have weaker correlation
                    value = base_value + (centroid_factor * std * 0.15)  # 15% correlation
                
                # Ensure we don't get negative values for features that should be positive
                features[feature_name] = float(max(0.0001, value))
            
            # Add RMS energy (not used for classification but needed for completeness)
            features["rms"] = float(np.random.uniform(0.1, 1.0))
            
            # Create sample
            samples.append(DrumSample(features=features, label=label))
    
    # Shuffle samples
    np.random.shuffle(samples)
    
    # Augment data with noise
    noisy_samples = []
    num_noise_samples = min(num_samples // 5, 200)  # Add up to 20% noisy samples, max 200
    
    # Add more augmentation for problematic classes
    problematic_classes = ["hihat_closed", "hihat_open", "crash", "ride"]
    for _ in range(num_noise_samples):
        # Bias towards problematic classes
        if np.random.random() < 0.6:  # 60% chance to pick a problematic class
            # Get samples from problematic classes
            problematic_samples = [s for s in samples if s.label in problematic_classes]
            if problematic_samples:
                original = problematic_samples[np.random.randint(0, len(problematic_samples))]
            else:
                original = samples[np.random.randint(0, len(samples))]
        else:
            # Pick a random sample to modify
            original = samples[np.random.randint(0, len(samples))]
        
        # Create a copy with noise
        noisy_features = {}
        for name, value in original.features.items():
            # Add 5-15% random noise
            noise_factor = np.random.uniform(0.95, 1.15)
            noisy_features[name] = float(max(0.0001, value * noise_factor))
        
        # Add to samples list
        noisy_samples.append(DrumSample(features=noisy_features, label=original.label))
    
    # Additional extreme samples for hi-hats and cymbals
    extreme_samples = []
    for label in ["hihat_closed", "hihat_open", "crash", "ride"]:
        template = drum_templates[label]
        # Generate a few samples with more extreme values
        for _ in range(30):  # 30 extreme samples per difficult class
            features = {}
            
            # Use more extreme values (± 20% from mean)
            for feature_name, (mean, std) in template.items():
                # More extreme variations
                factor = np.random.choice([-1, 1]) * np.random.uniform(0.8, 1.2)
                value = mean + (factor * std)
                features[feature_name] = float(max(0.0001, value))
            
            features["rms"] = float(np.random.uniform(0.1, 1.0))
            extreme_samples.append(DrumSample(features=features, label=label))
    
    # Combine and shuffle again
    all_samples = samples + noisy_samples + extreme_samples
    np.random.shuffle(all_samples)
    
    logger.info(f"Generated {len(all_samples)} synthetic samples "
               f"({len(samples)} regular + {len(noisy_samples)} augmented + {len(extreme_samples)} extreme)")
    
    return all_samples 