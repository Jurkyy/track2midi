"""Machine learning based drum hit classifier."""

import os
import pickle
import logging
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neural_network import MLPClassifier

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
    
    def __init__(self, model_type: str = "mlp") -> None:
        """Initialize the classifier.
        
        Args:
            model_type: Type of model to use ("random_forest" or "mlp").
        """
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.model_type = model_type
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
        logger.info(f"Training drum classifier ({self.model_type}) on {len(samples)} samples")
        
        # Extract features and labels
        # X = np.array([[sample.features.get(name, 0.0) for name in self.feature_names] 
        #              for sample in samples], dtype=np.float64)
        
        # Alternative X construction:
        feature_list_of_lists = []
        for i, sample in enumerate(samples):
            row = []
            for name in self.feature_names:
                val = sample.features.get(name, 0.0)
                if not isinstance(val, (int, float)):
                    logger.warning(f"DEBUG: Non-numeric value encountered at sample index {i}, feature '{name}': {val} (type: {type(val)}). Using 0.0 instead.")
                    val = 0.0
                row.append(float(val)) # Ensure it's a float
            feature_list_of_lists.append(row)
        
        try:
            X = np.array(feature_list_of_lists, dtype=np.float64)
        except Exception as e:
            logger.error(f"DEBUG: Error converting list of lists to np.array: {e}")
            logger.error(f"DEBUG: First 3 rows of feature_list_of_lists before np.array conversion:")
            for i in range(min(3, len(feature_list_of_lists))):
                logger.error(f"DEBUG: Row {i}: {feature_list_of_lists[i]}")
            raise

        y = np.array([sample.label for sample in samples])
        
        logger.info(f"Created feature array X with shape: {X.shape} and dtype: {X.dtype}")
        logger.info(f"Created label array y with shape: {y.shape} and dtype: {y.dtype} (before encoding)")
        
        # Encode string labels to numerical labels
        if y.size > 0:
            y = self.label_encoder.fit_transform(y)
            logger.info(f"Encoded label array y with shape: {y.shape} and dtype: {y.dtype} (after encoding)")
            logger.info(f"Label encoder classes: {list(self.label_encoder.classes_)}")
        else:
            logger.warning("Label array y is empty, skipping label encoding.")
            # If y is empty, train_test_split will fail. This case should ideally be handled
            # by the existing check: if X.shape[0] == 0: return ...
        
        # --- Filter out classes with too few samples for stratification --- 
        unique_labels, counts = np.unique(y, return_counts=True)
        min_samples_per_class = 2 # For train_test_split with stratification
        
        labels_to_keep = unique_labels[counts >= min_samples_per_class]
        labels_to_remove = unique_labels[counts < min_samples_per_class]

        if len(labels_to_remove) > 0:
            logger.warning(f"Removing classes with fewer than {min_samples_per_class} samples: {list(labels_to_remove)}")
            logger.warning(f"Counts for removed classes: {dict(zip(labels_to_remove, counts[counts < min_samples_per_class]))}")
            
            indices_to_keep = np.isin(y, labels_to_keep)
            X = X[indices_to_keep]
            y = y[indices_to_keep]
            
            if X.shape[0] == 0:
                logger.error("No samples left after filtering for minimum class size. Cannot train.")
                return {"accuracy": 0.0, "classification_report": {}, "confusion_matrix": np.array([]), "feature_importance": {}}

            logger.info(f"Dataset shape after filtering: X={X.shape}, y={y.shape}")
        # --- End filtering ---

        # Force conversion and re-check for NaNs/Infs, just in case of subtle type issues
        try:
            X = X.astype(np.float64)
        except ValueError as e:
            logger.error(f"Could not convert X to float64: {e}. Dumping first 5 rows of X:")
            for i in range(min(5, X.shape[0])):
                logger.error(f"X[{i}]: {X[i]}")
            raise e

        logger.info(f"X dtype after astype(np.float64): {X.dtype}")
        logger.info(f"DEBUG: X.flags after astype(np.float64):\n{X.flags}") # Log array flags

        # === Remove Enhanced Debugging Block and manual NaN loop ===
        # (Previous manual loop and debug logs for X.dtype are removed here)

        # Attempt to replace NaNs and Infs directly using np.nan_to_num
        try:
            # Check for NaNs/Infs before attempting to replace, for logging purposes
            has_nans_before = np.any(np.isnan(X))
            has_infs_before = np.any(np.isinf(X))
            if has_nans_before:
                logger.warning("DEBUG: NaNs detected by np.any(np.isnan(X)) BEFORE np.nan_to_num.")
            if has_infs_before:
                logger.warning("DEBUG: Infs detected by np.any(np.isinf(X)) BEFORE np.nan_to_num.")

            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0) # Replace NaNs and Infs
            logger.info("DEBUG: Successfully processed X with np.nan_to_num.")

        except TypeError as te_nan_to_num:
            logger.error(f"DEBUG: TypeError during np.nan_to_num(X). Error: {te_nan_to_num}")
            logger.error(f"DEBUG: X.dtype at this point: {X.dtype}, type(X): {type(X)}")
            logger.error("DEBUG: First 3 rows of X before np.nan_to_num failure (if possible):")
            try:
                for i in range(min(3, X.shape[0])):
                    logger.error(f"DEBUG: Row {i}: {X[i]}")
            except Exception as e_log:
                logger.error(f"DEBUG: Could not log X rows: {e_log}")
            raise # Re-raise the TypeError to see it in the main GUI error message
        
        # The explicit if np.any(np.isinf(X)) block is removed as np.nan_to_num handles infs too.

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Scale features
        self.scaler = StandardScaler()
        
        # Check for zero variance before fitting scaler
        if np.any(np.std(X_train, axis=0) == 0):
            logger.error("Zero variance found in one or more features of X_train BEFORE scaling!")
            zero_var_cols = np.where(np.std(X_train, axis=0) == 0)[0]
            logger.error(f"Columns with zero variance indices: {zero_var_cols}")
            logger.error(f"Feature names with zero variance: {[self.feature_names[i] for i in zero_var_cols]}")
            # This is problematic for StandardScaler. Consider removing these features or handling them.

        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Debugging: Check for NaNs or Infs after scaling
        if np.any(np.isnan(X_train_scaled)):
            logger.error("NaNs found in X_train_scaled AFTER scaling!")
        if np.any(np.isnan(X_test_scaled)):
            logger.error("NaNs found in X_test_scaled AFTER scaling!")
        
        feature_importance = {}

        if self.model_type == "random_forest":
            self.model = RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_split=4,
                min_samples_leaf=1,
                max_features='sqrt',
                bootstrap=True,
                random_state=42,
                class_weight='balanced',
                n_jobs=-1
            )
            self.model.fit(X_train_scaled, y_train)
            if hasattr(self.model, 'feature_importances_'):
                feature_importance = dict(zip(self.feature_names, 
                                            self.model.feature_importances_))
        elif self.model_type == "mlp":
            self.model = MLPClassifier(
                hidden_layer_sizes=(100, 50),
                activation='relu',
                solver='adam',
                alpha=0.0001,
                batch_size='auto',
                learning_rate='constant',
                learning_rate_init=0.001,
                max_iter=300,
                shuffle=True,
                random_state=42,
                early_stopping=True,
                n_iter_no_change=10,
                verbose=False
            )
            self.model.fit(X_train_scaled, y_train)
            logger.info("MLP model trained. Feature importance is not directly available for MLP like Random Forest.")
        else:
            raise ValueError(f"Unsupported model_type: {self.model_type}")

        # Evaluate model
        y_pred = self.model.predict(X_test_scaled)
        
        # Calculate metrics
        # Note: y_test and y_pred are numerically encoded here.
        # For a human-readable report, you might want to decode them using self.label_encoder.inverse_transform
        # However, classification_report handles integer labels correctly.
        # If class names are needed in the report keys, they can be obtained from self.label_encoder.classes_
        
        target_names_for_report = None
        if hasattr(self.label_encoder, 'classes_') and len(self.label_encoder.classes_) > 0 :
            try:
                # Ensure y_test and y_pred are within the range of learned labels
                # This is important if filtering removed all samples of some original classes
                # that might still be present in label_encoder.classes_
                unique_test_labels = np.unique(y_test)
                unique_pred_labels = np.unique(y_pred)
                max_label_in_data = max(unique_test_labels.max(), unique_pred_labels.max())

                if max_label_in_data < len(self.label_encoder.classes_):
                    # Filter target_names to only those present in y_test/y_pred if using integer labels
                    # Or, pass all self.label_encoder.classes_ if they match the full set of possible labels
                    # For simplicity, let's try to use all learned class names if possible.
                    # sklearn's classification_report can also take labels=np.unique(np.concatenate((y_test, y_pred)))
                    # to determine which labels to show.
                    target_names_for_report = self.label_encoder.classes_
                    logger.info(f"Using target names for classification report: {target_names_for_report}")
                else:
                    logger.warning(f"Max label in y_test/y_pred ({max_label_in_data}) exceeds "
                                   f"label_encoder classes count ({len(self.label_encoder.classes_)}). "
                                   "Not using target_names for report.")
            except Exception as e:
                logger.warning(f"Could not determine target names for classification_report: {e}")


        class_report = classification_report(y_test, y_pred, output_dict=True, zero_division=0, labels=np.arange(len(self.label_encoder.classes_)), target_names=target_names_for_report if target_names_for_report is not None else None)
        conf_matrix = confusion_matrix(y_test, y_pred, labels=np.arange(len(self.label_encoder.classes_)))
        
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
        predicted_class_encoded = self.model.predict(X_scaled)[0]
        
        # Decode the predicted class back to its original string label
        predicted_class_str = "unknown"
        if hasattr(self.label_encoder, 'classes_') and len(self.label_encoder.classes_) > 0:
            try:
                predicted_class_str = self.label_encoder.inverse_transform([predicted_class_encoded])[0]
            except ValueError as e:
                logger.error(f"Error decoding predicted class {predicted_class_encoded}: {e}. Known classes: {self.label_encoder.classes_}")
        else:
            logger.warning("LabelEncoder not fitted or has no classes; cannot decode predicted class.")
            # Fallback or error if classes_ is not available from label_encoder
            # This might happen if an old model is loaded without a label_encoder or if training had no labels
            # For now, we return the encoded class as a string or a placeholder.
            predicted_class_str = str(predicted_class_encoded) # Or some default like "unknown_encoded_class"

        class_probs_encoded = self.model.predict_proba(X_scaled)[0]
        class_probs_str_keys = {}
        if hasattr(self.label_encoder, 'classes_') and len(self.label_encoder.classes_) == len(class_probs_encoded):
            class_probs_str_keys = dict(zip(self.label_encoder.classes_, class_probs_encoded))
        else:
            logger.warning("LabelEncoder classes not available or mismatch with probability array size. "
                           "Returning probabilities with encoded class keys.")
            # Fallback: use encoded class indices as keys if string names aren't available or don't match
            class_probs_str_keys = dict(zip(map(str, self.model.classes_), class_probs_encoded))
        
        return predicted_class_str, class_probs_str_keys
    
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
                'label_encoder': self.label_encoder,
                'feature_names': self.feature_names,
                'model_type': self.model_type
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
                # Load label_encoder, with fallback for older models
                if 'label_encoder' in data:
                    self.label_encoder = data['label_encoder']
                else:
                    logger.warning("LabelEncoder not found in model file. Initializing a new one. "
                                   "Retrain model for full label functionality.")
                    self.label_encoder = LabelEncoder()
                self.feature_names = data['feature_names']
                self.model_type = data.get('model_type', 'random_forest')
            
            logger.info(f"Model ({self.model_type}) loaded from {model_path}")
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
    # Initialize with the desired default type (Random Forest for better imbalance handling)
    classifier = DrumClassifierML(model_type="random_forest") 
    model_path = get_default_model_path()
    
    if os.path.exists(model_path):
        logger.info(f"Attempting to load existing model from: {model_path}")
        # Attempt to load; the load method sets the classifier.model_type from the file.
        if classifier.load(model_path):
            logger.info(f"Successfully loaded model of type: {classifier.model_type} from {model_path}")
            # If the loaded model is not the desired default type (e.g., an old Random Forest),
            # and we want to enforce the new default, we could re-initialize `classifier` here.
            # For now, we use what's loaded. If the user wants to switch to MLP, they should retrain.
            # If classifier.load() fails, it returns False, and we proceed with the fresh MLP instance.
        else:
            logger.warning(f"Failed to load model from {model_path}. A new {classifier.model_type} classifier will be used.")
            # `classifier` is already the desired new MLP instance initialized above.
    else:
        logger.info(f"No existing model found at {model_path}. Creating a new {classifier.model_type} classifier.")
        # `classifier` is already the desired new MLP instance initialized above.
        
    return classifier


# Example Usage (for testing or standalone script)
if __name__ == "__main__":
    pass # Intentionally empty block after removing synthetic data generation 