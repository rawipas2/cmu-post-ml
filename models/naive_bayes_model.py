"""
Naive Bayes Model with GPU acceleration (cuML)
Falls back to scikit-learn CPU version if cuML is not available
"""
import numpy as np
import pickle
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.gpu_utils import (
    HAS_RAPIDS, get_naive_bayes_model, to_gpu_array, to_cpu_array
)
import config


class NaiveBayesModel:
    """Naive Bayes model using cuML for GPU acceleration (with CPU fallback)"""
    
    def __init__(self, alpha=1.0):
        """
        Initialize Naive Bayes model
        
        Args:
            alpha: Additive (Laplace/Lidstone) smoothing parameter
        """
        self.model = get_naive_bayes_model(alpha=alpha)
        self.model_name = 'Naive_Bayes'
        if not HAS_RAPIDS:
            self.model_name += '_CPU'
        
    def train(self, X_train, y_train, X_valid=None, y_valid=None):
        """Train the Naive Bayes model"""
        print(f"\n{'='*60}")
        print(f"🚀 Training {self.model_name}...")
        if HAS_RAPIDS:
            print("   Using GPU acceleration (cuML)")
        else:
            print("   Using CPU (scikit-learn)")
        print(f"{'='*60}")
        
        # Ensure all values are non-negative for MultinomialNB
        X_train_pos = np.abs(X_train)
        X_train_gpu = to_gpu_array(X_train_pos)
        y_train_gpu = to_gpu_array(y_train)
        
        # Train model
        self.model.fit(X_train_gpu, y_train_gpu)
        
        print(f"✅ {self.model_name} training completed!")
        
        # Evaluate on validation set if provided
        if X_valid is not None and y_valid is not None:
            X_valid_pos = np.abs(X_valid)
            X_valid_gpu = to_gpu_array(X_valid_pos)
            train_acc = self.model.score(X_train_gpu, y_train_gpu)
            valid_acc = self.model.score(X_valid_gpu, to_gpu_array(y_valid))
            print(f"   Training Accuracy: {train_acc:.4f}")
            print(f"   Validation Accuracy: {valid_acc:.4f}")
    
    def predict(self, X):
        """Make predictions"""
        X_pos = np.abs(X)
        X_gpu = to_gpu_array(X_pos)
        predictions = self.model.predict(X_gpu)
        return to_cpu_array(predictions)
    
    def predict_proba(self, X):
        """Predict class probabilities"""
        X_pos = np.abs(X)
        X_gpu = to_gpu_array(X_pos)
        probas = self.model.predict_proba(X_gpu)
        probas_cpu = to_cpu_array(probas)
        # Return probability of positive class
        if len(probas_cpu.shape) > 1 and probas_cpu.shape[1] > 1:
            return probas_cpu[:, 1]
        return probas_cpu.flatten()
    
    def save(self, filepath):
        """Save model to disk"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(self.model, f)
        print(f"💾 {self.model_name} saved to: {filepath}")
    
    def load(self, filepath):
        """Load model from disk"""
        with open(filepath, 'rb') as f:
            self.model = pickle.load(f)
        print(f"📂 {self.model_name} loaded from: {filepath}")


def create_model(**kwargs):
    """Factory function to create Naive Bayes model"""
    return NaiveBayesModel(**kwargs)
