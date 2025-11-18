"""
Support Vector Machine Model with GPU acceleration (cuML)
Falls back to scikit-learn CPU version if cuML is not available
"""
import numpy as np
import torch
import pickle
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.gpu_utils import (
    HAS_RAPIDS, get_svc_model, to_gpu_array, to_cpu_array
)
import config


class SVMModel:
    """SVM model using cuML for GPU acceleration (with CPU fallback)"""
    
    def __init__(self, kernel='rbf', C=1.0, gamma='scale'):
        """
        Initialize SVM model
        
        Args:
            kernel: Kernel type ('rbf', 'linear', 'poly', 'sigmoid')
            C: Regularization parameter
            gamma: Kernel coefficient
        """
        self.model = get_svc_model(
            kernel=kernel,
            C=C,
            gamma=gamma,
            probability=True,
            random_state=config.RANDOM_STATE
        )
        self.model_name = 'SVM'
        if not HAS_RAPIDS:
            self.model_name += '_CPU'
        
    def train(self, X_train, y_train, X_valid=None, y_valid=None):
        """Train the SVM model"""
        print(f"\n{'='*60}")
        print(f"🚀 Training {self.model_name}...")
        if HAS_RAPIDS:
            print("   Using GPU acceleration (cuML)")
        else:
            print("   Using CPU (scikit-learn)")
        print(f"{'='*60}")
        
        # Convert to appropriate array type
        X_train_gpu = to_gpu_array(X_train)
        y_train_gpu = to_gpu_array(y_train)
        
        # Train model
        self.model.fit(X_train_gpu, y_train_gpu)
        
        print(f"✅ {self.model_name} training completed!")
        
        # Evaluate on validation set if provided
        if X_valid is not None and y_valid is not None:
            X_valid_gpu = to_gpu_array(X_valid)
            train_acc = self.model.score(X_train_gpu, y_train_gpu)
            valid_acc = self.model.score(X_valid_gpu, to_gpu_array(y_valid))
            print(f"   Training Accuracy: {train_acc:.4f}")
            print(f"   Validation Accuracy: {valid_acc:.4f}")
    
    def predict(self, X):
        """Make predictions"""
        X_gpu = to_gpu_array(X)
        predictions = self.model.predict(X_gpu)
        return to_cpu_array(predictions)
    
    def predict_proba(self, X):
        """Predict class probabilities"""
        X_gpu = to_gpu_array(X)
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
    """Factory function to create SVM model"""
    return SVMModel(**kwargs)
