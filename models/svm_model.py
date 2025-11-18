"""
Support Vector Machine Model with GPU acceleration (cuML)
"""
import numpy as np
import torch
import pickle
import os
from cuml.svm import SVC
import config


class SVMModel:
    """SVM model using cuML for GPU acceleration"""
    
    def __init__(self, kernel='rbf', C=1.0, gamma='scale'):
        """
        Initialize SVM model
        
        Args:
            kernel: Kernel type ('rbf', 'linear', 'poly', 'sigmoid')
            C: Regularization parameter
            gamma: Kernel coefficient
        """
        self.model = SVC(
            kernel=kernel,
            C=C,
            gamma=gamma,
            probability=True,
            random_state=config.RANDOM_STATE
        )
        self.model_name = 'SVM'
        
    def train(self, X_train, y_train, X_valid=None, y_valid=None):
        """Train the SVM model"""
        print(f"\n{'='*60}")
        print(f"🚀 Training {self.model_name} on GPU...")
        print(f"{'='*60}")
        
        # Convert to cuDF/cuPy if needed
        import cupy as cp
        X_train_gpu = cp.asarray(X_train)
        y_train_gpu = cp.asarray(y_train)
        
        # Train model
        self.model.fit(X_train_gpu, y_train_gpu)
        
        print(f"✅ {self.model_name} training completed!")
        
        # Evaluate on validation set if provided
        if X_valid is not None and y_valid is not None:
            X_valid_gpu = cp.asarray(X_valid)
            train_acc = self.model.score(X_train_gpu, y_train_gpu)
            valid_acc = self.model.score(X_valid_gpu, cp.asarray(y_valid))
            print(f"   Training Accuracy: {train_acc:.4f}")
            print(f"   Validation Accuracy: {valid_acc:.4f}")
    
    def predict(self, X):
        """Make predictions"""
        import cupy as cp
        X_gpu = cp.asarray(X)
        predictions = self.model.predict(X_gpu)
        return cp.asnumpy(predictions)
    
    def predict_proba(self, X):
        """Predict class probabilities"""
        import cupy as cp
        X_gpu = cp.asarray(X)
        probas = self.model.predict_proba(X_gpu)
        return cp.asnumpy(probas)
    
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
