"""
Naive Bayes Model with GPU acceleration (cuML)
"""
import numpy as np
import pickle
import os
from cuml.naive_bayes import MultinomialNB
import config


class NaiveBayesModel:
    """Naive Bayes model using cuML for GPU acceleration"""
    
    def __init__(self, alpha=1.0):
        """
        Initialize Naive Bayes model
        
        Args:
            alpha: Additive (Laplace/Lidstone) smoothing parameter
        """
        self.model = MultinomialNB(alpha=alpha)
        self.model_name = 'Naive_Bayes'
        
    def train(self, X_train, y_train, X_valid=None, y_valid=None):
        """Train the Naive Bayes model"""
        print(f"\n{'='*60}")
        print(f"🚀 Training {self.model_name} on GPU...")
        print(f"{'='*60}")
        
        # Convert to cuPy arrays for GPU
        import cupy as cp
        
        # Ensure all values are non-negative for MultinomialNB
        X_train_pos = np.abs(X_train)
        X_train_gpu = cp.asarray(X_train_pos)
        y_train_gpu = cp.asarray(y_train)
        
        # Train model
        self.model.fit(X_train_gpu, y_train_gpu)
        
        print(f"✅ {self.model_name} training completed!")
        
        # Evaluate on validation set if provided
        if X_valid is not None and y_valid is not None:
            X_valid_pos = np.abs(X_valid)
            X_valid_gpu = cp.asarray(X_valid_pos)
            train_acc = self.model.score(X_train_gpu, y_train_gpu)
            valid_acc = self.model.score(X_valid_gpu, cp.asarray(y_valid))
            print(f"   Training Accuracy: {train_acc:.4f}")
            print(f"   Validation Accuracy: {valid_acc:.4f}")
    
    def predict(self, X):
        """Make predictions"""
        import cupy as cp
        X_pos = np.abs(X)
        X_gpu = cp.asarray(X_pos)
        predictions = self.model.predict(X_gpu)
        return cp.asnumpy(predictions)
    
    def predict_proba(self, X):
        """Predict class probabilities"""
        import cupy as cp
        X_pos = np.abs(X)
        X_gpu = cp.asarray(X_pos)
        probas = self.model.predict_proba(X_gpu)
        # Return probability of positive class
        return cp.asnumpy(probas[:, 1])
    
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
