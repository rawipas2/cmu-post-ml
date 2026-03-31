"""
Naive Bayes Model with GPU acceleration (cuML)
Falls back to scikit-learn CPU version if cuML is not available
"""
import numpy as np
import torch
from sklearn.naive_bayes import MultinomialNB
import pickle
import os
import config

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False

class NaiveBayesModel:
    """Naive Bayes with GPU acceleration via CuPy"""
    
    def __init__(self, config, alpha=1.0):
        self.config = config
        self.device = config.DEVICE
        
        if CUPY_AVAILABLE and torch.cuda.is_available():
            print("   Using GPU (CuPy + sklearn)")
            self.use_gpu = True
        else:
            print("   Using CPU (sklearn)")
            self.use_gpu = False
        
        self.model = MultinomialNB(alpha=alpha)
    
    def _to_gpu(self, X):
        if self.use_gpu and isinstance(X, np.ndarray):
            return cp.asarray(X)
        return X
    
    def _to_cpu(self, X):
        if self.use_gpu and isinstance(X, cp.ndarray):
            return cp.asnumpy(X)
        return X
    
    def train(self, X_train, y_train):
        print(f"\n🟡 Training Naive Bayes...")
        
        # Convert to dense if sparse
        if hasattr(X_train, 'toarray'):
            X_train = X_train.toarray()
        
        # Naive Bayes requires non-negative features
        # Shift to make all values positive
        if self.use_gpu:
            print("   GPU preprocessing...")
            X_gpu = self._to_gpu(X_train)
            # Shift to positive values
            min_val = cp.min(X_gpu)
            if min_val < 0:
                X_gpu = X_gpu - min_val + 1e-10
            else:
                X_gpu = X_gpu + 1e-10
            X_train = self._to_cpu(X_gpu)
        else:
            min_val = np.min(X_train)
            if min_val < 0:
                X_train = X_train - min_val + 1e-10
            else:
                X_train = X_train + 1e-10
        
        print("   Fitting Naive Bayes model...")
        self.model.fit(X_train, y_train)
        print("   ✅ Training complete")
    
    def predict(self, X):
        if hasattr(X, 'toarray'):
            X = X.toarray()
        
        if self.use_gpu:
            X_gpu = self._to_gpu(X)
            min_val = cp.min(X_gpu)
            if min_val < 0:
                X_gpu = X_gpu - min_val + 1e-10
            else:
                X_gpu = X_gpu + 1e-10
            X = self._to_cpu(X_gpu)
        else:
            min_val = np.min(X)
            if min_val < 0:
                X = X - min_val + 1e-10
            else:
                X = X + 1e-10
        
        return self.model.predict(X)
    
    def predict_proba(self, X):
        if hasattr(X, 'toarray'):
            X = X.toarray()
        
        if self.use_gpu:
            X_gpu = self._to_gpu(X)
            min_val = cp.min(X_gpu)
            if min_val < 0:
                X_gpu = X_gpu - min_val + 1e-10
            else:
                X_gpu = X_gpu + 1e-10
            X = self._to_cpu(X_gpu)
        else:
            min_val = np.min(X)
            if min_val < 0:
                X = X - min_val + 1e-10
            else:
                X = X + 1e-10
        
        return self.model.predict_proba(X)
    
    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self.model, f)
    
    def load(self, path):
        with open(path, 'rb') as f:
            self.model = pickle.load(f)


def create_model(config_module=config, **kwargs):
    """Factory function for compatibility with quick tests."""
    return NaiveBayesModel(config_module, **kwargs)
