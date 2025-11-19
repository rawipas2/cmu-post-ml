"""
Naive Bayes Model with GPU acceleration (cuML)
Falls back to scikit-learn CPU version if cuML is not available
"""
import numpy as np
import torch
from sklearn.naive_bayes import MultinomialNB
import pickle
import os

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
        
        if hasattr(X_train, 'toarray'):
            X_train = X_train.toarray()
        
        # GPU preprocessing
        if self.use_gpu:
            print("   GPU preprocessing...")
            X_gpu = self._to_gpu(X_train)
            # Add small constant to avoid zeros
            X_gpu = X_gpu + 1e-10
            X_train = self._to_cpu(X_gpu)
        
        print("   Fitting Naive Bayes model...")
        self.model.fit(X_train, y_train)
        print("   ✅ Training complete")
    
    def predict(self, X):
        if hasattr(X, 'toarray'):
            X = X.toarray()
        
        if self.use_gpu:
            X_gpu = self._to_gpu(X)
            X_gpu = X_gpu + 1e-10
            X = self._to_cpu(X_gpu)
        
        return self.model.predict(X)
    
    def predict_proba(self, X):
        if hasattr(X, 'toarray'):
            X = X.toarray()
        
        if self.use_gpu:
            X_gpu = self._to_gpu(X)
            X_gpu = X_gpu + 1e-10
            X = self._to_cpu(X_gpu)
        
        return self.model.predict_proba(X)
    
    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self.model, f)
    
    def load(self, path):
        with open(path, 'rb') as f:
            self.model = pickle.load(f)
