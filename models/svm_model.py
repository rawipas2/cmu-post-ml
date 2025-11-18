"""
Support Vector Machine Model with GPU acceleration (CuPy + scikit-learn)
Falls back to scikit-learn CPU version if CuPy is not available
"""
import numpy as np
import torch
from sklearn.svm import SVC
import pickle
import os

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False


class SVMModel:
    """Support Vector Machine with GPU acceleration via CuPy"""
    
    def __init__(self, config):
        self.config = config
        self.device = config.DEVICE
        
        # ใช้ CuPy ถ้ามี, ถ้าไม่มีใช้ CPU
        if CUPY_AVAILABLE and torch.cuda.is_available():
            print("   Using GPU (CuPy + sklearn)")
            self.use_gpu = True
        else:
            print("   Using CPU (sklearn)")
            self.use_gpu = False
        
        # SVM parameters
        self.model = SVC(
            kernel='rbf',
            C=1.0,
            gamma='scale',
            probability=True,
            cache_size=1000,
            verbose=False,
            max_iter=1000
        )
    
    def _to_gpu(self, X):
        """Convert to CuPy array for GPU"""
        if self.use_gpu and isinstance(X, np.ndarray):
            return cp.asarray(X)
        return X
    
    def _to_cpu(self, X):
        """Convert back to numpy for sklearn"""
        if self.use_gpu and isinstance(X, cp.ndarray):
            return cp.asnumpy(X)
        return X
    
    def train(self, X_train, y_train):
        """Train SVM with GPU preprocessing"""
        print(f"\n🔵 Training SVM...")
        
        # Convert to dense if sparse
        if hasattr(X_train, 'toarray'):
            X_train = X_train.toarray()
        
        # GPU acceleration for preprocessing
        if self.use_gpu:
            print("   GPU preprocessing...")
            X_gpu = self._to_gpu(X_train)
            # Normalize on GPU
            mean = cp.mean(X_gpu, axis=0)
            std = cp.std(X_gpu, axis=0) + 1e-8
            X_gpu = (X_gpu - mean) / std
            X_train = self._to_cpu(X_gpu)
        
        # Train on CPU (sklearn SVM)
        print("   Fitting SVM model...")
        self.model.fit(X_train, y_train)
        print("   ✅ Training complete")
    
    def predict(self, X):
        """Predict with GPU preprocessing"""
        if hasattr(X, 'toarray'):
            X = X.toarray()
        
        # GPU preprocessing
        if self.use_gpu:
            X_gpu = self._to_gpu(X)
            mean = cp.mean(X_gpu, axis=0)
            std = cp.std(X_gpu, axis=0) + 1e-8
            X_gpu = (X_gpu - mean) / std
            X = self._to_cpu(X_gpu)
        
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Predict probabilities with GPU preprocessing"""
        if hasattr(X, 'toarray'):
            X = X.toarray()
        
        # GPU preprocessing
        if self.use_gpu:
            X_gpu = self._to_gpu(X)
            mean = cp.mean(X_gpu, axis=0)
            std = cp.std(X_gpu, axis=0) + 1e-8
            X_gpu = (X_gpu - mean) / std
            X = self._to_cpu(X_gpu)
        
        return self.model.predict_proba(X)
    
    def save(self, path):
        """Save model"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self.model, f)
    
    def load(self, path):
        """Load model"""
        with open(path, 'rb') as f:
            self.model = pickle.load(f)
