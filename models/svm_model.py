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
    
    def __init__(self, config, kernel='rbf', C=1.0, gamma='scale'):
        self.config = config
        self.device = config.DEVICE
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        
        # ใช้ CuPy ถ้ามี, ถ้าไม่มีใช้ CPU
        if CUPY_AVAILABLE and torch.cuda.is_available():
            print("   Using GPU (CuPy + sklearn)")
            self.use_gpu = True
        else:
            print("   Using CPU (sklearn)")
            self.use_gpu = False
        
        # Store normalization parameters
        self.mean_ = None
        self.std_ = None
        
        # SVM parameters
        self.model = SVC(
            kernel=kernel,
            C=C,
            gamma=gamma,
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
            # Normalize on GPU and store parameters
            self.mean_ = cp.mean(X_gpu, axis=0)
            self.std_ = cp.std(X_gpu, axis=0) + 1e-8
            X_gpu = (X_gpu - self.mean_) / self.std_
            X_train = self._to_cpu(X_gpu)
        else:
            # CPU normalization
            self.mean_ = np.mean(X_train, axis=0)
            self.std_ = np.std(X_train, axis=0) + 1e-8
            X_train = (X_train - self.mean_) / self.std_
        
        # Train on CPU (sklearn SVM)
        print("   Fitting SVM model...")
        self.model.fit(X_train, y_train)
        print("   ✅ Training complete")
    
    def predict(self, X):
        """Predict with GPU preprocessing"""
        if hasattr(X, 'toarray'):
            X = X.toarray()
        
        # GPU preprocessing with stored normalization
        if self.use_gpu and self.mean_ is not None:
            X_gpu = self._to_gpu(X)
            X_gpu = (X_gpu - self.mean_) / self.std_
            X = self._to_cpu(X_gpu)
        elif self.mean_ is not None:
            X = (X - self._to_cpu(self.mean_) if self.use_gpu else self.mean_) / (self._to_cpu(self.std_) if self.use_gpu else self.std_)
        
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Predict probabilities with GPU preprocessing"""
        if hasattr(X, 'toarray'):
            X = X.toarray()
        
        # GPU preprocessing with stored normalization
        if self.use_gpu and self.mean_ is not None:
            X_gpu = self._to_gpu(X)
            X_gpu = (X_gpu - self.mean_) / self.std_
            X = self._to_cpu(X_gpu)
        elif self.mean_ is not None:
            X = (X - self._to_cpu(self.mean_) if self.use_gpu else self.mean_) / (self._to_cpu(self.std_) if self.use_gpu else self.std_)
        
        return self.model.predict_proba(X)
    
    def save(self, path):
        """Save model"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        save_dict = {
            'model': self.model,
            'mean': self._to_cpu(self.mean_) if self.use_gpu and self.mean_ is not None else self.mean_,
            'std': self._to_cpu(self.std_) if self.use_gpu and self.std_ is not None else self.std_,
            'kernel': self.kernel,
            'C': self.C,
            'gamma': self.gamma
        }
        with open(path, 'wb') as f:
            pickle.dump(save_dict, f)
    
    def load(self, path):
        """Load model"""
        with open(path, 'rb') as f:
            save_dict = pickle.load(f)
        
        if isinstance(save_dict, dict):
            self.model = save_dict['model']
            self.mean_ = self._to_gpu(save_dict['mean']) if self.use_gpu else save_dict['mean']
            self.std_ = self._to_gpu(save_dict['std']) if self.use_gpu else save_dict['std']
            self.kernel = save_dict.get('kernel', 'rbf')
            self.C = save_dict.get('C', 1.0)
            self.gamma = save_dict.get('gamma', 'scale')
        else:
            # Backward compatibility
            self.model = save_dict
