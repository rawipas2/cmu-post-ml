"""
Support Vector Machine Model with GPU acceleration (CuPy + scikit-learn)
Falls back to scikit-learn CPU version if CuPy is not available
Uses LinearSVC and SGDClassifier for faster training on large datasets
"""
import numpy as np
import torch
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.calibration import CalibratedClassifierCV
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False


class SVMModel:
    """Support Vector Machine with GPU acceleration via CuPy"""
    
    def __init__(self, config, kernel='linear', C=1.0, gamma='scale', use_sgd=True):
        self.config = config
        self.device = config.DEVICE
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.use_sgd = use_sgd
        
        # ใช้ CuPy ถ้ามี, ถ้าไม่มีใช้ CPU
        if CUPY_AVAILABLE and torch.cuda.is_available():
            print("   Using GPU (CuPy for preprocessing)")
            self.use_gpu = True
        else:
            print("   Using CPU (NumPy)")
            self.use_gpu = False
        
        # Store normalization parameters
        self.mean_ = None
        self.std_ = None
        
        # SVM model - ใช้ SGDClassifier สำหรับ dataset ใหญ่ (เร็วกว่ามาก)
        if use_sgd:
            print("   Using SGDClassifier (faster for large datasets)")
            self.model = SGDClassifier(
                loss='hinge',  # SVM loss
                penalty='l2',
                alpha=1.0/C if C > 0 else 0.0001,  # alpha = 1/C
                max_iter=1000,
                tol=1e-3,
                random_state=42,
                n_jobs=-1,
                verbose=1
            )
        else:
            print("   Using LinearSVC")
            self.model = LinearSVC(
                C=C,
                max_iter=1000,
                tol=1e-4,
                verbose=1,
                random_state=42
            )
        
        self.calibrated_model = None
    
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
        print(f"   Dataset size: {X_train.shape}")
        
        # Convert to dense if sparse
        if hasattr(X_train, 'toarray'):
            print("   Converting sparse to dense...")
            X_train = X_train.toarray()
        
        # Note: Normalization is now done in prepare_data_for_model (v1.2.2)
        # No need to normalize again here
        self.mean_ = None  # Not used anymore
        self.std_ = None
        
        # Train model
        print(f"   Fitting SVM model ({'SGD' if self.use_sgd else 'LinearSVC'})...")
        print("   (This may take a few minutes...)")
        
        self.model.fit(X_train, y_train)
        
        # Calibrate for probability estimates (only for LinearSVC)
        if not self.use_sgd:
            print("   Calibrating for probability estimates...")
            self.calibrated_model = CalibratedClassifierCV(self.model, cv=3)
            # Use subset for calibration to save time
            subset_size = min(5000, len(X_train))
            indices = np.random.choice(len(X_train), subset_size, replace=False)
            self.calibrated_model.fit(X_train[indices], y_train[indices])
        
        print("   ✅ Training complete")
    
    def predict(self, X):
        """Predict (no normalization needed - done in prepare_data_for_model)"""
        if hasattr(X, 'toarray'):
            X = X.toarray()
        
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Predict probabilities with GPU preprocessing"""
        if hasattr(X, 'toarray'):
            X = X.toarray()
        
        # GPU preprocessing with stored normalization
    def predict_proba(self, X):
        """Predict probabilities (no normalization needed)"""
        if hasattr(X, 'toarray'):
            X = X.toarray()
        
        # Use calibrated model for probabilities if available
        if self.use_sgd:
            # SGDClassifier has decision_function, convert to probabilities
            decision = self.model.decision_function(X)
            # Simple sigmoid calibration
            proba_positive = 1 / (1 + np.exp(-decision))
            return np.column_stack([1 - proba_positive, proba_positive])
        elif self.calibrated_model is not None:
            return self.calibrated_model.predict_proba(X)
        else:
            # Fallback: use decision function
            decision = self.model.decision_function(X)
            proba_positive = 1 / (1 + np.exp(-decision))
            return np.column_stack([1 - proba_positive, proba_positive])
    
    def save(self, path):
        """Save model"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        save_dict = {
            'model': self.model,
            'calibrated_model': self.calibrated_model,
            'mean': self._to_cpu(self.mean_) if self.use_gpu and self.mean_ is not None else self.mean_,
            'std': self._to_cpu(self.std_) if self.use_gpu and self.std_ is not None else self.std_,
            'kernel': self.kernel,
            'C': self.C,
            'gamma': self.gamma,
            'use_sgd': self.use_sgd
        }
        with open(path, 'wb') as f:
            pickle.dump(save_dict, f)
    
    def load(self, path):
        """Load model"""
        with open(path, 'rb') as f:
            save_dict = pickle.load(f)
        
        if isinstance(save_dict, dict):
            self.model = save_dict['model']
            self.calibrated_model = save_dict.get('calibrated_model', None)
            self.mean_ = self._to_gpu(save_dict['mean']) if self.use_gpu else save_dict['mean']
            self.std_ = self._to_gpu(save_dict['std']) if self.use_gpu else save_dict['std']
            self.kernel = save_dict.get('kernel', 'linear')
            self.C = save_dict.get('C', 1.0)
            self.gamma = save_dict.get('gamma', 'scale')
            self.use_sgd = save_dict.get('use_sgd', True)
        else:
            # Backward compatibility
            self.model = save_dict
            self.calibrated_model = None
