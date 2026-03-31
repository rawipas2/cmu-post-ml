"""
Support Vector Machine model with optional SGD backend.
"""
import os
import pickle
import warnings

import numpy as np
import torch
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC

import config

warnings.filterwarnings('ignore')

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False


class SVMModel:
    """Support Vector Machine with GPU-aware preprocessing hooks."""

    def __init__(self, config, kernel='linear', C=1.0, gamma='scale', use_sgd=True):
        self.config = config
        self.device = config.DEVICE
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.use_sgd = use_sgd

        if CUPY_AVAILABLE and torch.cuda.is_available():
            print("   Using GPU (CuPy for preprocessing)")
            self.use_gpu = True
        else:
            print("   Using CPU (NumPy)")
            self.use_gpu = False

        self.mean_ = None
        self.std_ = None
        self.calibrated_model = None

        if use_sgd:
            print("   Using SGDClassifier (faster for large datasets)")
            self.model = SGDClassifier(
                loss='hinge',
                penalty='l2',
                alpha=1.0 / C if C > 0 else 0.0001,
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

    def _to_gpu(self, X):
        if self.use_gpu and isinstance(X, np.ndarray):
            return cp.asarray(X)
        return X

    def _to_cpu(self, X):
        if self.use_gpu and isinstance(X, cp.ndarray):
            return cp.asnumpy(X)
        return X

    def train(self, X_train, y_train):
        """Train the SVM."""
        print(f"\n๐”ต Training SVM...")
        print(f"   Dataset size: {X_train.shape}")

        if hasattr(X_train, 'toarray'):
            print("   Converting sparse to dense...")
            X_train = X_train.toarray()

        print(f"   Fitting SVM model ({'SGD' if self.use_sgd else 'LinearSVC'})...")
        print("   (This may take a few minutes...)")
        self.model.fit(X_train, y_train)

        if not self.use_sgd:
            print("   Calibrating for probability estimates...")
            self.calibrated_model = CalibratedClassifierCV(self.model, cv=3)
            subset_size = min(5000, len(X_train))
            indices = np.random.choice(len(X_train), subset_size, replace=False)
            self.calibrated_model.fit(X_train[indices], y_train[indices])

        print("   โ… Training complete")

    def decision_function(self, X):
        if hasattr(X, 'toarray'):
            X = X.toarray()
        return self.model.decision_function(X)

    def predict(self, X):
        if hasattr(X, 'toarray'):
            X = X.toarray()
        return self.model.predict(X)

    def predict_proba(self, X):
        if hasattr(X, 'toarray'):
            X = X.toarray()

        if self.calibrated_model is not None:
            return self.calibrated_model.predict_proba(X)

        decision = self.model.decision_function(X)
        proba_positive = 1.0 / (1.0 + np.exp(-decision))
        return np.column_stack([1.0 - proba_positive, proba_positive])

    def save(self, path):
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
        with open(path, 'wb') as handle:
            pickle.dump(save_dict, handle)

    def load(self, path):
        with open(path, 'rb') as handle:
            save_dict = pickle.load(handle)

        if isinstance(save_dict, dict):
            self.model = save_dict['model']
            self.calibrated_model = save_dict.get('calibrated_model')
            self.mean_ = self._to_gpu(save_dict['mean']) if self.use_gpu else save_dict['mean']
            self.std_ = self._to_gpu(save_dict['std']) if self.use_gpu else save_dict['std']
            self.kernel = save_dict.get('kernel', 'linear')
            self.C = save_dict.get('C', 1.0)
            self.gamma = save_dict.get('gamma', 'scale')
            self.use_sgd = save_dict.get('use_sgd', True)
        else:
            self.model = save_dict
            self.calibrated_model = None


def create_model(config_module=config, **kwargs):
    """Factory function for compatibility with quick tests."""
    return SVMModel(config_module, **kwargs)
