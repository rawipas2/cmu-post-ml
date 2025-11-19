"""
Configuration file for Thai Depression Classification Project
"""
import os
import torch

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'saved_models')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
VERSIONS_DIR = os.path.join(BASE_DIR, 'versions')

# Data files
TRAIN_FILE = os.path.join(DATA_DIR, 'train.json')
TEST_FILE = os.path.join(DATA_DIR, 'test.json')
VALID_FILE = os.path.join(DATA_DIR, 'valid.json')

# Device configuration - Try CUDA, fallback to CPU if needed
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
if DEVICE != 'cuda':
    import warnings
    warnings.warn(
        "\n⚠️  CUDA is not available! Running on CPU. "
        "\nFor GPU acceleration, ensure NVIDIA drivers and CUDA are installed."
    )
else:
    print(f"🚀 Configuration loaded. Using device: {DEVICE}")
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
RANDOM_STATE = 42
MAX_FEATURES = 5000
MAX_SEQ_LENGTH = 100
EMBEDDING_DIM = 300
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 0.0005

# Ensemble parameters
META_MODEL_EPOCHS = 150
META_MODEL_LR = 0.0005
META_HIDDEN_DIM = 128

# Version management
CURRENT_VERSION = "v1.1"

# Target accuracy
TARGET_ACCURACY = 0.80

# Model-specific hyperparameters
MODEL_PARAMS = {
    'neural_network': {
        'hidden_dims': [512, 256, 128],
        'dropout': 0.4,
        'epochs': 100,
        'learning_rate': 0.0005
    },
    'deep_learning': {
        'hidden_dims': [1024, 512, 256, 128, 64],
        'dropout': 0.5,
        'epochs': 100,
        'learning_rate': 0.0003
    },
    'bayesian_network': {
        'hidden_dims': [256, 128, 64],
        'epochs': 100,
        'learning_rate': 0.0005
    },
    'maximum_entropy': {
        'l2_reg': 0.01,
        'epochs': 100,
        'learning_rate': 0.001
    },
    'svm': {
        'kernel': 'linear',  # linear เร็วกว่า rbf มาก
        'C': 1.0,
        'gamma': 'scale',
        'use_sgd': True  # ใช้ SGDClassifier (เร็วที่สุดสำหรับ large dataset)
    },
    'naive_bayes': {
        'alpha': 0.5
    }
}

# Model names
MODELS = [
    'svm',
    'neural_network',
    'deep_learning',
    'naive_bayes',
    'bayesian_network',
    'maximum_entropy'
]

print(f"🚀 Configuration loaded. Using device: {DEVICE}")
if DEVICE == 'cuda':
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
