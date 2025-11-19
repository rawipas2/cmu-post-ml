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
MAX_FEATURES = 10000  # เพิ่มจาก 5000 สำหรับ v1.2.1 - ความละเอียดขึ้น
MAX_SEQ_LENGTH = 100
EMBEDDING_DIM = 300
BATCH_SIZE = 16  # ลดจาก 32 เพื่อให้ model เรียนรู้ดีขึ้นจาก features มากขึ้น
EPOCHS = 150  # เพิ่มจาก 100 เพื่อให้มีเวลาเรียนรู้มากขึ้น (มี early stopping)
LEARNING_RATE = 0.0003  # ลดลงจาก 0.0005 เพื่อ stability

# Ensemble parameters
META_MODEL_EPOCHS = 200  # เพิ่มจาก 150
META_MODEL_LR = 0.0003  # ลดจาก 0.0005 เพื่อ stability
META_HIDDEN_DIM = 256  # เพิ่มจาก 128 เพื่อ capacity มากขึ้น

# Version management
CURRENT_VERSION = "v1.2.1"

# Target accuracy
TARGET_ACCURACY = 0.80

# Model-specific hyperparameters
MODEL_PARAMS = {
    'neural_network': {
        'hidden_dims': [1024, 512, 256],  # เพิ่ม capacity เพื่อรองรับ features 10000
        'dropout': 0.5,
        'epochs': 150,
        'learning_rate': 0.0002  # ลดลงเพื่อ stability
    },
    'deep_learning': {
        'hidden_dims': [2048, 1024, 512, 256, 128],  # เพิ่ม capacity
        'dropout': 0.6,
        'epochs': 150,
        'learning_rate': 0.00015
    },
    'bayesian_network': {
        'hidden_dims': [512, 256, 128],  # เพิ่ม capacity
        'epochs': 150,
        'learning_rate': 0.0002
    },
    'maximum_entropy': {
        'l2_reg': 0.1,
        'epochs': 150,
        'learning_rate': 0.002,
        'use_class_weight': True
    },
    'svm': {
        'kernel': 'linear',
        'C': 1.0,
        'gamma': 'scale',
        'use_sgd': True
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
