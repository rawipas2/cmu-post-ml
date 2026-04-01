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
MAX_FEATURES = 5000  # ย้อนกลับจาก 10000 - v1.2.2 เน้น model-specific preprocessing
MAX_SEQ_LENGTH = 100
EMBEDDING_DIM = 300
BATCH_SIZE = 32  # เพิ่มกลับเป็น 32 เพื่อความเร็ว
EPOCHS = 100  # ลดกลับเป็น 100 พร้อม early stopping
LEARNING_RATE = 0.0005  # เพิ่มกลับเพื่อ faster convergence

# Ensemble parameters
META_MODEL_EPOCHS = 150  # ปรับกลับ
META_MODEL_LR = 0.0005  # ปรับกลับ
META_HIDDEN_DIM = 128  # ปรับกลับ

# Version management
CURRENT_VERSION = "v2.0"

# Target accuracy
TARGET_ACCURACY = 0.80

# Model-specific hyperparameters
MODEL_PARAMS = {
    'neural_network': {
        'hidden_dims': [512, 256, 128],  # ย้อนกลับ v1.2.2 - เหมาะกับ 5000 features
        'dropout': 0.5,
        'epochs': 100,
        'learning_rate': 0.0003,
        'use_focal_loss': True  # ใหม่! แก้ class imbalance
    },
    'deep_learning': {
        'hidden_dims': [1024, 512, 256, 128, 64],  # ย้อนกลับ
        'dropout': 0.6,
        'epochs': 100,
        'learning_rate': 0.0002,
        'use_focal_loss': True  # ใหม่!
    },
    'bayesian_network': {
        'hidden_dims': [256, 128, 64],  # ย้อนกลับ
        'epochs': 100,
        'learning_rate': 0.0003,
        'use_focal_loss': True  # ใหม่!
    },
    'maximum_entropy': {
        'l2_reg': 0.1,
        'epochs': 100,
        'learning_rate': 0.002,
        'use_class_weight': True,
        'use_focal_loss': True  # ใหม่!
    },
    'svm': {
        'kernel': 'linear',
        'C': 1.0,
        'gamma': 'scale',
        'use_sgd': True,
        'n_features_select': 3000  # ใหม่! Feature selection
    },
    'naive_bayes': {
        'alpha': 0.5,
        'use_count': True  # ใหม่! Use Count instead of TF-IDF
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
