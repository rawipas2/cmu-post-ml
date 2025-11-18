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

# Device configuration - Force CUDA usage
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
if DEVICE != 'cuda':
    raise RuntimeError("CUDA is not available! This project requires GPU.")

# Model parameters
RANDOM_STATE = 42
MAX_FEATURES = 5000
MAX_SEQ_LENGTH = 100
EMBEDDING_DIM = 300
BATCH_SIZE = 64
EPOCHS = 50
LEARNING_RATE = 0.001

# Ensemble parameters
META_MODEL_EPOCHS = 100
META_MODEL_LR = 0.001

# Version management
CURRENT_VERSION = "v1.0"

# Target accuracy
TARGET_ACCURACY = 0.80

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
