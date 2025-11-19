"""
Data loading and preprocessing utilities
"""
import json
import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import LabelEncoder
from pythainlp.tokenize import word_tokenize
from typing import Tuple, List
import config


class ThaiTextPreprocessor:
    """Preprocessor for Thai text data"""
    
    def __init__(self):
        self.label_encoder = LabelEncoder()
        self.tfidf_vectorizer = None
        self.count_vectorizer = None
        
    def load_data(self, filepath: str) -> Tuple[List[str], List[str]]:
        """Load data from JSON file"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        texts = [item[0] for item in data]
        labels = [item[1] for item in data]
        return texts, labels
    
    def tokenize_thai(self, text: str) -> str:
        """Tokenize Thai text"""
        tokens = word_tokenize(text, engine='newmm')
        return ' '.join(tokens)
    
    def preprocess_texts(self, texts: List[str]) -> List[str]:
        """Preprocess list of texts"""
        return [self.tokenize_thai(text) for text in texts]
    
    def fit_tfidf(self, texts: List[str]):
        """Fit TF-IDF vectorizer"""
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=config.MAX_FEATURES,
            ngram_range=(1, 2)
        )
        self.tfidf_vectorizer.fit(texts)
    
    def fit_count(self, texts: List[str]):
        """Fit Count vectorizer"""
        self.count_vectorizer = CountVectorizer(
            max_features=config.MAX_FEATURES,
            ngram_range=(1, 2)
        )
        self.count_vectorizer.fit(texts)
    
    def transform_tfidf(self, texts: List[str]) -> np.ndarray:
        """Transform texts using TF-IDF"""
        return self.tfidf_vectorizer.transform(texts).toarray()
    
    def transform_count(self, texts: List[str]) -> np.ndarray:
        """Transform texts using Count vectorizer"""
        return self.count_vectorizer.transform(texts).toarray()
    
    def encode_labels(self, labels: List[str], fit: bool = False) -> np.ndarray:
        """Encode labels to numerical values"""
        if fit:
            return self.label_encoder.fit_transform(labels)
        return self.label_encoder.transform(labels)
    
    def decode_labels(self, encoded_labels: np.ndarray) -> List[str]:
        """Decode numerical labels to original labels"""
        return self.label_encoder.inverse_transform(encoded_labels)


def load_all_data():
    """Load and preprocess all datasets"""
    preprocessor = ThaiTextPreprocessor()
    
    # Load data
    print("📂 Loading datasets...")
    train_texts, train_labels = preprocessor.load_data(config.TRAIN_FILE)
    test_texts, test_labels = preprocessor.load_data(config.TEST_FILE)
    valid_texts, valid_labels = preprocessor.load_data(config.VALID_FILE)
    
    print(f"   Train: {len(train_texts)} samples")
    print(f"   Valid: {len(valid_texts)} samples")
    print(f"   Test: {len(test_texts)} samples")
    
    # Preprocess texts
    print("🔤 Tokenizing Thai text...")
    train_texts_processed = preprocessor.preprocess_texts(train_texts)
    valid_texts_processed = preprocessor.preprocess_texts(valid_texts)
    test_texts_processed = preprocessor.preprocess_texts(test_texts)
    
    # Fit vectorizers on training data
    print("📊 Fitting vectorizers...")
    preprocessor.fit_tfidf(train_texts_processed)
    preprocessor.fit_count(train_texts_processed)
    
    # Transform texts
    print("🔄 Transforming texts...")
    X_train = preprocessor.transform_tfidf(train_texts_processed)
    X_valid = preprocessor.transform_tfidf(valid_texts_processed)
    X_test = preprocessor.transform_tfidf(test_texts_processed)
    
    # Encode labels
    print("🏷️  Encoding labels...")
    y_train = preprocessor.encode_labels(train_labels, fit=True)
    y_valid = preprocessor.encode_labels(valid_labels)
    y_test = preprocessor.encode_labels(test_labels)
    
    print("✅ Data loading complete!")
    
    return {
        'X_train': X_train,
        'X_valid': X_valid,
        'X_test': X_test,
        'y_train': y_train,
        'y_valid': y_valid,
        'y_test': y_test,
        'preprocessor': preprocessor,
        'train_texts': train_texts_processed,
        'valid_texts': valid_texts_processed,
        'test_texts': test_texts_processed
    }


def to_gpu_tensor(data: np.ndarray, dtype=torch.float32) -> torch.Tensor:
    """Convert numpy array to GPU tensor"""
    return torch.tensor(data, dtype=dtype).to(config.DEVICE)


def prepare_data_for_model(X, model_type='neural'):
    """
    Prepare data specifically for different model types
    
    Args:
        X: Input data (numpy array or sparse matrix)
        model_type: Type of model ('neural', 'svm', 'naive_bayes')
    
    Returns:
        Processed data suitable for the model
    """
    # Convert sparse to dense if needed
    if hasattr(X, 'toarray'):
        X = X.toarray()
    
    if model_type == 'naive_bayes':
        # Naive Bayes requires non-negative features
        # Shift all values to positive range
        min_val = np.min(X)
        if min_val < 0:
            X = X - min_val + 1e-10
        else:
            X = X + 1e-10
    
    elif model_type == 'svm':
        # SVM benefits from normalized features
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0) + 1e-8
        X = (X - mean) / std
    
    elif model_type in ['neural', 'deep', 'bayesian', 'maxent']:
        # Neural networks work better with standardized data
        # But TF-IDF is already normalized, so just ensure dtype
        X = X.astype(np.float32)
    
    return X
