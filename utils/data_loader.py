"""
Data loading and preprocessing utilities
v1.2.2: Model-specific data preparation for optimal performance
"""
import json
import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import chi2, SelectKBest
from pythainlp.tokenize import word_tokenize
from typing import Tuple, List, Dict, Optional
import config
from .augmentation import augment_dataset


class ThaiTextPreprocessor:
    """Preprocessor for Thai text data with model-specific optimization"""
    
    def __init__(self):
        self.label_encoder = LabelEncoder()
        self.tfidf_vectorizer = None
        self.count_vectorizer = None
        self.feature_selectors = {}  # แยก selector ตาม model
        
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
    
    def fit_tfidf(self, texts: List[str], model_type: str = 'default'):
        """Fit TF-IDF vectorizer with model-specific settings"""
        # Model-specific ngram settings
        ngram_settings = {
            'svm': (1, 2),           # Bigrams work well for SVM
            'naive_bayes': (1, 1),   # Unigrams only for NB (ทำงานดีกว่า)
            'neural': (1, 2),        # Bigrams for neural nets
            'deep': (1, 2),          # Bigrams for deep learning
            'bayesian': (1, 2),      # Bigrams for Bayesian
            'maxent': (1, 2),        # Bigrams for MaxEnt
            'default': (1, 2)        # Default bigrams
        }
        
        ngram_range = ngram_settings.get(model_type, (1, 2))
        
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=config.MAX_FEATURES,
            ngram_range=ngram_range,
            min_df=2,
            max_df=0.95,
            sublinear_tf=True  # ใช้ log scaling (ดีกับ Thai)
        )
        self.tfidf_vectorizer.fit(texts)
    
    def fit_count(self, texts: List[str], model_type: str = 'default'):
        """Fit Count vectorizer (best for Naive Bayes)"""
        self.count_vectorizer = CountVectorizer(
            max_features=config.MAX_FEATURES,
            ngram_range=(1, 1),  # Unigrams only for count-based
            min_df=2,
            max_df=0.95
        )
        self.count_vectorizer.fit(texts)
    
    def fit_feature_selector(self, X: np.ndarray, y: np.ndarray, 
                            model_type: str, n_features: int = 3000):
        """Fit feature selector for specific model type"""
        if model_type not in self.feature_selectors:
            selector = SelectKBest(chi2, k=min(n_features, X.shape[1]))
            selector.fit(X, y)
            self.feature_selectors[model_type] = selector
            print(f"   ✓ Feature selector for {model_type}: {X.shape[1]} → {n_features}")
    
    def apply_feature_selection(self, X: np.ndarray, model_type: str) -> np.ndarray:
        """Apply feature selection if available"""
        if model_type in self.feature_selectors:
            return self.feature_selectors[model_type].transform(X)
        return X
    
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


def load_all_data(
    use_augmentation: bool = True,
    aug_per_sample: int = 1,
    balance_classes: bool = True
):
    """Load and preprocess all datasets."""
    preprocessor = ThaiTextPreprocessor()
    
    # Load data
    print("📂 Loading datasets...")
    train_texts, train_labels = preprocessor.load_data(config.TRAIN_FILE)
    test_texts, test_labels = preprocessor.load_data(config.TEST_FILE)
    valid_texts, valid_labels = preprocessor.load_data(config.VALID_FILE)
    
    print(f"   Train: {len(train_texts)} samples")
    print(f"   Valid: {len(valid_texts)} samples")
    print(f"   Test: {len(test_texts)} samples")

    original_train_size = len(train_texts)
    augmentation_config = {
        'enabled': use_augmentation,
        'aug_per_sample': aug_per_sample,
        'balance_classes': balance_classes,
        'methods': ['delete', 'swap', 'synonym', 'insert'],
    }

    if use_augmentation:
        print("   Applying ThaiTextAugmenter to training split...")
        train_texts, train_labels = augment_dataset(
            train_texts,
            train_labels,
            aug_per_sample=aug_per_sample,
            balance_classes=balance_classes
        )
    else:
        print("   Augmentation disabled. Using original training split.")

    augmented_train_size = len(train_texts)
    print(f"   Training samples after augmentation: {augmented_train_size}")
    
    # Preprocess texts
    print("🔤 Tokenizing Thai text...")
    train_texts_processed = preprocessor.preprocess_texts(train_texts)
    valid_texts_processed = preprocessor.preprocess_texts(valid_texts)
    test_texts_processed = preprocessor.preprocess_texts(test_texts)
    
    # Fit vectorizers on training data
    print("📊 Fitting vectorizers (model-specific)...")
    preprocessor.fit_tfidf(train_texts_processed, model_type='default')
    preprocessor.fit_count(train_texts_processed, model_type='naive_bayes')
    
    # Transform texts (default: TF-IDF)
    print("🔄 Transforming texts...")
    X_train = preprocessor.transform_tfidf(train_texts_processed)
    X_valid = preprocessor.transform_tfidf(valid_texts_processed)
    X_test = preprocessor.transform_tfidf(test_texts_processed)
    
    # For Naive Bayes: Count-based features
    print("🔄 Preparing Count features for Naive Bayes...")
    X_train_count = preprocessor.transform_count(train_texts_processed)
    X_valid_count = preprocessor.transform_count(valid_texts_processed)
    X_test_count = preprocessor.transform_count(test_texts_processed)
    
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
        'X_train_count': X_train_count,  # For Naive Bayes
        'X_valid_count': X_valid_count,
        'X_test_count': X_test_count,
        'y_train': y_train,
        'y_valid': y_valid,
        'y_test': y_test,
        'preprocessor': preprocessor,
        'train_texts': train_texts_processed,
        'valid_texts': valid_texts_processed,
        'test_texts': test_texts_processed,
        'original_train_size': original_train_size,
        'augmented_train_size': augmented_train_size,
        'augmentation_enabled': use_augmentation,
        'augmentation_config': augmentation_config,
    }


def to_gpu_tensor(data: np.ndarray, dtype=torch.float32) -> torch.Tensor:
    """Convert numpy array to GPU tensor"""
    return torch.tensor(data, dtype=dtype).to(config.DEVICE)


def prepare_data_for_model(X, y=None, model_type='neural', preprocessor=None):
    """
    Prepare data specifically for different model types
    v1.2.2: Enhanced with feature selection and model-specific optimization
    
    Args:
        X: Input data (numpy array or sparse matrix)
        y: Labels (for feature selection)
        model_type: Type of model ('neural', 'svm', 'naive_bayes', 'deep', 'bayesian', 'maxent')
        preprocessor: Preprocessor instance (for feature selection)
    
    Returns:
        Processed data suitable for the model
    """
    # Convert sparse to dense if needed
    if hasattr(X, 'toarray'):
        X = X.toarray()
    
    # Model-specific preprocessing
    if model_type == 'naive_bayes':
        # Naive Bayes requires non-negative features
        # Count features are already non-negative, but ensure
        X = np.abs(X) + 1e-10
    
    elif model_type == 'svm':
        # SVM: Feature selection + normalization
        if preprocessor:
            # If y is provided, fit the selector (training data)
            if y is not None and 'svm' not in preprocessor.feature_selectors:
                n_features = config.MODEL_PARAMS['svm'].get('n_features_select', 3000)
                preprocessor.fit_feature_selector(X, y, 'svm', n_features)
            
            # Apply selection (for both train and test)
            if 'svm' in preprocessor.feature_selectors:
                X = preprocessor.apply_feature_selection(X, 'svm')
        
        # L2 normalization
        from sklearn.preprocessing import normalize
        X = normalize(X, norm='l2')
    
    elif model_type in ['neural', 'deep', 'bayesian', 'maxent']:
        # Neural networks: standardization
        # TF-IDF already normalized, just ensure dtype
        X = X.astype(np.float32)
    
    return X
