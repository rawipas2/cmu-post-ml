"""
Fallback implementations for models when cuML/cuPy is not available
Falls back to CPU-based alternatives from scikit-learn
"""
import warnings

# ตรวจสอบว่ามี cuML และ cuPy หรือไม่
try:
    import cuml
    import cupy
    HAS_RAPIDS = True
except ImportError:
    HAS_RAPIDS = False
    warnings.warn(
        "\n⚠️  RAPIDS AI (cuML, cuPy) not found! "
        "\nFalling back to CPU-based alternatives. "
        "\nFor best GPU performance, install RAPIDS: "
        "\n  conda install -c rapidsai -c conda-forge -c nvidia cuml cupy"
    )

# Import ที่เหมาะสมตาม availability
if HAS_RAPIDS:
    from cuml.svm import SVC as GPU_SVC
    from cuml.naive_bayes import MultinomialNB as GPU_MultinomialNB
    import cupy as cp
else:
    from sklearn.svm import SVC as GPU_SVC
    from sklearn.naive_bayes import MultinomialNB as GPU_MultinomialNB
    import numpy as cp  # ใช้ numpy แทน cupy


def to_gpu_array(data):
    """Convert data to GPU array if available, otherwise return numpy array"""
    if HAS_RAPIDS:
        return cp.asarray(data)
    return data


def to_cpu_array(data):
    """Convert GPU array to CPU if needed"""
    if HAS_RAPIDS and isinstance(data, cp.ndarray):
        return cp.asnumpy(data)
    return data


def get_svc_model(**kwargs):
    """Get SVM model (GPU if available, CPU fallback)"""
    if not HAS_RAPIDS:
        # Ensure probability is enabled for CPU version
        kwargs['probability'] = True
    return GPU_SVC(**kwargs)


def get_naive_bayes_model(**kwargs):
    """Get Naive Bayes model (GPU if available, CPU fallback)"""
    return GPU_MultinomialNB(**kwargs)


__all__ = [
    'HAS_RAPIDS',
    'GPU_SVC',
    'GPU_MultinomialNB',
    'to_gpu_array',
    'to_cpu_array',
    'get_svc_model',
    'get_naive_bayes_model',
    'cp'
]
