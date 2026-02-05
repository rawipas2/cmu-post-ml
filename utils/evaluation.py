"""
Evaluation metrics and visualization utilities
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc,
    roc_auc_score, precision_recall_curve, average_precision_score
)
import json
import os
from datetime import datetime
import config


def _ensure_binary_labels(y_true):
    y_true = np.asarray(y_true)
    unique = np.unique(y_true)
    if unique.size != 2:
        raise ValueError(
            f"Expected binary labels, got {unique.size} unique values: {unique.tolist()}"
        )
    if set(unique.tolist()) == {0, 1}:
        return y_true.astype(int)
    negative, positive = unique.min(), unique.max()
    return (y_true == positive).astype(int)


def positive_class_score(y_pred_proba):
    """
    Normalize probability outputs to a 1D array of positive-class scores.
    Supports shape (n_samples,) or (n_samples, 2).
    """
    scores = np.asarray(y_pred_proba)
    if scores.ndim == 1:
        return scores
    if scores.ndim == 2 and scores.shape[1] == 2:
        return scores[:, 1]
    raise ValueError(f"Unsupported probability shape: {scores.shape}")


def evaluate_model(y_true, y_pred, y_pred_proba=None, model_name='Model'):
    """Evaluate model performance"""
    metrics = {
        'model_name': model_name,
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted'),
        'recall': recall_score(y_true, y_pred, average='weighted'),
        'f1_score': f1_score(y_true, y_pred, average='weighted')
    }
    
    if y_pred_proba is not None:
        metrics['auc_roc'] = roc_auc_score(_ensure_binary_labels(y_true), positive_class_score(y_pred_proba))
    
    return metrics


def print_metrics(metrics):
    """Print evaluation metrics"""
    print(f"\n{'='*60}")
    print(f"📊 {metrics['model_name']} - Evaluation Results")
    print(f"{'='*60}")
    print(f"Accuracy:  {metrics['accuracy']:.4f} {'✅' if metrics['accuracy'] >= config.TARGET_ACCURACY else '❌'}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1 Score:  {metrics['f1_score']:.4f}")
    if 'auc_roc' in metrics:
        print(f"AUC-ROC:   {metrics['auc_roc']:.4f}")
    print(f"{'='*60}\n")


def plot_confusion_matrix(y_true, y_pred, model_name, save_path):
    """Plot and save confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   Saved confusion matrix: {save_path}")


def plot_roc_curve(y_true, y_pred_proba, model_name, save_path):
    """Plot and save ROC curve"""
    y_true = _ensure_binary_labels(y_true)
    y_score = positive_class_score(y_pred_proba)
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name}')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   Saved ROC curve: {save_path}")


def plot_pr_curve(y_true, y_pred_proba, model_name, save_path):
    """Plot and save Precision-Recall curve"""
    y_true = _ensure_binary_labels(y_true)
    y_score = positive_class_score(y_pred_proba)

    precision, recall, _ = precision_recall_curve(y_true, y_score)
    ap = average_precision_score(y_true, y_score)
    prevalence = float(np.mean(y_true))

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='purple', lw=2, label=f'PR curve (AP = {ap:.4f})')
    plt.hlines(prevalence, 0.0, 1.0, colors='gray', linestyles='--', lw=1.5, label='Baseline')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve - {model_name}')
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   Saved PR curve: {save_path}")


def tune_threshold_by_f1(y_true, y_score):
    """Tune a decision threshold to maximize F1 on a validation set."""
    y_true = _ensure_binary_labels(y_true)
    y_score = positive_class_score(y_score)

    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    if thresholds.size == 0:
        return {
            "threshold": 0.5,
            "f1": float("nan"),
            "precision": float("nan"),
            "recall": float("nan"),
        }

    precision_t = precision[:-1]
    recall_t = recall[:-1]
    f1 = (2 * precision_t * recall_t) / (precision_t + recall_t + 1e-12)
    best_idx = int(np.nanargmax(f1))
    return {
        "threshold": float(thresholds[best_idx]),
        "f1": float(f1[best_idx]),
        "precision": float(precision_t[best_idx]),
        "recall": float(recall_t[best_idx]),
    }


def plot_confusion_matrix_at_threshold(y_true, y_score, threshold, model_name, save_path):
    """Plot confusion matrix generated from probability scores at a given threshold."""
    y_true = _ensure_binary_labels(y_true)
    y_score = positive_class_score(y_score)
    y_pred = (y_score >= threshold).astype(int)
    title = f"{model_name} (τ* = {threshold:.3f})"
    plot_confusion_matrix(y_true, y_pred, title, save_path)


def plot_roc_pr_curves(curves, save_path, title_suffix=""):
    """
    Plot ROC and PR curves side-by-side.
    curves: list of dicts with keys: label, y_true, y_score
    """
    if not curves:
        raise ValueError("No curves provided")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))

    # ROC
    ax = axes[0]
    ax.plot([0, 1], [0, 1], color='navy', lw=1.5, linestyle='--', alpha=0.8)
    for item in curves:
        y_true = _ensure_binary_labels(item["y_true"])
        y_score = positive_class_score(item["y_score"])
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, lw=2, label=f"{item['label']} (AUC={roc_auc:.3f})")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'ROC curves{title_suffix}')
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower right", fontsize=8)

    # PR
    ax = axes[1]
    all_y = _ensure_binary_labels(curves[0]["y_true"])
    prevalence = float(np.mean(all_y))
    ax.hlines(prevalence, 0.0, 1.0, colors='gray', linestyles='--', lw=1.5, label='Baseline')
    for item in curves:
        y_true = _ensure_binary_labels(item["y_true"])
        y_score = positive_class_score(item["y_score"])
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        ap = average_precision_score(y_true, y_score)
        ax.plot(recall, precision, lw=2, label=f"{item['label']} (AP={ap:.3f})")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title(f'PR curves{title_suffix}')
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower left", fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"   Saved ROC+PR curves: {save_path}")


def plot_model_comparison(all_metrics, save_path):
    """Plot comparison of all models"""
    model_names = [m['model_name'] for m in all_metrics]
    accuracies = [m['accuracy'] for m in all_metrics]
    f1_scores = [m['f1_score'] for m in all_metrics]
    
    x = np.arange(len(model_names))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width/2, accuracies, width, label='Accuracy', alpha=0.8)
    bars2 = ax.bar(x + width/2, f1_scores, width, label='F1 Score', alpha=0.8)
    
    ax.set_xlabel('Models')
    ax.set_ylabel('Scores')
    ax.set_title('Model Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=45, ha='right')
    ax.legend()
    ax.axhline(y=config.TARGET_ACCURACY, color='r', linestyle='--', 
               label=f'Target Accuracy ({config.TARGET_ACCURACY})')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   Saved comparison plot: {save_path}")


def save_results(metrics, save_dir, model_name):
    """Save evaluation results to JSON"""
    os.makedirs(save_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(save_dir, f'{model_name}_{timestamp}.json')
    
    # Add timestamp to metrics
    metrics['timestamp'] = timestamp
    metrics['version'] = config.CURRENT_VERSION
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=4, ensure_ascii=False)
    
    print(f"   Saved results: {filepath}")
    return filepath


def generate_classification_report(y_true, y_pred, preprocessor, save_path):
    """Generate and save detailed classification report"""
    report = classification_report(
        y_true, y_pred, 
        target_names=preprocessor.label_encoder.classes_,
        digits=4
    )
    
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"   Saved classification report: {save_path}")
    return report
