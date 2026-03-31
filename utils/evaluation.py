"""
Evaluation metrics and visualization utilities
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc,
    roc_auc_score
)
import json
import os
from datetime import datetime
import config


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
        metrics['auc_roc'] = roc_auc_score(y_true, y_pred_proba)
    
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
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
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


def save_results(metrics, save_dir, model_name, version_name=None, metadata=None):
    """Save evaluation results to JSON"""
    os.makedirs(save_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(save_dir, f'{model_name}_{timestamp}.json')
    
    # Add timestamp to metrics
    metrics['timestamp'] = timestamp
    metrics['version'] = version_name or config.CURRENT_VERSION
    if metadata:
        metrics.update(metadata)
    
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
