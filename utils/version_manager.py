"""
Utilities for managing model versions and documentation
"""
import os
import json
from datetime import datetime
import config


def create_version_directory(version_name=None):
    """Create a new version directory"""
    if version_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        version_name = f"v_{timestamp}"
    
    version_dir = os.path.join(config.VERSIONS_DIR, version_name)
    os.makedirs(version_dir, exist_ok=True)
    
    # Create subdirectories
    os.makedirs(os.path.join(version_dir, 'models'), exist_ok=True)
    os.makedirs(os.path.join(version_dir, 'plots'), exist_ok=True)
    os.makedirs(os.path.join(version_dir, 'metrics'), exist_ok=True)
    
    return version_dir


def generate_readme(version_dir, metrics_list, notes=""):
    """Generate README.md for version"""
    
    readme_path = os.path.join(version_dir, 'README.md')
    
    # Calculate summary statistics
    best_model = max(metrics_list, key=lambda x: x['accuracy'])
    avg_accuracy = sum(m['accuracy'] for m in metrics_list) / len(metrics_list)
    
    # Get ensemble metrics if exists
    ensemble_metrics = next((m for m in metrics_list if 'ensemble' in m['model_name'].lower()), None)
    
    content = f"""# Thai Depression Classification - {os.path.basename(version_dir)}

## Overview
Training run completed on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Performance Summary

### Target
- **Target Accuracy**: {config.TARGET_ACCURACY * 100}%

### Results
- **Best Single Model**: {best_model['model_name']} ({best_model['accuracy']:.4f})
- **Average Accuracy**: {avg_accuracy:.4f}
"""
    
    if ensemble_metrics:
        content += f"- **Ensemble Accuracy**: {ensemble_metrics['accuracy']:.4f} {'✅ TARGET MET!' if ensemble_metrics['accuracy'] >= config.TARGET_ACCURACY else '❌ Below target'}\n"
    
    content += f"""
## Individual Model Performance

| Model | Accuracy | Precision | Recall | F1 Score | AUC-ROC |
|-------|----------|-----------|--------|----------|---------|
"""
    
    for metrics in sorted(metrics_list, key=lambda x: x['accuracy'], reverse=True):
        auc = metrics.get('auc_roc', 'N/A')
        if isinstance(auc, float):
            auc = f"{auc:.4f}"
        content += f"| {metrics['model_name']} | {metrics['accuracy']:.4f} | {metrics['precision']:.4f} | {metrics['recall']:.4f} | {metrics['f1_score']:.4f} | {auc} |\n"
    
    content += f"""
## Models Used

1. **Support Vector Machine (SVM)** - GPU-accelerated using cuML
2. **Neural Network** - PyTorch MLP with GPU support
3. **Deep Learning** - Deep neural network with dropout
4. **Naive Bayes** - Multinomial Naive Bayes with GPU acceleration
5. **Bayesian Network** - Neural network with Bayesian principles
6. **Maximum Entropy** - Logistic Regression (MaxEnt) on GPU

## Ensemble Method

**Stacking Ensemble**: Meta-learner trained on predictions from all 6 base models

## Strengths ✅

"""
    
    # Analyze strengths
    high_performers = [m for m in metrics_list if m['accuracy'] >= config.TARGET_ACCURACY]
    if high_performers:
        content += f"- {len(high_performers)} model(s) achieved target accuracy\n"
    
    if ensemble_metrics and ensemble_metrics['accuracy'] >= config.TARGET_ACCURACY:
        content += "- Ensemble model successfully met target accuracy\n"
    
    content += f"- All models trained on GPU for optimal performance\n"
    content += f"- Comprehensive evaluation metrics and visualizations\n"
    
    content += f"""
## Weaknesses ❌

"""
    
    # Analyze weaknesses
    low_performers = [m for m in metrics_list if m['accuracy'] < config.TARGET_ACCURACY]
    if low_performers:
        content += f"- {len(low_performers)} model(s) below target accuracy\n"
        for m in low_performers[:3]:  # Show top 3 weakest
            content += f"  - {m['model_name']}: {m['accuracy']:.4f}\n"
    
    if ensemble_metrics and ensemble_metrics['accuracy'] < config.TARGET_ACCURACY:
        content += "- Ensemble model did not achieve target accuracy\n"
    
    content += f"""
## Improvements for Next Version

- Fine-tune hyperparameters for underperforming models
- Experiment with different ensemble techniques (voting, boosting)
- Try advanced Thai text preprocessing (subword tokenization)
- Increase model complexity or add more features
- Use pre-trained Thai language models (WangchanBERTa)
- Augment training data

## Files

### Models
- All trained models saved in `models/` directory
- Models can be loaded for inference or further training

### Visualizations
- Confusion matrices for each model
- ROC curves showing model discrimination
- Model comparison charts

### Metrics
- Detailed JSON files with all evaluation metrics
- Classification reports for each model

## Notes

{notes if notes else "No additional notes for this version."}

## Configuration

```python
MAX_FEATURES: {config.MAX_FEATURES}
BATCH_SIZE: {config.BATCH_SIZE}
EPOCHS: {config.EPOCHS}
LEARNING_RATE: {config.LEARNING_RATE}
DEVICE: {config.DEVICE}
```

---
Generated automatically by Thai Depression Classification System
"""
    
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"📝 Generated README: {readme_path}")
    return readme_path
