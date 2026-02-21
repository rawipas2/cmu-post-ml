# Thai Depression Classification - v1.2.1

## Overview
Training run completed on: 2026-02-21 08:21:00

## Performance Summary

### Target
- **Target Accuracy**: 80.0%

### Results
- **Best Single Model**: Maximum_Entropy (0.7631)
- **Average Accuracy**: 0.6790

## Individual Model Performance

| Model | Accuracy | Precision | Recall | F1 Score | AUC-ROC |
|-------|----------|-----------|--------|----------|---------|
| Maximum_Entropy | 0.7631 | 0.7633 | 0.7631 | 0.7629 | 0.8461 |
| Naive_Bayes | 0.7619 | 0.7619 | 0.7619 | 0.7618 | 0.8451 |
| SVM | 0.5119 | 0.2620 | 0.5119 | 0.3466 | 0.7837 |

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

- All models trained on GPU for optimal performance
- Comprehensive evaluation metrics and visualizations

## Weaknesses ❌

- 3 model(s) below target accuracy
  - SVM: 0.5119
  - Naive_Bayes: 0.7619
  - Maximum_Entropy: 0.7631

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


This is version v1.2.1 of the Thai Depression Classification system.

## Training Details
- Total samples trained: 25077
- Validation samples: 3344
- Test samples: 5015
- Feature dimension: 5000
- Models successfully trained: 3

## Hardware
- Device: cpu
- GPU: Not Available


## Configuration

```python
MAX_FEATURES: 5000
BATCH_SIZE: 32
EPOCHS: 100
LEARNING_RATE: 0.0005
DEVICE: cpu
```

---
Generated automatically by Thai Depression Classification System
