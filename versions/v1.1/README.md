# Thai Depression Classification - v1.1

## Overview
Training run completed on: 2025-11-19 10:00:13

## Performance Summary

### Target
- **Target Accuracy**: 80.0%

### Results
- **Best Single Model**: SVM (0.7519)
- **Average Accuracy**: 0.7357

## Individual Model Performance

| Model | Accuracy | Precision | Recall | F1 Score | AUC-ROC |
|-------|----------|-----------|--------|----------|---------|
| SVM | 0.7519 | 0.7552 | 0.7519 | 0.7506 | 0.8307 |
| Naive_Bayes | 0.7517 | 0.7517 | 0.7517 | 0.7517 | 0.8353 |
| Deep_Learning | 0.7452 | 0.7452 | 0.7452 | 0.7452 | 0.8067 |
| Bayesian_Network | 0.7392 | 0.7393 | 0.7392 | 0.7390 | 0.8172 |
| Neural_Network | 0.7330 | 0.7347 | 0.7330 | 0.7329 | 0.8083 |
| Maximum_Entropy | 0.6931 | 0.7210 | 0.6931 | 0.6808 | 0.7890 |

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

- 6 model(s) below target accuracy
  - SVM: 0.7519
  - Neural_Network: 0.7330
  - Deep_Learning: 0.7452

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


This is version v1.1 of the Thai Depression Classification system.

## Training Details
- Total samples trained: 25077
- Validation samples: 3344
- Test samples: 5015
- Feature dimension: 5000
- Models successfully trained: 6

## Hardware
- Device: cuda
- GPU: Available


## Configuration

```python
MAX_FEATURES: 5000
BATCH_SIZE: 32
EPOCHS: 100
LEARNING_RATE: 0.0005
DEVICE: cuda
```

---
Generated automatically by Thai Depression Classification System
