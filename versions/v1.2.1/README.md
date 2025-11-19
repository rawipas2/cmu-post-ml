# Thai Depression Classification - v1.2.1

## Overview
Training run completed on: 2025-11-19 19:55:45

## Performance Summary

### Target
- **Target Accuracy**: 80.0%

### Results
- **Best Single Model**: Naive_Bayes (0.7567)
- **Average Accuracy**: 0.7444

## Individual Model Performance

| Model | Accuracy | Precision | Recall | F1 Score | AUC-ROC |
|-------|----------|-----------|--------|----------|---------|
| Naive_Bayes | 0.7567 | 0.7568 | 0.7567 | 0.7567 | 0.8395 |
| SVM | 0.7490 | 0.7505 | 0.7490 | 0.7482 | 0.8290 |
| Deep_Learning | 0.7454 | 0.7455 | 0.7454 | 0.7452 | 0.8144 |
| Bayesian_Network | 0.7406 | 0.7407 | 0.7406 | 0.7406 | 0.8277 |
| Neural_Network | 0.7302 | 0.7333 | 0.7302 | 0.7298 | 0.8125 |

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

- 5 model(s) below target accuracy
  - SVM: 0.7490
  - Neural_Network: 0.7302
  - Deep_Learning: 0.7454

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
- Feature dimension: 10000
- Models successfully trained: 5

## Hardware
- Device: cuda
- GPU: Available


## Configuration

```python
MAX_FEATURES: 10000
BATCH_SIZE: 16
EPOCHS: 150
LEARNING_RATE: 0.0003
DEVICE: cuda
```

---
Generated automatically by Thai Depression Classification System
