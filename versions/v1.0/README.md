# Thai Depression Classification - v1.0

## Overview
Training run completed on: 2025-11-19 00:16:27

## Performance Summary

### Target
- **Target Accuracy**: 80.0%

### Results
- **Best Single Model**: Neural_Network (0.7362)
- **Average Accuracy**: 0.7195
- **Ensemble Accuracy**: 0.7328 ❌ Below target

## Individual Model Performance

| Model | Accuracy | Precision | Recall | F1 Score | AUC-ROC |
|-------|----------|-----------|--------|----------|---------|
| Neural_Network | 0.7362 | 0.7364 | 0.7362 | 0.7362 | 0.8199 |
| Ensemble_Stacking | 0.7328 | 0.7345 | 0.7328 | 0.7327 | 0.8223 |
| Maximum_Entropy | 0.6895 | 0.7217 | 0.6895 | 0.6753 | 0.7905 |

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
  - Neural_Network: 0.7362
  - Maximum_Entropy: 0.6895
  - Ensemble_Stacking: 0.7328
- Ensemble model did not achieve target accuracy

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


This is version v1.0 of the Thai Depression Classification system.

## Training Details
- Total samples trained: 25077
- Validation samples: 3344
- Test samples: 5015
- Feature dimension: 5000
- Models successfully trained: 2

## Hardware
- Device: cuda
- GPU: Available


## Configuration

```python
MAX_FEATURES: 5000
BATCH_SIZE: 64
EPOCHS: 50
LEARNING_RATE: 0.001
DEVICE: cuda
```

---
Generated automatically by Thai Depression Classification System
