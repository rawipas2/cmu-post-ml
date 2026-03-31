# Thai Depression Classification - factorial_shared-focal

## Overview
Training run completed on: 2026-04-01 00:12:26

## Performance Summary
- **Best Single Model**: Naive_Bayes (0.7535)
- **Average Accuracy**: 0.7053
- **Ensemble Accuracy**: 0.7386

## Individual Model Performance

| Model | Accuracy | Precision | Recall | F1 Score | AUC-ROC |
|---|---:|---:|---:|---:|---:|
| Naive_Bayes | 0.7535 | 0.7535 | 0.7535 | 0.7535 | 0.8356 |
| Neural_Network | 0.7414 | 0.7418 | 0.7414 | 0.7410 | 0.8200 |
| Deep_Learning | 0.7414 | 0.7468 | 0.7414 | 0.7392 | 0.8238 |
| Bayesian_Network | 0.7406 | 0.7492 | 0.7406 | 0.7374 | 0.8277 |
| Ensemble_Stacking | 0.7386 | 0.7391 | 0.7386 | 0.7381 | 0.8145 |
| Maximum_Entropy | 0.7099 | 0.7767 | 0.7099 | 0.6886 | 0.8463 |
| SVM | 0.5119 | 0.2620 | 0.5119 | 0.3466 | 0.7899 |

## Experiment Configuration
- **Experiment Name**: factorial_shared-focal
- **Version Name**: factorial_shared-focal
- **Preprocessing Mode**: shared
- **Loss Mode**: focal
- **SVM Policy**: freeze_v1_2

## Highlights
- Top-performing models in this run:
  - Naive_Bayes: 0.7535
  - Neural_Network: 0.7414
  - Deep_Learning: 0.7414

## Cautions
- Lowest-performing models in this run:
  - SVM: 0.5119
  - Maximum_Entropy: 0.7099
  - Ensemble_Stacking: 0.7386

## Notes
This run used preprocessing_mode=shared, loss_mode=focal, svm_policy=freeze_v1_2.

## Training Details
- Total samples trained: 25077
- Validation samples: 3344
- Test samples: 5015
- Shared feature dimension: 5000
- Models successfully trained: 6

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
