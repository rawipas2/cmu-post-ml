# Thai Depression Classification - factorial_shared-bce

## Overview
Training run completed on: 2026-03-31 23:59:28

## Performance Summary
- **Best Single Model**: Maximum_Entropy (0.7629)
- **Average Accuracy**: 0.7125
- **Ensemble Accuracy**: 0.7382

## Individual Model Performance

| Model | Accuracy | Precision | Recall | F1 Score | AUC-ROC |
|---|---:|---:|---:|---:|---:|
| Maximum_Entropy | 0.7629 | 0.7630 | 0.7629 | 0.7628 | 0.8463 |
| Naive_Bayes | 0.7535 | 0.7535 | 0.7535 | 0.7535 | 0.8356 |
| Bayesian_Network | 0.7416 | 0.7418 | 0.7416 | 0.7413 | 0.8278 |
| Neural_Network | 0.7402 | 0.7402 | 0.7402 | 0.7400 | 0.8157 |
| Deep_Learning | 0.7394 | 0.7430 | 0.7394 | 0.7378 | 0.8182 |
| Ensemble_Stacking | 0.7382 | 0.7417 | 0.7382 | 0.7366 | 0.8026 |
| SVM | 0.5119 | 0.2620 | 0.5119 | 0.3466 | 0.7899 |

## Experiment Configuration
- **Experiment Name**: factorial_shared-bce
- **Version Name**: factorial_shared-bce
- **Preprocessing Mode**: shared
- **Loss Mode**: bce
- **SVM Policy**: freeze_v1_2

## Highlights
- Top-performing models in this run:
  - Maximum_Entropy: 0.7629
  - Naive_Bayes: 0.7535
  - Bayesian_Network: 0.7416

## Cautions
- Lowest-performing models in this run:
  - SVM: 0.5119
  - Ensemble_Stacking: 0.7382
  - Deep_Learning: 0.7394

## Notes
This run used preprocessing_mode=shared, loss_mode=bce, svm_policy=freeze_v1_2.

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
