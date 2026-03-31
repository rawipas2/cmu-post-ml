# Thai Depression Classification - factorial_shared-bce_20260331_234626

## Overview
Training run completed on: 2026-04-01 00:00:45

## Performance Summary
- **Best Single Model**: Maximum_Entropy (0.7631)
- **Average Accuracy**: 0.7139
- **Ensemble Accuracy**: 0.7394

## Individual Model Performance

| Model | Accuracy | Precision | Recall | F1 Score | AUC-ROC |
|---|---:|---:|---:|---:|---:|
| Maximum_Entropy | 0.7631 | 0.7633 | 0.7631 | 0.7629 | 0.8462 |
| Naive_Bayes | 0.7535 | 0.7535 | 0.7535 | 0.7535 | 0.8356 |
| Bayesian_Network | 0.7466 | 0.7466 | 0.7466 | 0.7464 | 0.8302 |
| Neural_Network | 0.7424 | 0.7426 | 0.7424 | 0.7421 | 0.8228 |
| Deep_Learning | 0.7402 | 0.7426 | 0.7402 | 0.7390 | 0.8196 |
| Ensemble_Stacking | 0.7394 | 0.7415 | 0.7394 | 0.7383 | 0.8178 |
| SVM | 0.5119 | 0.2620 | 0.5119 | 0.3466 | 0.7899 |

## Experiment Configuration
- **Experiment Name**: factorial_shared-bce
- **Version Name**: factorial_shared-bce
- **Preprocessing Mode**: shared
- **Loss Mode**: bce
- **SVM Policy**: freeze_v1_2

## Highlights
- Top-performing models in this run:
  - Maximum_Entropy: 0.7631
  - Naive_Bayes: 0.7535
  - Bayesian_Network: 0.7466

## Cautions
- Lowest-performing models in this run:
  - SVM: 0.5119
  - Ensemble_Stacking: 0.7394
  - Deep_Learning: 0.7402

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
