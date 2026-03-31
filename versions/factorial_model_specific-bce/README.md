# Thai Depression Classification - factorial_model_specific-bce

## Overview
Training run completed on: 2026-04-01 00:15:25

## Performance Summary
- **Best Single Model**: Maximum_Entropy (0.7629)
- **Average Accuracy**: 0.7162
- **Ensemble Accuracy**: 0.7480

## Individual Model Performance

| Model | Accuracy | Precision | Recall | F1 Score | AUC-ROC |
|---|---:|---:|---:|---:|---:|
| Maximum_Entropy | 0.7629 | 0.7630 | 0.7629 | 0.7627 | 0.8462 |
| Naive_Bayes | 0.7617 | 0.7617 | 0.7617 | 0.7616 | 0.8459 |
| Ensemble_Stacking | 0.7480 | 0.7487 | 0.7480 | 0.7480 | 0.7944 |
| Deep_Learning | 0.7466 | 0.7470 | 0.7466 | 0.7466 | 0.8252 |
| Bayesian_Network | 0.7422 | 0.7422 | 0.7422 | 0.7421 | 0.8290 |
| Neural_Network | 0.7404 | 0.7404 | 0.7404 | 0.7403 | 0.8190 |
| SVM | 0.5119 | 0.2620 | 0.5119 | 0.3466 | 0.7899 |

## Experiment Configuration
- **Experiment Name**: factorial_model_specific-bce
- **Version Name**: factorial_model_specific-bce
- **Preprocessing Mode**: model_specific
- **Loss Mode**: bce
- **SVM Policy**: freeze_v1_2

## Highlights
- Top-performing models in this run:
  - Maximum_Entropy: 0.7629
  - Naive_Bayes: 0.7617
  - Ensemble_Stacking: 0.7480

## Cautions
- Lowest-performing models in this run:
  - SVM: 0.5119
  - Neural_Network: 0.7404
  - Bayesian_Network: 0.7422

## Notes
This run used preprocessing_mode=model_specific, loss_mode=bce, svm_policy=freeze_v1_2.

## Training Details
- Total samples trained: 25077
- Validation samples: 3344
- Test samples: 5015
- Shared feature dimension: 5000
- Models successfully trained: 6

## Reporting Note
- SVM uses the shared TF-IDF path as the stable v1.2-style baseline.
- This avoids the known calibration/threshold collapse from the current model-specific SVM path.

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
