# Thai Depression Classification - v2.0-baseline

## Overview
Training run completed on: 2026-04-01 12:16:25

## Performance Summary
- **Best Single Model**: Naive_Bayes (0.7617)
- **Average Accuracy**: 0.7077
- **Ensemble Accuracy**: 0.7424

## Individual Model Performance

| Model | Accuracy | Precision | Recall | F1 Score | AUC-ROC |
|---|---:|---:|---:|---:|---:|
| Naive_Bayes | 0.7617 | 0.7617 | 0.7617 | 0.7616 | 0.8459 |
| Bayesian_Network | 0.7466 | 0.7531 | 0.7466 | 0.7441 | 0.8304 |
| Deep_Learning | 0.7450 | 0.7481 | 0.7450 | 0.7436 | 0.8212 |
| Ensemble_Stacking | 0.7424 | 0.7425 | 0.7424 | 0.7422 | 0.8226 |
| Neural_Network | 0.7372 | 0.7419 | 0.7372 | 0.7351 | 0.8196 |
| Maximum_Entropy | 0.7093 | 0.7752 | 0.7093 | 0.6881 | 0.8465 |
| SVM | 0.5119 | 0.2620 | 0.5119 | 0.3466 | 0.7899 |

## Experiment Configuration
- **Experiment Name**: v2.0-baseline
- **Version Name**: v2.0-baseline
- **Preprocessing Mode**: model_specific
- **Loss Mode**: focal
- **SVM Policy**: freeze_v1_2
- **Augmentation Enabled**: False
- **Augment Per Sample**: 1
- **Balance Classes**: True
- **Train Samples (Original)**: 25077
- **Train Samples (Final)**: 25077

## Highlights
- Top-performing models in this run:
  - Naive_Bayes: 0.7617
  - Bayesian_Network: 0.7466
  - Deep_Learning: 0.7450

## Cautions
- Lowest-performing models in this run:
  - SVM: 0.5119
  - Maximum_Entropy: 0.7093
  - Neural_Network: 0.7372

## Notes
This run used preprocessing_mode=model_specific, loss_mode=focal, svm_policy=freeze_v1_2.

v2.0 default recipe keeps ThaiTextAugmenter disabled for the training split.

## Training Details
- Total samples trained: 25077
- Original train samples: 25077
- Final train samples after augmentation: 25077
- Validation samples: 3344
- Test samples: 5015
- Shared feature dimension: 5000
- Models successfully trained: 6

## Augmentation
- Enabled: False
- Augment per sample: 1
- Balance classes: True

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
