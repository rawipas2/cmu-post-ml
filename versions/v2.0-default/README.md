# Thai Depression Classification - v2.0-default

## Overview
Training run completed on: 2026-04-01 12:17:16

## Performance Summary
- **Best Single Model**: Naive_Bayes (0.7577)
- **Average Accuracy**: 0.7059
- **Ensemble Accuracy**: 0.7404

## Individual Model Performance

| Model | Accuracy | Precision | Recall | F1 Score | AUC-ROC |
|---|---:|---:|---:|---:|---:|
| Naive_Bayes | 0.7577 | 0.7578 | 0.7577 | 0.7577 | 0.8449 |
| Bayesian_Network | 0.7470 | 0.7538 | 0.7470 | 0.7444 | 0.8308 |
| Ensemble_Stacking | 0.7404 | 0.7425 | 0.7404 | 0.7393 | 0.8079 |
| Deep_Learning | 0.7396 | 0.7471 | 0.7396 | 0.7367 | 0.8205 |
| Neural_Network | 0.7326 | 0.7405 | 0.7326 | 0.7294 | 0.8148 |
| Maximum_Entropy | 0.7123 | 0.7753 | 0.7123 | 0.6922 | 0.8465 |
| SVM | 0.5119 | 0.2620 | 0.5119 | 0.3466 | 0.7908 |

## Experiment Configuration
- **Experiment Name**: v2.0-default
- **Version Name**: v2.0-default
- **Preprocessing Mode**: model_specific
- **Loss Mode**: focal
- **SVM Policy**: freeze_v1_2
- **Augmentation Enabled**: True
- **Augment Per Sample**: 1
- **Balance Classes**: True
- **Train Samples (Original)**: 25077
- **Train Samples (Final)**: 25674

## Highlights
- Top-performing models in this run:
  - Naive_Bayes: 0.7577
  - Bayesian_Network: 0.7470
  - Ensemble_Stacking: 0.7404

## Cautions
- Lowest-performing models in this run:
  - SVM: 0.5119
  - Maximum_Entropy: 0.7123
  - Neural_Network: 0.7326

## Notes
This run used preprocessing_mode=model_specific, loss_mode=focal, svm_policy=freeze_v1_2.

v2.0 default recipe keeps ThaiTextAugmenter enabled for the training split.

## Training Details
- Total samples trained: 25674
- Original train samples: 25077
- Final train samples after augmentation: 25674
- Validation samples: 3344
- Test samples: 5015
- Shared feature dimension: 5000
- Models successfully trained: 6

## Augmentation
- Enabled: True
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
