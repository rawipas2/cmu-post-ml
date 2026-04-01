# Thai Depression Classification - v2.0-aug3

## Overview
Training run completed on: 2026-04-01 12:18:41

## Performance Summary
- **Best Single Model**: Naive_Bayes (0.7543)
- **Average Accuracy**: 0.7031
- **Ensemble Accuracy**: 0.7378

## Individual Model Performance

| Model | Accuracy | Precision | Recall | F1 Score | AUC-ROC |
|---|---:|---:|---:|---:|---:|
| Naive_Bayes | 0.7543 | 0.7549 | 0.7543 | 0.7544 | 0.8441 |
| Bayesian_Network | 0.7458 | 0.7507 | 0.7458 | 0.7438 | 0.8311 |
| Deep_Learning | 0.7396 | 0.7440 | 0.7396 | 0.7377 | 0.8214 |
| Ensemble_Stacking | 0.7378 | 0.7391 | 0.7378 | 0.7370 | 0.8134 |
| Neural_Network | 0.7354 | 0.7436 | 0.7354 | 0.7322 | 0.8174 |
| Maximum_Entropy | 0.7206 | 0.7760 | 0.7206 | 0.7036 | 0.8465 |
| SVM | 0.4881 | 0.2383 | 0.4881 | 0.3202 | 0.7971 |

## Experiment Configuration
- **Experiment Name**: v2.0-aug3
- **Version Name**: v2.0-aug3
- **Preprocessing Mode**: model_specific
- **Loss Mode**: focal
- **SVM Policy**: freeze_v1_2
- **Augmentation Enabled**: True
- **Augment Per Sample**: 3
- **Balance Classes**: True
- **Train Samples (Original)**: 25077
- **Train Samples (Final)**: 26868

## Highlights
- Top-performing models in this run:
  - Naive_Bayes: 0.7543
  - Bayesian_Network: 0.7458
  - Deep_Learning: 0.7396

## Cautions
- Lowest-performing models in this run:
  - SVM: 0.4881
  - Maximum_Entropy: 0.7206
  - Neural_Network: 0.7354

## Notes
This run used preprocessing_mode=model_specific, loss_mode=focal, svm_policy=freeze_v1_2.

v2.0 default recipe keeps ThaiTextAugmenter enabled for the training split.

## Training Details
- Total samples trained: 26868
- Original train samples: 25077
- Final train samples after augmentation: 26868
- Validation samples: 3344
- Test samples: 5015
- Shared feature dimension: 5000
- Models successfully trained: 6

## Augmentation
- Enabled: True
- Augment per sample: 3
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
