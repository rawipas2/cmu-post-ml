# Thai Depression Classification - v2.0-aug2

## Overview
Training run completed on: 2026-04-01 12:18:44

## Performance Summary
- **Best Single Model**: Naive_Bayes (0.7555)
- **Average Accuracy**: 0.7077
- **Ensemble Accuracy**: 0.7402

## Individual Model Performance

| Model | Accuracy | Precision | Recall | F1 Score | AUC-ROC |
|---|---:|---:|---:|---:|---:|
| Naive_Bayes | 0.7555 | 0.7557 | 0.7555 | 0.7556 | 0.8446 |
| Bayesian_Network | 0.7442 | 0.7508 | 0.7442 | 0.7417 | 0.8294 |
| Neural_Network | 0.7438 | 0.7501 | 0.7438 | 0.7413 | 0.8251 |
| Deep_Learning | 0.7412 | 0.7427 | 0.7412 | 0.7403 | 0.8225 |
| Ensemble_Stacking | 0.7402 | 0.7401 | 0.7402 | 0.7401 | 0.8159 |
| Maximum_Entropy | 0.7170 | 0.7763 | 0.7170 | 0.6986 | 0.8466 |
| SVM | 0.5119 | 0.2620 | 0.5119 | 0.3466 | 0.7882 |

## Experiment Configuration
- **Experiment Name**: v2.0-aug2
- **Version Name**: v2.0-aug2
- **Preprocessing Mode**: model_specific
- **Loss Mode**: focal
- **SVM Policy**: freeze_v1_2
- **Augmentation Enabled**: True
- **Augment Per Sample**: 2
- **Balance Classes**: True
- **Train Samples (Original)**: 25077
- **Train Samples (Final)**: 26271

## Highlights
- Top-performing models in this run:
  - Naive_Bayes: 0.7555
  - Bayesian_Network: 0.7442
  - Neural_Network: 0.7438

## Cautions
- Lowest-performing models in this run:
  - SVM: 0.5119
  - Maximum_Entropy: 0.7170
  - Ensemble_Stacking: 0.7402

## Notes
This run used preprocessing_mode=model_specific, loss_mode=focal, svm_policy=freeze_v1_2.

v2.0 default recipe keeps ThaiTextAugmenter enabled for the training split.

## Training Details
- Total samples trained: 26271
- Original train samples: 25077
- Final train samples after augmentation: 26271
- Validation samples: 3344
- Test samples: 5015
- Shared feature dimension: 5000
- Models successfully trained: 6

## Augmentation
- Enabled: True
- Augment per sample: 2
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
