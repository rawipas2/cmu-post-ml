# Thai Depression Classification - factorial_model_specific-focal

## Overview
Training run completed on: 2026-04-01 00:28:53

## Performance Summary
- **Best Single Model**: Naive_Bayes (0.7617)
- **Average Accuracy**: 0.7079
- **Ensemble Accuracy**: 0.7438

## Individual Model Performance

| Model | Accuracy | Precision | Recall | F1 Score | AUC-ROC |
|---|---:|---:|---:|---:|---:|
| Naive_Bayes | 0.7617 | 0.7617 | 0.7617 | 0.7616 | 0.8459 |
| Bayesian_Network | 0.7454 | 0.7510 | 0.7454 | 0.7432 | 0.8294 |
| Ensemble_Stacking | 0.7438 | 0.7437 | 0.7438 | 0.7438 | 0.8229 |
| Deep_Learning | 0.7420 | 0.7433 | 0.7420 | 0.7412 | 0.8239 |
| Neural_Network | 0.7408 | 0.7430 | 0.7408 | 0.7397 | 0.8197 |
| Maximum_Entropy | 0.7097 | 0.7754 | 0.7097 | 0.6886 | 0.8464 |
| SVM | 0.5119 | 0.2620 | 0.5119 | 0.3466 | 0.7899 |

## Experiment Configuration
- **Experiment Name**: factorial_model_specific-focal
- **Version Name**: factorial_model_specific-focal
- **Preprocessing Mode**: model_specific
- **Loss Mode**: focal
- **SVM Policy**: freeze_v1_2

## Highlights
- Top-performing models in this run:
  - Naive_Bayes: 0.7617
  - Bayesian_Network: 0.7454
  - Ensemble_Stacking: 0.7438

## Cautions
- Lowest-performing models in this run:
  - SVM: 0.5119
  - Maximum_Entropy: 0.7097
  - Neural_Network: 0.7408

## Notes
This run used preprocessing_mode=model_specific, loss_mode=focal, svm_policy=freeze_v1_2.

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
