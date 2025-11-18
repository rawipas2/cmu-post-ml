# 🏗️ System Architecture

## ภาพรวมระบบ

```
┌─────────────────────────────────────────────────────────────────┐
│                   Thai Depression Classification                 │
│                        Machine Learning System                   │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ 1. DATA LAYER                                                   │
├─────────────────────────────────────────────────────────────────┤
│  📁 data/                                                       │
│     ├── train.json  (100,310 samples)                          │
│     ├── valid.json  (13,378 samples)                           │
│     └── test.json   (20,062 samples)                           │
│                                                                 │
│  📊 Preprocessing:                                              │
│     - PyThaiNLP tokenization                                    │
│     - TF-IDF vectorization (5000 features)                      │
│     - Label encoding                                            │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ 2. MODEL LAYER (6 Base Models)                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  🔵 Model 1: Support Vector Machine                            │
│     - cuML SVC (GPU accelerated)                               │
│     - RBF kernel                                               │
│     - C=1.0, gamma='scale'                                     │
│                                                                 │
│  🔵 Model 2: Neural Network                                    │
│     - PyTorch MLP                                              │
│     - Hidden: [512, 256, 128]                                  │
│     - Dropout: 0.3, BatchNorm                                  │
│                                                                 │
│  🔵 Model 3: Deep Learning                                     │
│     - Deep PyTorch NN                                          │
│     - Hidden: [1024, 512, 256, 128, 64]                        │
│     - Dropout: 0.4, LR Scheduler                               │
│                                                                 │
│  🔵 Model 4: Naive Bayes                                       │
│     - cuML MultinomialNB (GPU)                                 │
│     - Alpha: 1.0 (Laplace smoothing)                           │
│                                                                 │
│  🔵 Model 5: Bayesian Network                                  │
│     - Bayesian Neural Network                                  │
│     - Variational inference                                    │
│     - Hidden: [256, 128, 64]                                   │
│                                                                 │
│  🔵 Model 6: Maximum Entropy                                   │
│     - Logistic Regression (PyTorch)                            │
│     - L2 regularization: 0.01                                  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
                            ↓
         ┌──────────┬──────────┬──────────┬──────────┬──────────┐
         │  Pred 1  │  Pred 2  │  Pred 3  │  Pred 4  │  Pred 5  │ Pred 6
         └──────────┴──────────┴──────────┴──────────┴──────────┴────────
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ 3. ENSEMBLE LAYER (Stacking)                                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  🟢 Meta-Learner (Stacking Ensemble)                           │
│     - Input: 6 base model predictions                          │
│     - Architecture: [6 → 64 → 32 → 1]                          │
│     - PyTorch Neural Network                                   │
│     - Dropout: 0.3, 0.2                                        │
│     - Epochs: 100                                              │
│                                                                 │
│  🎯 Final Prediction                                            │
│     - Binary classification (depression / not)                 │
│     - Probability output                                       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ 4. EVALUATION LAYER                                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  📊 Metrics:                                                    │
│     - Accuracy (Target: ≥80%)                                  │
│     - Precision, Recall, F1-Score                              │
│     - AUC-ROC                                                  │
│                                                                 │
│  📈 Visualizations:                                             │
│     - Confusion matrices                                       │
│     - ROC curves                                               │
│     - Model comparison charts                                  │
│     - Classification reports                                   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ 5. PERSISTENCE LAYER                                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  💾 Model Storage:                                              │
│     - saved_models/*.pth                                       │
│     - versions/<version>/models/*.pth                          │
│                                                                 │
│  📁 Version Management:                                         │
│     - Auto version directories                                 │
│     - README.md generation                                     │
│     - Metrics tracking                                         │
│     - Plot storage                                             │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🔄 Training Pipeline Flow

```
START
  ↓
[Load Data] → train.json, valid.json, test.json
  ↓
[Preprocess] → Tokenize → TF-IDF → Encode Labels
  ↓
[Create Version Dir] → versions/v1.0/
  ↓
┌─────────────────────────────────────┐
│  FOR EACH BASE MODEL (6 models):   │
│    1. Initialize model (GPU)        │
│    2. Train on train data           │
│    3. Validate on valid data        │
│    4. Predict on test data          │
│    5. Calculate metrics             │
│    6. Save model (.pth)             │
│    7. Generate plots                │
│    8. Save metrics (.json)          │
└─────────────────────────────────────┘
  ↓
[Collect Base Predictions]
  ↓
[Train Ensemble Stacking]
  ├─ Use base predictions as features
  ├─ Train meta-learner
  ├─ Validate & evaluate
  └─ Save ensemble model
  ↓
[Generate Visualizations]
  ├─ Confusion matrices (all models)
  ├─ ROC curves (all models)
  └─ Model comparison chart
  ↓
[Save Results]
  ├─ Models → versions/v1.0/models/
  ├─ Plots → versions/v1.0/plots/
  └─ Metrics → versions/v1.0/metrics/
  ↓
[Generate README.md]
  ├─ Performance summary
  ├─ Strengths & weaknesses
  └─ Improvement suggestions
  ↓
[Check Target]
  └─ Accuracy >= 80% ?
      ├─ YES → ✅ Success!
      └─ NO  → 💡 Suggestions for next version
  ↓
END
```

---

## 🎮 GPU Acceleration Strategy

```
┌─────────────────────────────────────────────┐
│         GPU ACCELERATION LAYER              │
├─────────────────────────────────────────────┤
│                                             │
│  CUDA Core (NVIDIA GPU)                    │
│    ↓                     ↓                  │
│  PyTorch            cuML/cuPy               │
│    ↓                     ↓                  │
│  ├─ Neural Network    ├─ SVM               │
│  ├─ Deep Learning     └─ Naive Bayes       │
│  ├─ Bayesian Network                       │
│  ├─ Maximum Entropy                        │
│  └─ Ensemble Stacking                      │
│                                             │
│  ⚡ NO CPU COMPUTATION                      │
│     All models run on GPU                  │
│                                             │
└─────────────────────────────────────────────┘
```

---

## 📦 Module Structure

```
thai-depression/
│
├── 🔧 Configuration
│   └── config.py
│
├── 🎯 Core Scripts
│   ├── train.py              # Main training
│   ├── predict.py            # Inference
│   ├── compare_versions.py   # Version comparison
│   ├── setup_check.py        # System validation
│   └── test_models.py        # Model testing
│
├── 🤖 Models (models/)
│   ├── svm_model.py
│   ├── neural_network_model.py
│   ├── deep_learning_model.py
│   ├── naive_bayes_model.py
│   ├── bayesian_network_model.py
│   ├── maximum_entropy_model.py
│   └── ensemble_stacking.py
│
├── 🛠️ Utilities (utils/)
│   ├── data_loader.py        # Data processing
│   ├── evaluation.py         # Metrics & plots
│   └── version_manager.py    # Version control
│
├── 📊 Data (data/)
│   ├── train.json
│   ├── valid.json
│   └── test.json
│
├── 💾 Output
│   ├── saved_models/         # Model files
│   ├── results/              # Comparisons
│   └── versions/             # Version history
│       └── v1.0/
│           ├── README.md
│           ├── models/
│           ├── plots/
│           └── metrics/
│
└── 📖 Documentation
    ├── README.md
    ├── QUICKSTART.md
    ├── PROJECT_SUMMARY.md
    ├── SETUP_COMPLETE.md
    └── requirements.txt
```

---

## 🔄 Workflow Example

```
User → python train.py
  ↓
Load 133,750 samples from data/
  ↓
Tokenize with PyThaiNLP
  ↓
Vectorize with TF-IDF (5000 features)
  ↓
Train SVM on GPU (cuML) ────────┐
Train Neural Net on GPU (PyTorch)│
Train Deep Learning on GPU ─────┤→ 6 predictions
Train Naive Bayes on GPU ───────┤  per sample
Train Bayesian Net on GPU ──────┤
Train MaxEnt on GPU ────────────┘
  ↓
Stack predictions → Meta-learner (GPU)
  ↓
Evaluate all 7 models
  ↓
Generate plots & metrics
  ↓
Save to versions/v1.0/
  ↓
Generate README.md
  ↓
✅ Done! Check versions/v1.0/README.md
```

---

## 📊 Data Flow

```
Raw Text (Thai)
    ↓
[PyThaiNLP Tokenizer]
    ↓
Tokenized Text
    ↓
[TF-IDF Vectorizer]
    ↓
Feature Matrix (n_samples × 5000)
    ↓
[GPU Memory]
    ↓
    ├─→ [SVM] ──────┐
    ├─→ [NN] ───────┤
    ├─→ [DL] ───────┼─→ Predictions
    ├─→ [NB] ───────┤   (n_samples × 6)
    ├─→ [BN] ───────┤
    └─→ [ME] ───────┘
         ↓
    [Ensemble]
         ↓
  Final Prediction
```

---

**System designed for maximum performance and ease of customization!** 🚀
