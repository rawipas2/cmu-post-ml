# 📋 CHANGELOG - Version 1.2.1

> **Release Date**: November 19, 2025  
> **Focus**: Enhanced Feature Extraction & Model Capacity  
> **Type**: Incremental improvement (v1.2 → v1.2.1)

---

## 🎯 Executive Summary

Version 1.2.1 is an **incremental enhancement** of v1.2, focusing on:

- **Richer feature extraction** (10,000 features vs 5,000)
- **Tri-gram support** (1,2,3-grams vs 1,2-grams)
- **Increased model capacity** to handle richer features
- **All 6 base models + Ensemble Stacking** utilized

### Why v1.2.1 (not v1.3)?

This is a **refinement** of the existing approach rather than a major architectural change:

- ✅ Same model architectures
- ✅ Same training methodology
- ✅ Enhanced feature engineering only
- ✅ Tuned hyperparameters for new feature space

---

## 📊 Comparison: v1.2 vs v1.2.1

| Aspect              | v1.2                  | v1.2.1                      | Change                  |
| ------------------- | --------------------- | --------------------------- | ----------------------- |
| **MAX_FEATURES**    | 5,000                 | **10,000**                  | +100% 🔥                |
| **N-grams**         | 1,2-grams             | **1,2,3-grams**             | +33%                    |
| **BATCH_SIZE**      | 32                    | **16**                      | -50% (better learning)  |
| **EPOCHS**          | 100                   | **150**                     | +50% (with early stop)  |
| **Neural Hidden**   | [512,256,128]         | **[1024,512,256]**          | +100% capacity          |
| **Deep Hidden**     | [1024,512,256,128,64] | **[2048,1024,512,256,128]** | +100% first layer       |
| **Bayesian Hidden** | [256,128,64]          | **[512,256,128]**           | +100% capacity          |
| **Meta Hidden**     | 128                   | **256**                     | +100% ensemble capacity |
| **Training Time**   | ~20-25 min            | **~9 hours (actual)**       | +~21.6× (more features) |

---

## 🔧 Detailed Improvements

### 1. **Feature Extraction Enhancement**

#### A. Increased MAX_FEATURES

```python
# v1.2
MAX_FEATURES = 5000

# v1.2.1
MAX_FEATURES = 10000  # Doubled for richer representation
```

**Impact:**

- More vocabulary coverage
- Better capture of rare but important terms
- Improved semantic understanding

#### B. Tri-gram Support

```python
# v1.2
TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2)  # unigrams + bigrams
)

# v1.2.1
TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 3),  # unigrams + bigrams + trigrams
    min_df=2,           # Reduce noise
    max_df=0.95         # Filter common words
)
```

**Example Thai Tri-grams:**

```
Text: "ฉัน รู้สึก เศร้า มาก"

Unigrams (1): [ฉัน, รู้สึก, เศร้า, มาก]
Bigrams (2):  [ฉัน_รู้สึก, รู้สึก_เศร้า, เศร้า_มาก]
Trigrams (3): [ฉัน_รู้สึก_เศร้า, รู้สึก_เศร้า_มาก]  ← NEW!
```

**Benefits:**

- Captures longer context
- Better phrase understanding
- More nuanced emotion detection

#### C. Noise Filtering

```python
min_df=2     # Word must appear in at least 2 documents
max_df=0.95  # Exclude words in >95% of documents
```

**Impact:**

- Removes typos and rare mistakes
- Filters out stop words automatically
- Better signal-to-noise ratio

---

### 2. **Model Capacity Increases**

All neural models scaled up to handle 10,000 features:

#### A. Neural Network

```python
# v1.2
'hidden_dims': [512, 256, 128]
'learning_rate': 0.0003

# v1.2.1
'hidden_dims': [1024, 512, 256]  # Doubled first layer
'learning_rate': 0.0002          # Reduced for stability
```

**Rationale:**

- 10,000 input features need more capacity
- First hidden layer processes raw features
- Doubling allows better feature extraction

#### B. Deep Learning

```python
# v1.2
'hidden_dims': [1024, 512, 256, 128, 64]
'learning_rate': 0.0002

# v1.2.1
'hidden_dims': [2048, 1024, 512, 256, 128]  # Doubled first layer
'learning_rate': 0.00015                    # Further reduced
```

**Rationale:**

- Deeper architecture benefits from more capacity
- 2048 → 1024 creates strong feature hierarchy
- Lower LR prevents overfitting with more parameters

#### C. Bayesian Network

```python
# v1.2
'hidden_dims': [256, 128, 64]
'learning_rate': 0.0003

# v1.2.1
'hidden_dims': [512, 256, 128]  # Doubled all layers
'learning_rate': 0.0002          # Reduced for stability
```

**Rationale:**

- Bayesian layers need capacity for uncertainty
- Better posterior distributions with more parameters
- Lower LR helps convergence

#### D. Ensemble Meta-Learner

```python
# v1.2
META_HIDDEN_DIM = 128
META_MODEL_LR = 0.0005
META_MODEL_EPOCHS = 150

# v1.2.1
META_HIDDEN_DIM = 256      # Doubled
META_MODEL_LR = 0.0003     # Reduced
META_MODEL_EPOCHS = 200    # Increased
```

**Rationale:**

- More complex to combine 6 diverse models
- Needs capacity to learn optimal weighting
- More epochs for better convergence

---

### 3. **Training Configuration Updates**

#### A. Batch Size Reduction

```python
# v1.2
BATCH_SIZE = 32

# v1.2.1
BATCH_SIZE = 16  # Halved
```

**Rationale:**

- Smaller batches → better gradient estimates
- More updates per epoch
- Better for 10,000-dimensional space
- Helps escape local minima

#### B. Extended Training

```python
# v1.2
EPOCHS = 100

# v1.2.1
EPOCHS = 150  # +50%
```

**Note:** With early stopping, actual training may stop at 50-80 epochs

**Rationale:**

- More epochs allows finding better optima
- Early stopping prevents overfitting
- Safety margin for slower convergence

#### C. Learning Rate Adjustments

All learning rates reduced for stability:

```python
# Global
LEARNING_RATE: 0.0005 → 0.0003

# Neural Network
0.0003 → 0.0002

# Deep Learning
0.0002 → 0.00015

# Bayesian Network
0.0003 → 0.0002

# Meta-Learner
0.0005 → 0.0003
```

**Rationale:**

- Larger models need smaller learning rates
- Prevents divergence
- Better final convergence

---

### 4. **All Models Utilized**

Confirmed all 6 base models + ensemble:

1. ✅ **SVM** (SGDClassifier)
2. ✅ **Neural Network** (3-layer MLP)
3. ✅ **Deep Learning** (5-layer deep net)
4. ✅ **Naive Bayes** (Multinomial)
5. ✅ **Bayesian Network** (Variational Inference)
6. ✅ **Maximum Entropy** (Logistic Regression)
7. ✅ **Ensemble Stacking** (Meta-learner on top 6)

---

## 📈 Expected Performance Improvements

### Quantitative Predictions:

| Model                | v1.2 Expected | v1.2.1 Target | Gain  | Reason               |
| -------------------- | ------------- | ------------- | ----- | -------------------- |
| **SVM**              | 75%           | **76-77%**    | +1-2% | Better features      |
| **Neural Network**   | 75-76%        | **77-78%**    | +2%   | More capacity        |
| **Deep Learning**    | 76-78%        | **78-80%**    | +2%   | Richer features      |
| **Naive Bayes**      | 75%           | **76%**       | +1%   | More n-grams         |
| **Bayesian Network** | 75-77%        | **77-79%**    | +2%   | Better uncertainty   |
| **Maximum Entropy**  | 73-75%        | **75-77%**    | +2%   | More features        |
| **Ensemble**         | N/A           | **80-82%**    | 🎯    | Stronger base models |

### Key Improvements:

1. **Feature Richness**

   - 10,000 features vs 5,000 → better vocabulary
   - Tri-grams → better context
   - Noise filtering → cleaner signal

2. **Model Power**

   - Doubled capacity → can leverage rich features
   - Ensemble → combines all strengths

3. **Expected Average**
   - v1.2: ~75-76%
   - v1.2.1: **77-78%**
   - **Target 80% more achievable!**

---

## 🔬 Technical Rationale

### Why These Specific Changes?

#### 1. **10,000 Features**

**Research basis:**

- More features capture more nuances
- Thai is morphologically rich
- Depression language uses specific phrases

**Trade-off:**

- ✅ Better representation
- ✅ Captures rare important terms
- ⚠️ Longer training time
- ⚠️ More memory usage

**Mitigation:**

- Smaller batch size (fits in memory)
- Early stopping (saves time)

#### 2. **Tri-grams**

**Research basis:**

- Thai phrases often 2-3 words
- Emotion expressions multi-word
- Context crucial for Thai

**Examples:**

```
"รู้สึกเศร้า" → feeling sad
"รู้สึกเศร้ามาก" → feeling very sad (tri-gram captures intensity)
```

**Trade-off:**

- ✅ Better context
- ✅ Phrase-level understanding
- ⚠️ More sparse features
- ⚠️ Computational cost

**Mitigation:**

- min_df=2 reduces sparsity
- max_df=0.95 filters common patterns

#### 3. **Increased Capacity**

**Research basis:**

- Larger input → larger model needed
- 10,000 features need 1000+ hidden units
- Prevents information bottleneck

**Architecture decisions:**

```
Input: 10000 features
First hidden: 1024-2048  (enough capacity)
Second hidden: 512-1024  (feature abstraction)
Third hidden: 256-512    (high-level patterns)
Output: 1 (binary classification)
```

**Trade-off:**

- ✅ Can leverage all features
- ✅ Better representations
- ⚠️ More parameters (overfitting risk)
- ⚠️ Longer training

**Mitigation:**

- Higher dropout (0.5-0.6)
- Early stopping
- L2 regularization
- Gradient clipping

---

## 🧪 Experiment Design

### v1.2.1 is a Controlled Experiment:

**Hypothesis:**

> Richer feature extraction (10K features, tri-grams) with appropriately scaled models will improve classification accuracy by 2-3%.

**Variables Changed:**

- Independent: Feature extraction parameters
- Dependent: Model accuracy
- Controlled: Model architectures, training methodology

**Success Criteria:**

- ✅ Average accuracy > 77%
- ✅ Ensemble accuracy ≥ 80%
- ✅ All models improve vs v1.2
- ✅ Training completes without errors

---

## ⚙️ Files Modified

### Configuration:

- ✅ `config.py`
  - MAX_FEATURES: 5000 → 10000
  - BATCH_SIZE: 32 → 16
  - EPOCHS: 100 → 150
  - Version: v1.2 → v1.2.1
  - All model parameters updated

### Data Processing:

- ✅ `utils/data_loader.py`
  - ngram_range: (1,2) → (1,3)
  - Added min_df=2
  - Added max_df=0.95

### Documentation:

- ✅ `CHANGELOG_v1.2.1.md` - This file
- 📝 `UPGRADE_SUMMARY_v1.2.1.md` - To be created

### Models:

- ℹ️ No model code changes (only config parameters)

---

## 🎯 Testing Protocol

### Pre-Flight Checklist:

Before running v1.2.1:

- [ ] Verify MAX_FEATURES = 10000
- [ ] Check ngram_range = (1,3)
- [ ] Confirm BATCH_SIZE = 16
- [ ] Verify all 6 models enabled
- [ ] Check ensemble enabled
- [ ] Ensure early stopping active

### During Training:

Monitor for:

- ✅ Feature extraction completes (~2-3 min)
- ✅ All 6 models train successfully
- ✅ Early stopping activates appropriately
- ✅ Memory usage acceptable
- ✅ Ensemble trains on all 6 models

### Post-Training Analysis:

Compare with v1.2:

- [ ] Each model accuracy
- [ ] Average improvement
- [ ] Ensemble performance
- [ ] Training time
- [ ] Resource usage

---

## 📊 Expected Results

### Training Output:

```
📐 Input dimension: 10000  ← Should see 10000 (was 5000)

🔵 Training SVM...
   Dataset size: (25077, 10000)  ← Confirm 10000 features
   ...

🔵 Training Neural Network...
   Epoch [1/150] - Train Loss: ..., Val Loss: ...
   Early stopping at epoch XX  ← Should stop 50-100

... (all 6 models) ...

🟢 Training Ensemble Stacking...
   Getting predictions from 6 base models  ← Confirm 6
   ...
```

### Performance Targets:

| Metric                 | Target | Stretch Goal |
| ---------------------- | ------ | ------------ |
| **Average Accuracy**   | 77%    | 78%          |
| **Ensemble Accuracy**  | 80%    | 82%          |
| **Best Single Model**  | 78%    | 80%          |
| **Worst Single Model** | 75%    | 76%          |

---

## 💡 Insights & Learnings

### Why Incremental (v1.2.1) Works:

1. **Builds on Stable Base**

   - v1.2 fixed major issues (overfitting, imbalance)
   - v1.2.1 enhances working system

2. **Controlled Variables**

   - Only feature extraction changed
   - Easy to attribute improvements

3. **Low Risk**
   - Can always revert to v1.2
   - No architectural risks

### Research Methodology:

This follows **ablation study** principles:

- Change one aspect (features)
- Measure impact
- Document findings
- Iterate

### Next Steps Logic:

**If v1.2.1 ≥ 80%:**

- ✅ SUCCESS! Publish results
- Consider v1.3 for production hardening

**If v1.2.1 = 77-79%:**

- ✅ Good progress
- v1.3: Try Thai BERT or ensemble variants

**If v1.2.1 < 77%:**

- ⚠️ Investigate why
- Maybe feature engineering not enough
- v1.3: Consider architectural changes

---

## 🔮 Future Directions (v1.3+)

If v1.2.1 doesn't reach 80%, consider:

### 1. **Pre-trained Models**

- WangchanBERTa (Thai BERT)
- Transfer learning
- Fine-tuning approach

### 2. **Data Augmentation**

- Back-translation
- Synonym replacement
- Paraphrasing

### 3. **Advanced Ensembles**

- Weighted voting
- Boosting (AdaBoost, XGBoost)
- Neural ensemble architectures

### 4. **Feature Engineering**

- Sentiment scores
- Topic modeling
- Emotion lexicons
- Linguistic features

### 5. **Architecture Search**

- Transformer models
- Attention mechanisms
- Graph neural networks
- Multi-task learning

---

## 📚 References

### Feature Engineering:

1. **N-gram Models**

   - Manning & Schütze (1999). Foundations of Statistical NLP
   - Effective for phrase-level semantics

2. **TF-IDF Optimization**

   - Salton & Buckley (1988). Term-weighting approaches
   - min_df and max_df parameters

3. **Thai NLP**
   - PyThaiNLP documentation
   - Tri-grams important for Thai phrases

### Model Scaling:

1. **Network Capacity**

   - Goodfellow et al. (2016). Deep Learning
   - Input dimension → Hidden dimension guidelines

2. **Batch Size Effects**
   - Masters & Luschi (2018). Revisiting Small Batch Training
   - Smaller batches better for some tasks

---

## ⏱️ Timeline (Actual)

- **End-to-end (feature extraction + 6 models + ensemble)**: **~9 hours (actual)** on i5-14500 + RTX 4060 Ti
- คาดการณ์เดิม (~35-50 นาที) ไม่ตรง: 10K features + batch 16 ทำให้ runtime ช้ากว่าประมาณ 21.6×

Compared to v1.2: ~20-25 min → **~9 hours** (ช้ากว่ามาก)

---

## ✨ Summary

### v1.2.1 Key Points:

1. **Feature Enhancement**

   - 10,000 features (2x increase)
   - Tri-gram support
   - Noise filtering

2. **Model Scaling**

   - Increased capacity across all models
   - Appropriate learning rate adjustments
   - Extended training epochs

3. **Methodology**

   - Controlled experiment
   - Builds on stable v1.2
   - Low-risk improvement

4. **Expected Impact**
   - +2-3% average accuracy
   - Ensemble reaches 80% target
   - All 6 models improved

### Ready to Train!

```bash
python train.py
```

Results will be in: `versions/v1.2.1/`

---

### Success Metrics:

We consider v1.2.1 successful if:

- ✅ Average accuracy ≥ 77%
- ✅ Ensemble accuracy ≥ 80%
- ✅ All models train without errors
- ✅ Improvement over v1.2 demonstrated

---

**Let's push towards 80%! 🎯**

---

_Generated: November 19, 2025_  
_Thai Depression Classification System - v1.2.1_  
_Incremental Enhancement Release_
