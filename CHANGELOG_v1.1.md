# 📝 Changelog - Version 1.1

## 🎯 สรุปการปรับปรุง

Version 1.1 เป็นการปรับปรุงครั้งใหญ่จาก v1.0 โดยเน้นการใช้งาน **ครบทั้ง 6 models** และ **ปรับแต่งพารามิเตอร์** เพื่อเพิ่มความแม่นยำให้ถึงเป้าหมาย 80%+

---

## 🔥 การเปลี่ยนแปลงสำคัญ

### 1. ✅ ใช้ Model ครบทั้ง 6 ตัว

เพิ่มการ train models ที่ขาดหายไปใน v1.0:

- ✅ **SVM** (Support Vector Machine) - ปรับ C=10.0 เพื่อเพิ่มความซับซ้อน
- ✅ **Neural Network** - 3-layer network
- ✅ **Deep Learning** - 5-layer deep network
- ✅ **Naive Bayes** - ปรับ alpha=0.5 สำหรับ smoothing ที่ดีขึ้น
- ✅ **Bayesian Network** - Variational inference
- ✅ **Maximum Entropy** - Logistic regression with L2 regularization

### 2. 🎛️ ปรับปรุงพารามิเตอร์ทุก Model

#### **Global Parameters** (`config.py`)

```python
BATCH_SIZE = 32           # ลดจาก 64 เพื่อ stable gradient
EPOCHS = 100              # เพิ่มจาก 50 เพื่อ convergence ที่ดีขึ้น
LEARNING_RATE = 0.0005    # ลดจาก 0.001 เพื่อ fine-tuning
```

#### **Model-Specific Parameters**

- **Neural Network**: dropout=0.4, epochs=100, lr=0.0005
- **Deep Learning**: dropout=0.5, epochs=100, lr=0.0003
- **Bayesian Network**: epochs=100, lr=0.0005
- **Maximum Entropy**: l2_reg=0.01, epochs=100, lr=0.001
- **SVM**: C=10.0 (เพิ่มจาก 1.0), gamma='scale'
- **Naive Bayes**: alpha=0.5 (ลดจาก 1.0)

### 3. 🧠 ปรับปรุง Ensemble Stacking

```python
META_MODEL_EPOCHS = 150      # เพิ่มจาก 100
META_MODEL_LR = 0.0005       # ลดจาก 0.001
META_HIDDEN_DIM = 128        # เพิ่มจาก 64

# Architecture: 6 → 128 → 64 → 32 → 1 (เพิ่ม layer)
# เพิ่ม Dropout: 0.4 → 0.3 → 0.2
# เพิ่ม Learning Rate Scheduler
# เพิ่ม Gradient Clipping
# เพิ่ม Weight Decay (L2=1e-5)
```

---

## 🔧 การปรับปรุงโค้ด

### `config.py`

- ✅ เพิ่ม `MODEL_PARAMS` dictionary สำหรับแต่ละ model
- ✅ เปลี่ยน `CURRENT_VERSION` เป็น `v1.1`
- ✅ ปรับ global hyperparameters
- ✅ เพิ่ม `META_HIDDEN_DIM` สำหรับ ensemble

### `train.py`

- ✅ เพิ่มการ train SVM (ขาดหายไปใน v1.0)
- ✅ เพิ่มการ train Naive Bayes (ขาดหายไปใน v1.0)
- ✅ อัพเดตทุก model ให้ใช้ parameters จาก config
- ✅ เพิ่ม error handling และ traceback

### Model Files

#### `neural_network_model.py`

- ✅ เพิ่ม `dropout` parameter
- ✅ รองรับ custom dropout rate

#### `deep_learning_model.py`

- ✅ เพิ่ม `dropout` parameter
- ✅ รองรับ custom dropout rate

#### `svm_model.py`

- ✅ รองรับ `kernel`, `C`, `gamma` parameters
- ✅ ปรับ constructor ให้รับค่าจากภายนอก

#### `naive_bayes_model.py`

- ✅ รองรับ `alpha` parameter
- ✅ ปรับ constructor ให้รับค่าจากภายนอก

#### `ensemble_stacking.py`

- ✅ ขยาย meta-learner architecture (6 → 128 → 64 → 32 → 1)
- ✅ เพิ่ม Learning Rate Scheduler (ReduceLROnPlateau)
- ✅ เพิ่ม Gradient Clipping (max_norm=1.0)
- ✅ เพิ่ม Weight Decay (1e-5)
- ✅ เพิ่ม dropout layers (0.4, 0.3, 0.2)

---

## 📊 เป้าหมาย Performance

### v1.0 Results:

- Neural Network: ~73% accuracy
- Maximum Entropy: ~68% accuracy (imbalanced)
- Ensemble Stacking: ~73% accuracy

### v1.1 Target:

- ✅ ใช้ครบทั้ง 6 base models
- 🎯 Ensemble accuracy >= 80%
- 🎯 Individual models >= 75%
- 🎯 Better precision/recall balance

---

## 🚀 วิธีใช้งาน

```bash
# Train version 1.1
python train.py

# ผลลัพธ์จะถูกบันทึกใน:
versions/v1.1/
  ├── models/           # Saved models (.pth, .pkl)
  ├── metrics/          # JSON metrics
  ├── plots/            # Confusion matrices, ROC curves
  └── README.md         # Version summary
```

---

## 🔍 Technical Details

### Why These Changes?

1. **เพิ่ม Epochs (50→100)**: Models ต้องการเวลามากขึ้นเพื่อ converge
2. **ลด Learning Rate (0.001→0.0005)**: Fine-tuning ที่ละเอียดขึ้น
3. **ลด Batch Size (64→32)**: Gradient มี noise น้อยลงและ stable
4. **เพิ่ม Dropout**: ลด overfitting ใน neural networks
5. **SVM C=10**: เพิ่ม model complexity สำหรับ non-linear patterns
6. **Naive Bayes alpha=0.5**: Smoother probability estimates
7. **Ensemble Improvements**:
   - ขยาย capacity ด้วย hidden layers
   - Gradient clipping ป้องกัน exploding gradients
   - LR scheduler ปรับ learning rate อัตโนมัติ

---

## ⚠️ Breaking Changes

- ไม่มี breaking changes - backward compatible
- Models เก่าจาก v1.0 ยังโหลดได้ปกติ

---

## 📦 Dependencies

ไม่มีการเปลี่ยนแปลง - ใช้ dependencies เดิมจาก v1.0

---

## 👨‍💻 Developed By

Thai Depression Classification Team
Date: November 19, 2025
Version: 1.1
