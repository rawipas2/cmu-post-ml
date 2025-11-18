# 🚀 Thai Depression Classification System - Complete Setup

## ✅ โปรเจกต์ได้ถูกสร้างเสร็จสมบูรณ์แล้ว!

### 📁 ไฟล์ทั้งหมดที่สร้าง (23 ไฟล์)

#### Core Files

- ✅ `config.py` - Configuration และ settings
- ✅ `train.py` - Main training pipeline
- ✅ `predict.py` - Inference script
- ✅ `compare_versions.py` - Version comparison
- ✅ `setup_check.py` - System validation
- ✅ `test_models.py` - Model testing

#### Models (7 ไฟล์)

- ✅ `models/__init__.py`
- ✅ `models/svm_model.py` - SVM with cuML (GPU)
- ✅ `models/neural_network_model.py` - PyTorch NN
- ✅ `models/deep_learning_model.py` - Deep NN
- ✅ `models/naive_bayes_model.py` - Naive Bayes (GPU)
- ✅ `models/bayesian_network_model.py` - Bayesian NN
- ✅ `models/maximum_entropy_model.py` - MaxEnt
- ✅ `models/ensemble_stacking.py` - Stacking Ensemble

#### Utils (4 ไฟล์)

- ✅ `utils/__init__.py`
- ✅ `utils/data_loader.py` - Data processing
- ✅ `utils/evaluation.py` - Metrics & visualization
- ✅ `utils/version_manager.py` - Version control

#### Documentation (6 ไฟล์)

- ✅ `README.md` - Full documentation
- ✅ `QUICKSTART.md` - Quick start guide
- ✅ `PROJECT_SUMMARY.md` - Project overview
- ✅ `VERSION_README_TEMPLATE.md` - Template info
- ✅ `requirements.txt` - Dependencies
- ✅ `.gitignore` - Git ignore rules

---

## 🎯 ระบบที่สร้างมีความสามารถ:

### ✨ Features

1. **6 Machine Learning Models**

   - Support Vector Machine (GPU)
   - Neural Network (PyTorch)
   - Deep Learning (PyTorch)
   - Naive Bayes (GPU)
   - Bayesian Network (PyTorch)
   - Maximum Entropy (PyTorch)

2. **Ensemble Learning**

   - Stacking meta-learner
   - รวม predictions จาก 6 models
   - Neural network สำหรับ combining

3. **GPU Acceleration**

   - ใช้ CUDA สำหรับ PyTorch
   - ใช้ cuML/cuPy สำหรับ SVM & NB
   - ไม่มีการใช้ CPU

4. **Thai Language Support**

   - PyThaiNLP tokenization
   - TF-IDF vectorization
   - รองรับภาษาไทยเต็มรูปแบบ

5. **Complete Pipeline**

   - Data loading & preprocessing
   - Model training
   - Evaluation & metrics
   - Visualization (plots, charts)
   - Model persistence (save/load)

6. **Version Management**

   - Auto version directories
   - README generation
   - Metrics tracking
   - Model comparison

7. **Comprehensive Evaluation**
   - Accuracy, Precision, Recall, F1
   - AUC-ROC
   - Confusion matrices
   - ROC curves
   - Classification reports
   - Model comparison charts

---

## 🚀 วิธีเริ่มต้นใช้งาน

### Step 1: ติดตั้ง Dependencies

```powershell
# สร้าง virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# ติดตั้ง PyTorch with CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# สำหรับ RAPIDS AI (cuML, cuPy) แนะนำใช้ conda:
conda create -n thai-depression python=3.9
conda activate thai-depression
conda install -c rapidsai -c conda-forge -c nvidia cuml cupy

# ติดตั้ง dependencies อื่นๆ
pip install pythainlp scikit-learn matplotlib seaborn pandas
```

### Step 2: ตรวจสอบระบบ

```powershell
python setup_check.py
```

ต้องผ่านทุก check:

- ✅ Python version
- ✅ CUDA & GPU
- ✅ cuML
- ✅ CuPy
- ✅ PyThaiNLP
- ✅ Other dependencies
- ✅ Data files

### Step 3: (Optional) Test Models

```powershell
python test_models.py
```

### Step 4: เริ่ม Training

```powershell
python train.py
```

---

## 📊 Output ที่จะได้

หลังจาก training เสร็จ:

```
versions/v1.0/
├── README.md                          # Auto-generated summary
├── models/                            # Saved models
│   ├── SVM.pth
│   ├── Neural_Network.pth
│   ├── Deep_Learning.pth
│   ├── Naive_Bayes.pth
│   ├── Bayesian_Network.pth
│   ├── Maximum_Entropy.pth
│   └── Ensemble_Stacking.pth
├── plots/                             # Visualizations
│   ├── SVM_confusion_matrix.png
│   ├── SVM_roc_curve.png
│   ├── SVM_classification_report.txt
│   ├── ... (same for all models)
│   └── model_comparison.png
└── metrics/                           # JSON metrics
    ├── SVM_20251118_153000.json
    └── ... (one per model)
```

---

## ⚙️ การ Customize

### แก้ไข Hyperparameters

ไฟล์ `config.py`:

```python
MAX_FEATURES = 5000     # เพิ่ม/ลด features
BATCH_SIZE = 64         # ลดถ้า OOM
EPOCHS = 50             # เพิ่มสำหรับ training นานขึ้น
LEARNING_RATE = 0.001   # ปรับ learning rate
```

### แก้ไข Model Architecture

แต่ละไฟล์ใน `models/`:

```python
# ตัวอย่าง: neural_network_model.py
nn = neural_network_model.create_model(
    input_dim=input_dim,
    hidden_dims=[1024, 512, 256],  # เปลี่ยน layers
    learning_rate=0.0001,
    epochs=100
)
```

---

## 🎯 Target & Goals

- **Target Accuracy**: ≥ 80%
- **Method**: Ensemble Stacking
- **Hardware**: GPU only (no CPU)
- **Models**: 6 base + 1 ensemble

---

## 📚 คำสั่งที่ใช้บ่อย

```powershell
# ตรวจสอบระบบ
python setup_check.py

# Test models
python test_models.py

# Train
python train.py

# Predict
python predict.py --text "ข้อความ" --version v1.0

# Compare versions
python compare_versions.py
```

---

## 🔧 Troubleshooting

### CUDA Not Available

```powershell
nvidia-smi  # ตรวจสอบ GPU
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### cuML Import Error

```powershell
conda install -c rapidsai -c conda-forge -c nvidia cuml cupy
```

### Out of Memory

แก้ไข `config.py`:

- ลด `BATCH_SIZE`
- ลด `MAX_FEATURES`
- ลด hidden dimensions

---

## 📈 การพัฒนาต่อ

1. Fine-tune hyperparameters
2. ใช้ pre-trained models (WangchanBERTa)
3. Data augmentation
4. Cross-validation
5. Grid/Random search
6. Deploy as API

---

## ✅ Checklist ก่อนเริ่ม

- [ ] Python 3.8-3.10 installed
- [ ] NVIDIA GPU with CUDA
- [ ] Dependencies installed
- [ ] Data files in `data/`
- [ ] `python setup_check.py` passed
- [ ] Ready to train!

---

## 🎉 สรุป

โปรเจกต์พร้อมใช้งาน 100%!

**คุณสมบัติครบถ้วน:**

- ✅ 6 base models (แยกไฟล์)
- ✅ Ensemble stacking
- ✅ GPU acceleration
- ✅ Thai text processing
- ✅ Complete evaluation
- ✅ Version management
- ✅ Auto documentation
- ✅ Easy customization

**เริ่มใช้งานได้ทันที:**

```powershell
python train.py
```

Good luck with your research! 🚀🎓

---

## 📧 Support

หากมีปัญหา:

1. ดู `README.md` สำหรับ documentation ครบถ้วน
2. ดู `QUICKSTART.md` สำหรับเริ่มต้นอย่างรวดเร็ว
3. ดู `PROJECT_SUMMARY.md` สำหรับภาพรวมโปรเจกต์
4. รัน `python setup_check.py` เพื่อตรวจสอบระบบ

---

**Created with ❤️ for Thai Depression Classification Research**
