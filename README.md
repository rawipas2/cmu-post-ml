# Thai Depression Classification System

ระบบการจำแนกภาวะซึมเศร้าจากข้อความภาษาไทยโดยใช้ Machine Learning และ Deep Learning

## 🎯 เป้าหมาย

สร้างระบบการจำแนกที่มีความแม่นยำมากกว่า **80%** โดยใช้ Ensemble Learning แบบ Stacking จาก 6 models:

1. **Support Vector Machine (SVM)** - GPU accelerated with cuML
2. **Neural Network** - PyTorch MLP
3. **Deep Learning** - Deep neural network with multiple layers
4. **Naive Bayes** - Multinomial Naive Bayes with cuML
5. **Bayesian Network** - Neural network with Bayesian inference
6. **Maximum Entropy** - Logistic Regression (MaxEnt)

## 📋 ความต้องการของระบบ

### Hardware Requirements

- **GPU**: NVIDIA GPU with CUDA support (Required!)
- **RAM**: 16GB+ recommended
- **Storage**: 5GB+ free space

### Software Requirements

- **Python**: 3.10 (Required)
- **CUDA**: 11.x or higher
- **cuDNN**: Compatible version with CUDA

## 🚀 การติดตั้ง

### ขั้นตอนที่ 0: ตรวจสอบ Python 3.10

```bash
# ตรวจสอบเวอร์ชัน Python
python --version

# หรือใช้สคริปต์ตรวจสอบ (Windows)
python check_python_version.py
# หรือ
check_python.bat
```

**หากยังไม่มี Python 3.10:**

- ดาวน์โหลดจาก: https://www.python.org/downloads/release/python-31011/
- หรือใช้ conda: `conda create -n thai-depression python=3.10`

### 1. ติดตั้ง CUDA และ cuDNN

ดาวน์โหลดและติดตั้ง CUDA Toolkit:

- [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)

### 2. ติดตั้ง Python Dependencies

```bash
# สร้าง virtual environment (แนะนำ)
python -m venv venv

# เปิดใช้งาน virtual environment
# Windows PowerShell:
.\venv\Scripts\Activate.ps1

# Windows CMD:
.\venv\Scripts\activate.bat

# ติดตั้ง PyTorch with CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# ติดตั้ง RAPIDS AI (cuML, cuPy)
# สำหรับ CUDA 11.x:
pip install cuml-cu11 cupy-cuda11x

# หรือใช้ conda (แนะนำสำหรับ RAPIDS):
conda install -c rapidsai -c conda-forge -c nvidia cuml cupy

# ติดตั้ง dependencies อื่นๆ
pip install -r requirements.txt
```

### 3. ตรวจสอบการติดตั้ง

```python
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"
python -c "import cupy; print('CuPy installed successfully')"
python -c "import cuml; print('cuML installed successfully')"
```

## 📁 โครงสร้างโปรเจกต์

```
thai-depression/
├── data/                      # Dataset files
│   ├── train.json
│   ├── valid.json
│   └── test.json
├── models/                    # Model implementations
│   ├── svm_model.py          # SVM with cuML
│   ├── neural_network_model.py
│   ├── deep_learning_model.py
│   ├── naive_bayes_model.py
│   ├── bayesian_network_model.py
│   ├── maximum_entropy_model.py
│   └── ensemble_stacking.py   # Stacking ensemble
├── utils/                     # Utility functions
│   ├── data_loader.py        # Data loading & preprocessing
│   ├── evaluation.py         # Metrics & visualization
│   └── version_manager.py    # Version control & docs
├── saved_models/             # Trained model files
├── results/                  # Training results
├── versions/                 # Version history with README
├── config.py                 # Configuration file
├── train.py                  # Main training script
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## 🎓 การใช้งาน

### การ Train Models

```bash
# รัน training pipeline
python train.py

# Baseline run without augmentation
python train.py --no-augmentation

# Custom augmentation intensity
python train.py --aug-per-sample 2 --no-balance-classes
```

Script จะทำการ:

1. โหลดและ augment training split (v2.0 default)
2. โหลดและ preprocess ข้อมูล
3. Train แต่ละ model ทั้ง 6 ตัว
4. Train ensemble stacking model
5. สร้างกราฟและสถิติ
6. บันทึก models และผลลัพธ์
7. สร้าง README.md สำหรับ version นั้นๆ

### v2.0 Augmentation Controls

- `--use-augmentation` / `--no-augmentation`
- `--aug-per-sample <int>`
- `--balance-classes` / `--no-balance-classes`

Default behavior in `v2.0`:

- augmentation enabled
- `aug_per_sample=1`
- `balance_classes=True`
- augmentation applied only to the training split

Generated metrics and README files now record:

- augmentation enabled/disabled
- augmentation intensity
- class balancing mode
- original vs final training sample counts

### การปรับแต่ง Hyperparameters

แก้ไขไฟล์ `config.py`:

```python
# Model parameters
MAX_FEATURES = 5000        # TF-IDF features
BATCH_SIZE = 64           # Training batch size
EPOCHS = 50               # Training epochs
LEARNING_RATE = 0.001     # Learning rate

# Ensemble parameters
META_MODEL_EPOCHS = 100
META_MODEL_LR = 0.001

# Target
TARGET_ACCURACY = 0.80
```

### การปรับแต่งแต่ละ Model

แต่ละ model สามารถปรับแต่งได้อย่างอิสระใน `train.py`:

```python
# ตัวอย่าง: ปรับ Neural Network
nn = neural_network_model.create_model(
    input_dim=input_dim,
    hidden_dims=[512, 256, 128],  # เปลี่ยน architecture
    learning_rate=0.001,
    epochs=50
)
```

## 📊 ผลลัพธ์และรายงาน

หลังจาก training เสร็จ จะมีการสร้าง:

### ใน `versions/<version_name>/`:

- **models/** - ไฟล์ model ที่ save ไว้
- **plots/** - Confusion matrices, ROC curves, comparisons
- **metrics/** - JSON files ที่มี metrics ละเอียด
- **README.md** - สรุปผลลัพธ์, จุดแข็ง/จุดอ่อน, ข้อเสนอแนะ

### Metrics ที่วัด:

- Accuracy
- Precision
- Recall
- F1 Score
- AUC-ROC

## 🔧 การแก้ไขปัญหาที่พบบ่อย

### CUDA Not Available

```bash
# ตรวจสอบ CUDA
nvidia-smi

# ติดตั้ง PyTorch ที่ตรงกับ CUDA version
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### cuML Import Error

```bash
# ใช้ conda สำหรับ RAPIDS (แนะนำ)
conda create -n thai-depression python=3.9
conda activate thai-depression
conda install -c rapidsai -c conda-forge -c nvidia cuml cupy
```

### Out of Memory (OOM)

- ลด `BATCH_SIZE` ใน config.py
- ลด `MAX_FEATURES`
- ลด hidden layer dimensions
- ใช้ GPU ที่มี memory มากกว่า

### Accuracy ต่ำกว่าเป้าหมาย

1. เพิ่ม `EPOCHS` สำหรับ training นานขึ้น
2. ปรับ `LEARNING_RATE` (ลองทั้งเพิ่มและลด)
3. เพิ่ม `MAX_FEATURES` สำหรับ features มากขึ้น
4. ลอง data augmentation
5. ใช้ pre-trained Thai language models

## 📝 Version Management

ระบบจะสร้าง version directory อัตโนมัติพร้อม:

- Training results
- Model files
- Visualizations
- Documentation (README.md)
- Metrics comparison

ตัวอย่าง version structure:

```
versions/
├── v1.0/
│   ├── README.md          # Version summary
│   ├── models/            # Saved models
│   ├── plots/             # Visualizations
│   └── metrics/           # Performance metrics
└── v2.0/
    └── ...
```

## 🎯 เป้าหมายต่อไป

1. ✅ สร้างระบบ ensemble stacking
2. ⬜ Fine-tune hyperparameters
3. ⬜ ใช้ pre-trained models (WangchanBERTa, PhayaThaiBERT)
4. ⬜ Data augmentation
5. ⬜ Cross-validation
6. ⬜ Deploy as REST API

## 📚 เอกสารอ้างอิง

- [PyTorch](https://pytorch.org/)
- [RAPIDS AI](https://rapids.ai/)
- [PyThaiNLP](https://github.com/PyThaiNLP/pythainlp)
- [scikit-learn](https://scikit-learn.org/)

## 📄 License

MIT License

## 👨‍💻 Author

Thai Depression Classification System
