# Thai Depression Classification - Quick Start Guide

## 🚀 เริ่มต้นอย่างรวดเร็ว

### ขั้นตอนที่ 1: ติดตั้ง Dependencies

```powershell
# สร้าง virtual environment
python -m venv venv

# เปิดใช้งาน
.\venv\Scripts\Activate.ps1

# ติดตั้ง PyTorch with CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# ติดตั้ง dependencies
pip install -r requirements.txt

# ติดตั้ง RAPIDS AI (cuML, cuPy) - แนะนำใช้ conda
# สร้าง conda environment
conda create -n thai-depression python=3.9
conda activate thai-depression
conda install -c rapidsai -c conda-forge -c nvidia cuml cupy
pip install pythainlp scikit-learn matplotlib seaborn pandas
```

### ขั้นตอนที่ 2: ตรวจสอบระบบ

```powershell
python setup_check.py
```

### ขั้นตอนที่ 3: เริ่ม Training

```powershell
python train.py
```

## 📋 คำสั่งที่ใช้บ่อย

### Training

```powershell
# Train ทุก models พร้อม ensemble
python train.py
```

### Prediction

```powershell
# ทำนายข้อความเดี่ยว
python predict.py --text "ข้อความที่ต้องการทำนาย" --version v1.0

# ทำนายจากไฟล์
python predict.py --file data.json --version v1.0
```

### Compare Versions

```powershell
# เปรียบเทียบผลลัพธ์จากหลาย versions
python compare_versions.py
```

## ⚙️ การปรับแต่ง Hyperparameters

แก้ไขไฟล์ `config.py`:

```python
# ตัวอย่างการปรับแต่ง
MAX_FEATURES = 5000      # เพิ่ม features
BATCH_SIZE = 32          # ลด batch size (ถ้า OOM)
EPOCHS = 100             # เพิ่ม epochs
LEARNING_RATE = 0.0001   # ลด learning rate
```

## 📊 ตรวจสอบผลลัพธ์

หลัง training เสร็จ:

```
versions/v1.0/
├── README.md                          # สรุปผลลัพธ์
├── models/                           # Models ที่ save ไว้
│   ├── SVM.pth
│   ├── Neural_Network.pth
│   └── ...
├── plots/                            # กราฟและ visualizations
│   ├── SVM_confusion_matrix.png
│   ├── SVM_roc_curve.png
│   └── model_comparison.png
└── metrics/                          # Metrics ละเอียด (JSON)
```

## 🔧 แก้ปัญหา

### CUDA Not Found

```powershell
# ตรวจสอบ GPU
nvidia-smi
```

### Out of Memory

ลดค่าในไฟล์ `config.py`:

- `BATCH_SIZE = 32` (หรือ 16)
- `MAX_FEATURES = 3000`
- ลด hidden layer dimensions

### Import Error

```powershell
# ใช้ conda สำหรับ RAPIDS
conda install -c rapidsai -c conda-forge -c nvidia cuml cupy
```

## 🎯 เป้าหมาย

- **Target Accuracy**: ≥ 80%
- **Models**: 6 base models + 1 ensemble
- **Hardware**: GPU with CUDA (Required)

## 📚 โครงสร้างไฟล์

```
thai-depression/
├── config.py              # การตั้งค่า
├── train.py              # Training script หลัก
├── predict.py            # Inference script
├── setup_check.py        # ตรวจสอบระบบ
├── compare_versions.py   # เปรียบเทียบ versions
├── models/               # Model implementations
├── utils/                # Utilities
├── data/                 # Dataset
└── versions/             # Training results
```

## ✅ Checklist ก่อนเริ่มต้น

- [ ] Python 3.8-3.10 ติดตั้งแล้ว
- [ ] NVIDIA GPU with CUDA
- [ ] Virtual environment สร้างแล้ว
- [ ] Dependencies ติดตั้งครบ
- [ ] ไฟล์ข้อมูลอยู่ใน data/
- [ ] รัน `python setup_check.py` ผ่าน

## 🚀 พร้อมแล้ว!

```powershell
python train.py
```

Good luck! 🎉
