# 🚀 Quick Setup Guide

## วิธีที่ 1: ติดตั้งอัตโนมัติ (แนะนำ)

### Windows

```powershell
# รัน PowerShell setup script
.\setup.ps1
```

### Linux/Mac

```bash
# รัน bash setup script
chmod +x setup.sh
./setup.sh
```

---

## วิธีที่ 2: ติดตั้งด้วยตนเอง

### แบบ Conda (แนะนำที่สุด - รองรับ cuML, cuPy ได้ดีที่สุด)

```bash
# 1. สร้าง environment
conda create -n thai-depression python=3.9
conda activate thai-depression

# 2. ติดตั้ง RAPIDS AI (cuML, cuPy)
conda install -c rapidsai -c conda-forge -c nvidia cuml cupy

# 3. ติดตั้ง PyTorch with CUDA
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia

# 4. ติดตั้ง dependencies อื่นๆ
pip install pythainlp scikit-learn matplotlib seaborn pandas tqdm attacut

# 5. ดาวน์โหลด Thai NLP data
python -c "from pythainlp.corpus import download; download('thai2fit_wv')"

# 6. ตรวจสอบ
python setup_check.py

# 7. เริ่มใช้งาน
python train.py
```

### แบบ pip + venv (ไม่แนะนำสำหรับ cuML)

```powershell
# Windows PowerShell

# 1. สร้าง virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# 2. อัพเกรด pip
python -m pip install --upgrade pip

# 3. ติดตั้ง PyTorch with CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 4. ติดตั้ง dependencies
pip install numpy pandas scikit-learn matplotlib seaborn tqdm
pip install pythainlp attacut

# 5. ติดตั้ง cuPy (อาจมีปัญหา)
pip install cupy-cuda11x

# 6. ดาวน์โหลด Thai NLP data
python -c "from pythainlp.corpus import download; download('thai2fit_wv')"

# 7. ตรวจสอบ
python setup_check.py

# 8. เริ่มใช้งาน
python train.py
```

```bash
# Linux/Mac

# 1. สร้าง virtual environment
python3 -m venv venv
source venv/bin/activate

# 2. อัพเกรด pip
pip install --upgrade pip

# 3. ติดตั้ง PyTorch with CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 4. ติดตั้ง dependencies
pip install numpy pandas scikit-learn matplotlib seaborn tqdm
pip install pythainlp attacut

# 5. ติดตั้ง cuPy
pip install cupy-cuda11x

# 6. ดาวน์โหลด Thai NLP data
python -c "from pythainlp.corpus import download; download('thai2fit_wv')"

# 7. ตรวจสอบ
python setup_check.py

# 8. เริ่มใช้งาน
python train.py
```

---

## ✅ Checklist

- [ ] Python 3.8-3.10 installed
- [ ] NVIDIA GPU with CUDA 11.x+
- [ ] NVIDIA Driver installed (check with `nvidia-smi`)
- [ ] Conda installed (แนะนำ) หรือ Python venv
- [ ] Environment created and activated
- [ ] PyTorch with CUDA installed
- [ ] RAPIDS AI (cuML, cuPy) installed หรือ cuPy
- [ ] Other dependencies installed
- [ ] Thai NLP data downloaded
- [ ] `python setup_check.py` ผ่านทุก check

---

## 🔧 การแก้ปัญหา

### ❌ CUDA Not Found

```bash
# ตรวจสอบ GPU
nvidia-smi

# ถ้าไม่มี ให้ติดตั้ง NVIDIA Driver
# https://www.nvidia.com/Download/index.aspx
```

### ❌ cuML/cuPy Installation Failed

```bash
# ใช้ conda แทน pip
conda install -c rapidsai -c conda-forge -c nvidia cuml cupy
```

### ❌ PyTorch ไม่เจอ CUDA

```bash
# ติดตั้ง PyTorch version ที่ตรงกับ CUDA
# ตรวจสอบ CUDA version: nvidia-smi

# CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# หรือใช้ conda
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia
```

### ❌ Out of Memory

ลดค่าใน `config.py`:

```python
BATCH_SIZE = 32  # ลดจาก 64
MAX_FEATURES = 3000  # ลดจาก 5000
```

---

## 📝 หมายเหตุ

1. **แนะนำให้ใช้ Conda** - การติดตั้ง RAPIDS AI (cuML, cuPy) จะง่ายและเสถียรกว่า
2. **GPU จำเป็น** - โปรเจกต์นี้ต้องใช้ NVIDIA GPU with CUDA
3. **CUDA Version** - ตรวจสอบให้ตรงกับ PyTorch และ RAPIDS AI
4. **Disk Space** - ต้องใช้พื้นที่ประมาณ 5-10 GB สำหรับ dependencies

---

## 🚀 เริ่มใช้งาน

หลังจากติดตั้งเสร็จ:

```bash
# ตรวจสอบระบบ
python setup_check.py

# ทดสอบ models (optional)
python test_models.py

# เริ่ม training
python train.py
```

---

**Need Help?** ดู `README.md` สำหรับข้อมูลเพิ่มเติม
