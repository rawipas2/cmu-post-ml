# 🚀 START HERE - เริ่มต้นใช้งาน

## ⚠️ ก่อนเริ่มต้น - ตรวจสอบ Python 3.10

โปรเจกต์นี้ต้องการ **Python 3.10** เท่านั้น

### ตรวจสอบเวอร์ชัน Python

```powershell
python --version
# ควรแสดง: Python 3.10.x
```

### ใช้สคริปต์ตรวจสอบ

**Windows - ดับเบิลคลิก:** `check_python.bat`

**หรือรัน:**

```powershell
python check_python_version.py
```

### ถ้ายังไม่มี Python 3.10

อ่านคู่มือการติดตั้งที่ `PYTHON_310_GUIDE.md`

---

## ติดตั้งอัตโนมัติ (แนะนำ)

### Windows

**วิธีที่ 1:** Double-click ไฟล์ `setup.bat`

**วิธีที่ 2:** เปิด PowerShell และรัน:

```powershell
.\setup.ps1
```

### Linux/Mac

```bash
chmod +x setup.sh
./setup.sh
```

---

## ตรวจสอบการติดตั้ง

### Windows

Double-click `run_check.bat` หรือ:

```powershell
python setup_check.py
```

### Linux/Mac

```bash
python setup_check.py
```

---

## เริ่ม Training

### Windows

Double-click `run_train.bat` หรือ:

```powershell
# ถ้าใช้ conda
conda activate thai-depression
python train.py

# ถ้าใช้ venv
.\venv\Scripts\Activate.ps1
python train.py
```

### Linux/Mac

```bash
# ถ้าใช้ conda
conda activate thai-depression
python train.py

# ถ้าใช้ venv
source venv/bin/activate
python train.py
```

---

## 📖 เอกสารเพิ่มเติม

- **INSTALL.md** - คู่มือติดตั้งแบบละเอียด
- **QUICKSTART.md** - คำสั่งที่ใช้บ่อย
- **README.md** - คู่มือหลักฉบับเต็ม

---

## ✅ Checklist

- [ ] ติดตั้ง Python 3.8-3.10
- [ ] มี NVIDIA GPU + CUDA
- [ ] รัน `setup.bat` หรือ `setup.ps1` (Windows) หรือ `setup.sh` (Linux/Mac)
- [ ] รัน `run_check.bat` หรือ `python setup_check.py` ผ่านทุก check
- [ ] พร้อม train! รัน `run_train.bat` หรือ `python train.py`

---

## 🎯 เป้าหมาย

- **Accuracy**: ≥ 80%
- **Models**: 6 base models + ensemble stacking
- **Hardware**: GPU (CUDA) required

---

## 🔧 หากมีปัญหา

1. ดู **INSTALL.md** สำหรับวิธีแก้ปัญหา
2. ตรวจสอบว่า GPU driver ติดตั้งแล้ว: `nvidia-smi`
3. แนะนำใช้ **conda** สำหรับติดตั้ง RAPIDS AI (cuML, cuPy)

---

**พร้อมใช้งาน! เพียงแค่รัน setup และ train** 🚀
