# 📌 Quick Reference - Python 3.10 Setup

## 🎯 สำหรับผู้ใช้ที่รีบ

### 1. ตรวจสอบ Python (ดับเบิลคลิก)

```
check_python.bat
```

### 2. ถ้ามี Python 3.10 แล้ว → รัน Setup

```
setup.bat
หรือ
.\setup.ps1
```

### 3. ถ้ายังไม่มี Python 3.10

**ดาวน์โหลด:**
https://www.python.org/downloads/release/python-31011/

**หรือใช้ Conda:**

```bash
conda create -n thai-depression python=3.10
conda activate thai-depression
.\setup.ps1
```

---

## 📋 คำสั่งที่ใช้บ่อย

### ตรวจสอบเวอร์ชัน

```powershell
python --version
python check_python_version.py
```

### ติดตั้ง Environment

```powershell
# แบบ Conda (แนะนำ)
conda create -n thai-depression python=3.10 -y
conda activate thai-depression

# แบบ venv
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### รัน Setup

```powershell
.\setup.ps1
```

### ตรวจสอบ Setup

```powershell
python setup_check.py
```

### เทรนโมเดล

```powershell
python train.py
```

---

## 🆘 แก้ปัญหาเร็ว

### ปัญหา: Python version ไม่ถูกต้อง

**วิธีแก้:** ใช้ Conda environment

```bash
conda create -n thai-depression python=3.10
conda activate thai-depression
```

### ปัญหา: setup.ps1 ไม่รัน

**วิธีแก้:** เปลี่ยน Execution Policy

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### ปัญหา: ไม่มี GPU

**คำตอบ:** โปรเจกต์นี้ต้องการ NVIDIA GPU

- ติดตั้ง CUDA: https://developer.nvidia.com/cuda-downloads

---

## 📚 เอกสารเพิ่มเติม

- **คู่มือละเอียด:** `PYTHON_310_GUIDE.md`
- **วิธีติดตั้ง:** `INSTALL.md`
- **เริ่มต้นใช้งาน:** `START_HERE.md`
- **สรุปการเปลี่ยนแปลง:** `PYTHON_VERSION_UPDATE_SUMMARY.md`

---

## ✅ Checklist

- [ ] ตรวจสอบ Python 3.10 (`check_python.bat`)
- [ ] ติดตั้ง Python 3.10 (ถ้ายังไม่มี)
- [ ] รัน Setup (`setup.ps1`)
- [ ] ตรวจสอบ Setup (`setup_check.py`)
- [ ] เทรนโมเดล (`train.py`)

---

พิมพ์ไฟล์นี้และติดไว้ข้างจอ! 🖨️
