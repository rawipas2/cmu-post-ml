# 🐍 คู่มือการใช้งาน Python 3.10 สำหรับโปรเจกต์ Thai Depression Classification

## ⚠️ ข้อกำหนดสำคัญ

โปรเจกต์นี้ต้องการ **Python 3.10** เท่านั้น

## 📌 วิธีตรวจสอบ Python Version

### Windows

```powershell
# ตรวจสอบเวอร์ชัน
python --version

# ควรแสดงผล: Python 3.10.x
```

### ใช้สคริปต์ตรวจสอบอัตโนมัติ

```powershell
# วิธีที่ 1: รันสคริปต์โดยตรง
python check_python_version.py

# วิธีที่ 2: ดับเบิลคลิกไฟล์
check_python.bat
```

สคริปต์จะตรวจสอบและแสดง:

- ✅ Python 3.10 พร้อมใช้งาน
- ⚠️ พบ Python 3.10 แต่ไม่ได้เป็น default
- ❌ ไม่พบ Python 3.10 + คำแนะนำการติดตั้ง

---

## 📥 วิธีติดตั้ง Python 3.10

### วิธีที่ 1: ดาวน์โหลดจาก Python.org (แนะนำ)

1. เข้าไปที่: https://www.python.org/downloads/release/python-31011/
2. เลื่อนลงไปที่ส่วน "Files"
3. ดาวน์โหลด **Windows installer (64-bit)**
4. รันตัวติดตั้ง และ **สำคัญ!**
   - ✅ เช็ค "Add Python 3.10 to PATH"
   - ✅ เลือก "Install Now"
5. รีสตาร์ท Terminal/PowerShell
6. ตรวจสอบด้วย: `python --version`

### วิธีที่ 2: ใช้ Windows Package Manager

```powershell
# ติดตั้งด้วย winget
winget install Python.Python.3.10
```

### วิธีที่ 3: ใช้ Chocolatey

```powershell
# ติดตั้งด้วย Chocolatey
choco install python310
```

### วิธีที่ 4: ใช้ Conda/Miniconda (แนะนำสำหรับโปรเจกต์นี้)

```bash
# 1. ติดตั้ง Miniconda
# ดาวน์โหลด: https://docs.conda.io/en/latest/miniconda.html

# 2. สร้าง environment พร้อม Python 3.10
conda create -n thai-depression python=3.10 -y

# 3. เปิดใช้งาน environment
conda activate thai-depression

# 4. ตรวจสอบ
python --version
# ควรแสดง: Python 3.10.x
```

---

## 🚀 การเริ่มต้นใช้งาน

### ถ้าติดตั้ง Python 3.10 แบบ Global

```powershell
# 1. ตรวจสอบเวอร์ชัน
python --version

# 2. รัน setup อัตโนมัติ
.\setup.ps1

# 3. เริ่มเทรนโมเดล
python train.py
```

### ถ้าใช้ Python Launcher (py.exe)

```powershell
# ถ้ามี Python หลายเวอร์ชัน ใช้ launcher
py -3.10 --version

# รัน setup
py -3.10 setup.ps1

# เทรนโมเดล
py -3.10 train.py
```

### ถ้าใช้ Conda Environment

```bash
# 1. เปิดใช้งาน environment
conda activate thai-depression

# 2. ตรวจสอบ
python --version

# 3. รัน setup
.\setup.ps1

# 4. เทรนโมเดล
python train.py
```

### ถ้าใช้ Virtual Environment (venv)

```powershell
# 1. สร้าง venv ด้วย Python 3.10
python -m venv venv

# 2. เปิดใช้งาน
.\venv\Scripts\Activate.ps1

# 3. อัพเกรด pip
python -m pip install --upgrade pip

# 4. ติดตั้ง dependencies
.\setup.ps1

# 5. เทรนโมเดล
python train.py
```

---

## ❓ การแก้ปัญหา

### ปัญหา: python --version แสดงเวอร์ชันอื่น

**วิธีแก้:**

1. ใช้ Python Launcher แทน

   ```powershell
   py -3.10 --version
   py -3.10 train.py
   ```

2. หรือสร้าง conda environment
   ```bash
   conda create -n thai-depression python=3.10
   conda activate thai-depression
   ```

### ปัญหา: ติดตั้ง Python 3.10 แล้วแต่ไม่อยู่ใน PATH

**วิธีแก้:**

1. เพิ่ม Python 3.10 เข้า System PATH

   - ค้นหาตำแหน่งติดตั้ง (เช่น `C:\Python310` หรือ `C:\Users\<username>\AppData\Local\Programs\Python\Python310`)
   - เพิ่มเข้า System Environment Variables > Path

2. หรือใช้ conda/venv environment แทน

### ปัญหา: มี Python 3.11/3.12/3.13 ไม่ใช่ 3.10

**วิธีแก้:**

โปรเจกต์นี้ต้องการ Python 3.10 เฉพาะเจาะจง เพราะ:

- รองรับ RAPIDS AI (cuML, cuPy) ได้ดีที่สุด
- รองรับ PyTorch + CUDA ได้ดี
- เสถียรสำหรับ dependencies ทั้งหมด

**แนะนำ:** ใช้ conda environment เพื่อจัดการหลาย Python version

```bash
# สร้าง environment ใหม่ด้วย Python 3.10
conda create -n thai-depression python=3.10
conda activate thai-depression
```

---

## 📝 สรุป Flow การติดตั้ง

```
1. ตรวจสอบ Python Version
   ↓
   python check_python_version.py

2. ถ้าไม่มี Python 3.10
   ↓
   ติดตั้งจาก python.org หรือใช้ conda

3. รัน Setup
   ↓
   .\setup.ps1

4. ตรวจสอบการติดตั้ง
   ↓
   python setup_check.py

5. เริ่มเทรนโมเดล
   ↓
   python train.py
```

---

## 🔗 ลิงก์ที่เป็นประโยชน์

- Python 3.10 Download: https://www.python.org/downloads/release/python-31011/
- Miniconda: https://docs.conda.io/en/latest/miniconda.html
- CUDA Toolkit: https://developer.nvidia.com/cuda-downloads
- RAPIDS AI: https://rapids.ai/start.html

---

## 💡 เคล็ดลับ

1. **แนะนำให้ใช้ Conda** สำหรับโปรเจกต์นี้ เพราะติดตั้ง GPU libraries ได้ง่ายที่สุด
2. **ตรวจสอบ Python version เสมอ** ก่อนรัน setup หรือ train
3. **ใช้ virtual environment** เพื่อไม่ให้กระทบกับ Python projects อื่น
4. **เก็บ log** ของการติดตั้งไว้ถ้ามีปัญหา

---

ต้องการความช่วยเหลือ? เปิด issue ใน GitHub repository!
