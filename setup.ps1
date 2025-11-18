# Thai Depression Classification - Dependencies Setup
# สคริปต์ติดตั้ง dependencies อัตโนมัติ

Write-Host "=" -NoNewline -ForegroundColor Cyan
Write-Host ("="*79) -ForegroundColor Cyan
Write-Host "🚀 Thai Depression Classification - Auto Setup" -ForegroundColor Green
Write-Host ("="*80) -ForegroundColor Cyan
Write-Host ""

# ตรวจสอบ Python version
Write-Host "🐍 Checking Python version..." -ForegroundColor Yellow
$pythonVersion = python --version 2>&1
Write-Host "   $pythonVersion" -ForegroundColor White

if ($pythonVersion -match "Python 3\.(8|9|10)\.") {
    Write-Host "   ✅ Python version OK" -ForegroundColor Green
} else {
    Write-Host "   ❌ Need Python 3.8-3.10" -ForegroundColor Red
    exit 1
}

# ตรวจสอบ NVIDIA GPU
Write-Host "`n🎮 Checking NVIDIA GPU..." -ForegroundColor Yellow
try {
    $nvidiaSmi = nvidia-smi 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "   ✅ NVIDIA GPU detected" -ForegroundColor Green
    } else {
        Write-Host "   ❌ NVIDIA GPU not found! This project requires GPU." -ForegroundColor Red
        exit 1
    }
} catch {
    Write-Host "   ❌ nvidia-smi not found! Install NVIDIA drivers." -ForegroundColor Red
    exit 1
}

# ตรวจสอบ conda
Write-Host "`n📦 Checking conda..." -ForegroundColor Yellow
$condaExists = Get-Command conda -ErrorAction SilentlyContinue
if ($condaExists) {
    Write-Host "   ✅ Conda available" -ForegroundColor Green
    $useConda = $true
} else {
    Write-Host "   ⚠️  Conda not found - will use pip only" -ForegroundColor Yellow
    Write-Host "   💡 Conda is recommended for RAPIDS AI (cuML, cuPy)" -ForegroundColor Cyan
    $useConda = $false
}

Write-Host "`n" -NoNewline
Write-Host ("="*80) -ForegroundColor Cyan
Write-Host "📥 Installing Dependencies" -ForegroundColor Green
Write-Host ("="*80) -ForegroundColor Cyan

if ($useConda) {
    # ใช้ Conda environment
    Write-Host "`n🔧 Creating conda environment: thai-depression" -ForegroundColor Yellow
    
    $envExists = conda env list | Select-String "thai-depression"
    if ($envExists) {
        Write-Host "   Environment already exists. Updating..." -ForegroundColor Cyan
    } else {
        conda create -n thai-depression python=3.9 -y
    }
    
    Write-Host "`n📦 Installing RAPIDS AI (cuML, cuPy)..." -ForegroundColor Yellow
    conda install -n thai-depression -c rapidsai -c conda-forge -c nvidia cuml cupy -y
    
    Write-Host "`n📦 Installing PyTorch with CUDA..." -ForegroundColor Yellow
    conda install -n thai-depression pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia -y
    
    Write-Host "`n📦 Installing other dependencies..." -ForegroundColor Yellow
    conda run -n thai-depression pip install pythainlp scikit-learn matplotlib seaborn pandas tqdm
    
    Write-Host "`n✅ Setup complete!" -ForegroundColor Green
    Write-Host "`n💡 Activate environment with:" -ForegroundColor Cyan
    Write-Host "   conda activate thai-depression" -ForegroundColor White
    
} else {
    # ใช้ pip และ venv
    Write-Host "`n🔧 Creating virtual environment..." -ForegroundColor Yellow
    
    if (Test-Path "venv") {
        Write-Host "   Virtual environment exists. Using existing..." -ForegroundColor Cyan
    } else {
        python -m venv venv
        Write-Host "   ✅ Virtual environment created" -ForegroundColor Green
    }
    
    Write-Host "`n🔧 Activating virtual environment..." -ForegroundColor Yellow
    & ".\venv\Scripts\Activate.ps1"
    
    Write-Host "`n📦 Upgrading pip..." -ForegroundColor Yellow
    python -m pip install --upgrade pip
    
    Write-Host "`n📦 Installing PyTorch with CUDA..." -ForegroundColor Yellow
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
    
    Write-Host "`n📦 Installing core dependencies..." -ForegroundColor Yellow
    pip install numpy pandas scikit-learn matplotlib seaborn tqdm
    
    Write-Host "`n📦 Installing PyThaiNLP..." -ForegroundColor Yellow
    pip install pythainlp attacut
    
    Write-Host "`n⚠️  RAPIDS AI (cuML, cuPy) installation:" -ForegroundColor Yellow
    Write-Host "   For best GPU performance, install via conda:" -ForegroundColor Cyan
    Write-Host "   1. Install Miniconda: https://docs.conda.io/en/latest/miniconda.html" -ForegroundColor White
    Write-Host "   2. Run this script again" -ForegroundColor White
    Write-Host ""
    Write-Host "   Or install cuPy with pip (may have issues):" -ForegroundColor Cyan
    Write-Host "   pip install cupy-cuda11x" -ForegroundColor White
    Write-Host ""
    
    $installCuPy = Read-Host "Try installing cuPy via pip now? (y/n)"
    if ($installCuPy -eq "y") {
        Write-Host "`n📦 Installing cuPy..." -ForegroundColor Yellow
        pip install cupy-cuda11x
    }
    
    Write-Host "`n✅ Basic setup complete!" -ForegroundColor Green
    Write-Host "`n💡 Virtual environment is activated" -ForegroundColor Cyan
}

# ดาวน์โหลด Thai NLP data
Write-Host "`n📥 Downloading Thai NLP data..." -ForegroundColor Yellow
if ($useConda) {
    conda run -n thai-depression python -c "from pythainlp.corpus import download; download('thai2fit_wv')"
} else {
    python -c "from pythainlp.corpus import download; download('thai2fit_wv')"
}

# รัน setup check
Write-Host "`n🔍 Running setup validation..." -ForegroundColor Yellow
if ($useConda) {
    conda run -n thai-depression python setup_check.py
} else {
    python setup_check.py
}

Write-Host "`n" -NoNewline
Write-Host ("="*80) -ForegroundColor Cyan
Write-Host "🎉 Setup Complete!" -ForegroundColor Green
Write-Host ("="*80) -ForegroundColor Cyan

if ($useConda) {
    Write-Host "`n📝 Next steps:" -ForegroundColor Cyan
    Write-Host "   1. conda activate thai-depression" -ForegroundColor White
    Write-Host "   2. python train.py" -ForegroundColor White
} else {
    Write-Host "`n📝 Next steps:" -ForegroundColor Cyan
    Write-Host "   1. Make sure virtual environment is activated" -ForegroundColor White
    Write-Host "   2. python train.py" -ForegroundColor White
}

Write-Host ""
