# Thai Depression Classification - Dependencies Setup
# สคริปต์ติดตั้ง dependencies อัตโนมัติ

Write-Host "=" -NoNewline -ForegroundColor Cyan
Write-Host ("="*79) -ForegroundColor Cyan
Write-Host "Thai Depression Classification - Auto Setup" -ForegroundColor Green
Write-Host ("="*80) -ForegroundColor Cyan
Write-Host ""

# ตรวจสอบ Python version - ต้องการ Python 3.10
Write-Host "Checking Python version..." -ForegroundColor Yellow
$pythonVersion = python --version 2>&1
Write-Host "   $pythonVersion" -ForegroundColor White

if ($pythonVersion -match "Python 3\.10\.") {
    Write-Host "   Python 3.10 detected - OK" -ForegroundColor Green
} else {
    Write-Host "   ERROR: This project requires Python 3.10" -ForegroundColor Red
    Write-Host "   Current version: $pythonVersion" -ForegroundColor Yellow
    Write-Host "" 
    Write-Host "   Running Python version checker..." -ForegroundColor Cyan
    python check_python_version.py
    Write-Host ""
    Write-Host "   Please install Python 3.10 and try again." -ForegroundColor Red
    exit 1
}

# ตรวจสอบ NVIDIA GPU
Write-Host "`nChecking NVIDIA GPU..." -ForegroundColor Yellow
try {
    $nvidiaSmi = nvidia-smi 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "   NVIDIA GPU detected" -ForegroundColor Green
    } else {
        Write-Host "   NVIDIA GPU not found! This project requires GPU." -ForegroundColor Red
        exit 1
    }
} catch {
    Write-Host "   nvidia-smi not found! Install NVIDIA drivers." -ForegroundColor Red
    exit 1
}

# ตรวจสอบ conda
Write-Host "`nChecking conda..." -ForegroundColor Yellow
$condaExists = Get-Command conda -ErrorAction SilentlyContinue
if ($condaExists) {
    Write-Host "   Conda available" -ForegroundColor Green
    $useConda = $true
} else {
    Write-Host "   Conda not found - will use pip only" -ForegroundColor Yellow
    Write-Host "   Conda is recommended for RAPIDS AI (cuML, cuPy)" -ForegroundColor Cyan
    $useConda = $false
}

Write-Host "`n" -NoNewline
Write-Host ("="*80) -ForegroundColor Cyan
Write-Host "Installing Dependencies" -ForegroundColor Green
Write-Host ("="*80) -ForegroundColor Cyan

if ($useConda) {
    # ใช้ Conda environment
    Write-Host "`nCreating conda environment: thai-depression" -ForegroundColor Yellow
    
    $envExists = conda env list | Select-String "thai-depression"
    if ($envExists) {
        Write-Host "   Environment already exists. Updating..." -ForegroundColor Cyan
    } else {
        conda create -n thai-depression python=3.10 -y
    }
    
    Write-Host "`nInstalling RAPIDS AI (cuML, cuPy)..." -ForegroundColor Yellow
    conda install -n thai-depression -c rapidsai -c conda-forge -c nvidia cuml cupy -y
    
    Write-Host "`nInstalling PyTorch with CUDA..." -ForegroundColor Yellow
    conda install -n thai-depression pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia -y
    
    Write-Host "`nInstalling other dependencies..." -ForegroundColor Yellow
    conda run -n thai-depression pip install pythainlp scikit-learn matplotlib seaborn pandas tqdm
    
    Write-Host "`nSetup complete!" -ForegroundColor Green
    Write-Host "`nActivate environment with:" -ForegroundColor Cyan
    Write-Host "   conda activate thai-depression" -ForegroundColor White
    
} else {
    # ใช้ pip และ venv
    Write-Host "`nCreating virtual environment..." -ForegroundColor Yellow
    
    if (Test-Path "venv") {
        Write-Host "   Virtual environment exists. Using existing..." -ForegroundColor Cyan
    } else {
        python -m venv venv
        Write-Host "   Virtual environment created" -ForegroundColor Green
    }
    
    Write-Host "`nActivating virtual environment..." -ForegroundColor Yellow
    & ".\venv\Scripts\Activate.ps1"
    
    Write-Host "`nUpgrading pip..." -ForegroundColor Yellow
    python -m pip install --upgrade pip
    
    Write-Host "`nInstalling PyTorch with CUDA..." -ForegroundColor Yellow
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
    
    Write-Host "`nInstalling core dependencies..." -ForegroundColor Yellow
    pip install numpy pandas scikit-learn matplotlib seaborn tqdm
    
    Write-Host "`nInstalling PyThaiNLP..." -ForegroundColor Yellow
    pip install pythainlp attacut
    
    Write-Host "`nRAPIDS AI (cuML, cuPy) installation:" -ForegroundColor Yellow
    Write-Host "   For best GPU performance, install via conda:" -ForegroundColor Cyan
    Write-Host "   1. Install Miniconda: https://docs.conda.io/en/latest/miniconda.html" -ForegroundColor White
    Write-Host "   2. Run this script again" -ForegroundColor White
    Write-Host ""
    Write-Host "   Or install cuPy with pip (may have issues):" -ForegroundColor Cyan
    Write-Host "   pip install cupy-cuda11x" -ForegroundColor White
    Write-Host ""
    
    $installCuPy = Read-Host "Try installing cuPy via pip now? (y/n)"
    if ($installCuPy -eq "y") {
        Write-Host "`nInstalling cuPy..." -ForegroundColor Yellow
        pip install cupy-cuda11x
    }
    
    Write-Host "`nBasic setup complete!" -ForegroundColor Green
    Write-Host "`nVirtual environment is activated" -ForegroundColor Cyan
}

# ดาวน์โหลด Thai NLP data
Write-Host "`nDownloading Thai NLP data..." -ForegroundColor Yellow
if ($useConda) {
    conda run -n thai-depression python -c "from pythainlp.corpus import download; download('thai2fit_wv')"
} else {
    python -c "from pythainlp.corpus import download; download('thai2fit_wv')"
}

# รัน setup check
Write-Host "`nRunning setup validation..." -ForegroundColor Yellow
if ($useConda) {
    conda run -n thai-depression python setup_check.py
} else {
    python setup_check.py
}

Write-Host "`n" -NoNewline
Write-Host ("="*80) -ForegroundColor Cyan
Write-Host "Setup Complete!" -ForegroundColor Green
Write-Host ("="*80) -ForegroundColor Cyan

if ($useConda) {
    Write-Host "`nNext steps:" -ForegroundColor Cyan
    Write-Host "   1. conda activate thai-depression" -ForegroundColor White
    Write-Host "   2. python train.py" -ForegroundColor White
} else {
    Write-Host "`nNext steps:" -ForegroundColor Cyan
    Write-Host "   1. Make sure virtual environment is activated" -ForegroundColor White
    Write-Host "   2. python train.py" -ForegroundColor White
}

Write-Host ""
