#!/bin/bash
# Thai Depression Classification - Dependencies Setup for Linux/Mac
# สคริปต์ติดตั้ง dependencies อัตโนมัติ

echo "================================================================================"
echo "🚀 Thai Depression Classification - Auto Setup"
echo "================================================================================"
echo ""

# ตรวจสอบ Python version
echo "🐍 Checking Python version..."
PYTHON_VERSION=$(python3 --version 2>&1)
echo "   $PYTHON_VERSION"

if python3 -c "import sys; sys.exit(0 if (3,8) <= sys.version_info[:2] <= (3,10) else 1)" 2>/dev/null; then
    echo "   ✅ Python version OK"
else
    echo "   ❌ Need Python 3.8-3.10"
    exit 1
fi

# ตรวจสอบ NVIDIA GPU
echo ""
echo "🎮 Checking NVIDIA GPU..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi &> /dev/null
    if [ $? -eq 0 ]; then
        echo "   ✅ NVIDIA GPU detected"
    else
        echo "   ❌ NVIDIA GPU not found! This project requires GPU."
        exit 1
    fi
else
    echo "   ❌ nvidia-smi not found! Install NVIDIA drivers."
    exit 1
fi

# ตรวจสอบ conda
echo ""
echo "📦 Checking conda..."
if command -v conda &> /dev/null; then
    echo "   ✅ Conda available"
    USE_CONDA=true
else
    echo "   ⚠️  Conda not found - will use pip only"
    echo "   💡 Conda is recommended for RAPIDS AI (cuML, cuPy)"
    USE_CONDA=false
fi

echo ""
echo "================================================================================"
echo "📥 Installing Dependencies"
echo "================================================================================"

if [ "$USE_CONDA" = true ]; then
    # ใช้ Conda environment
    echo ""
    echo "🔧 Creating conda environment: thai-depression"
    
    if conda env list | grep -q "thai-depression"; then
        echo "   Environment already exists. Updating..."
    else
        conda create -n thai-depression python=3.9 -y
    fi
    
    echo ""
    echo "📦 Installing RAPIDS AI (cuML, cuPy)..."
    conda install -n thai-depression -c rapidsai -c conda-forge -c nvidia cuml cupy -y
    
    echo ""
    echo "📦 Installing PyTorch with CUDA..."
    conda install -n thai-depression pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia -y
    
    echo ""
    echo "📦 Installing other dependencies..."
    conda run -n thai-depression pip install pythainlp scikit-learn matplotlib seaborn pandas tqdm
    
    echo ""
    echo "✅ Setup complete!"
    echo ""
    echo "💡 Activate environment with:"
    echo "   conda activate thai-depression"
    
else
    # ใช้ pip และ venv
    echo ""
    echo "🔧 Creating virtual environment..."
    
    if [ -d "venv" ]; then
        echo "   Virtual environment exists. Using existing..."
    else
        python3 -m venv venv
        echo "   ✅ Virtual environment created"
    fi
    
    echo ""
    echo "🔧 Activating virtual environment..."
    source venv/bin/activate
    
    echo ""
    echo "📦 Upgrading pip..."
    pip install --upgrade pip
    
    echo ""
    echo "📦 Installing PyTorch with CUDA..."
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
    
    echo ""
    echo "📦 Installing core dependencies..."
    pip install numpy pandas scikit-learn matplotlib seaborn tqdm
    
    echo ""
    echo "📦 Installing PyThaiNLP..."
    pip install pythainlp attacut
    
    echo ""
    echo "⚠️  RAPIDS AI (cuML, cuPy) installation:"
    echo "   For best GPU performance, install via conda:"
    echo "   1. Install Miniconda: https://docs.conda.io/en/latest/miniconda.html"
    echo "   2. Run this script again"
    echo ""
    echo "   Or install cuPy with pip:"
    echo "   pip install cupy-cuda11x"
    echo ""
    
    read -p "Try installing cuPy via pip now? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo ""
        echo "📦 Installing cuPy..."
        pip install cupy-cuda11x
    fi
    
    echo ""
    echo "✅ Basic setup complete!"
    echo ""
    echo "💡 Virtual environment is activated"
fi

# ดาวน์โหลด Thai NLP data
echo ""
echo "📥 Downloading Thai NLP data..."
if [ "$USE_CONDA" = true ]; then
    conda run -n thai-depression python -c "from pythainlp.corpus import download; download('thai2fit_wv')"
else
    python -c "from pythainlp.corpus import download; download('thai2fit_wv')"
fi

# รัน setup check
echo ""
echo "🔍 Running setup validation..."
if [ "$USE_CONDA" = true ]; then
    conda run -n thai-depression python setup_check.py
else
    python setup_check.py
fi

echo ""
echo "================================================================================"
echo "🎉 Setup Complete!"
echo "================================================================================"

if [ "$USE_CONDA" = true ]; then
    echo ""
    echo "📝 Next steps:"
    echo "   1. conda activate thai-depression"
    echo "   2. python train.py"
else
    echo ""
    echo "📝 Next steps:"
    echo "   1. source venv/bin/activate"
    echo "   2. python train.py"
fi

echo ""
