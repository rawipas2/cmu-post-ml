"""
Setup and validation script
Checks if all dependencies are installed and GPU is available
"""
import sys


def check_python_version():
    """Check Python version"""
    print("🐍 Checking Python version...")
    version = sys.version_info
    if version.major == 3 and version.minor >= 8 and version.minor <= 10:
        print(f"   ✅ Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"   ❌ Python {version.major}.{version.minor} (Need 3.8-3.10)")
        return False


def check_cuda():
    """Check CUDA availability"""
    print("\n🎮 Checking CUDA...")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"   ✅ CUDA Available")
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
            print(f"   CUDA Version: {torch.version.cuda}")
            print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
            return True
        else:
            print(f"   ❌ CUDA Not Available")
            print(f"   This project requires NVIDIA GPU with CUDA!")
            return False
    except ImportError:
        print(f"   ❌ PyTorch not installed")
        return False


def check_cuml():
    """Check cuML installation"""
    print("\n📊 Checking cuML (RAPIDS AI)...")
    try:
        import cuml
        print(f"   ✅ cuML {cuml.__version__}")
        return True
    except ImportError:
        print(f"   ❌ cuML not installed")
        print(f"   Install: conda install -c rapidsai -c conda-forge -c nvidia cuml")
        return False


def check_cupy():
    """Check CuPy installation"""
    print("\n🔢 Checking CuPy...")
    try:
        import cupy as cp
        print(f"   ✅ CuPy {cp.__version__}")
        # Test GPU computation
        a = cp.array([1, 2, 3])
        b = cp.array([4, 5, 6])
        c = a + b
        print(f"   GPU computation test: OK")
        return True
    except ImportError:
        print(f"   ❌ CuPy not installed")
        print(f"   Install: pip install cupy-cuda11x")
        return False


def check_pythainlp():
    """Check PyThaiNLP installation"""
    print("\n🇹🇭 Checking PyThaiNLP...")
    try:
        import pythainlp
        print(f"   ✅ PyThaiNLP {pythainlp.__version__}")
        
        # Download required data
        print(f"   Downloading Thai word tokenizer...")
        from pythainlp.corpus import download
        download('thai2fit_wv')
        print(f"   ✅ Thai NLP data ready")
        return True
    except ImportError:
        print(f"   ❌ PyThaiNLP not installed")
        print(f"   Install: pip install pythainlp")
        return False


def check_other_dependencies():
    """Check other required packages"""
    print("\n📦 Checking other dependencies...")
    
    packages = {
        'numpy': 'NumPy',
        'pandas': 'Pandas',
        'sklearn': 'scikit-learn',
        'matplotlib': 'Matplotlib',
        'seaborn': 'Seaborn'
    }
    
    all_ok = True
    for package, name in packages.items():
        try:
            __import__(package)
            print(f"   ✅ {name}")
        except ImportError:
            print(f"   ❌ {name} not installed")
            all_ok = False
    
    return all_ok


def check_data_files():
    """Check if data files exist"""
    print("\n📁 Checking data files...")
    import os
    
    data_files = ['train.json', 'valid.json', 'test.json']
    all_exist = True
    
    for filename in data_files:
        filepath = os.path.join('data', filename)
        if os.path.exists(filepath):
            size = os.path.getsize(filepath) / (1024 * 1024)  # MB
            print(f"   ✅ {filename} ({size:.2f} MB)")
        else:
            print(f"   ❌ {filename} not found")
            all_exist = False
    
    return all_exist


def create_directories():
    """Create required directories"""
    print("\n📂 Creating directories...")
    import os
    
    directories = [
        'saved_models',
        'results',
        'versions'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"   ✅ {directory}/")
    
    return True


def main():
    """Run all checks"""
    print("="*80)
    print("🚀 Thai Depression Classification - Setup & Validation")
    print("="*80)
    
    checks = [
        ("Python Version", check_python_version),
        ("CUDA & GPU", check_cuda),
        ("cuML (RAPIDS AI)", check_cuml),
        ("CuPy", check_cupy),
        ("PyThaiNLP", check_pythainlp),
        ("Other Dependencies", check_other_dependencies),
        ("Data Files", check_data_files),
        ("Directories", create_directories)
    ]
    
    results = []
    for name, check_func in checks:
        result = check_func()
        results.append((name, result))
    
    # Summary
    print("\n" + "="*80)
    print("📋 SETUP SUMMARY")
    print("="*80)
    
    all_passed = True
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status:12} - {name}")
        if not result:
            all_passed = False
    
    print("="*80)
    
    if all_passed:
        print("\n✅ All checks passed! You're ready to train models.")
        print("\n🚀 Run: python train.py")
    else:
        print("\n❌ Some checks failed. Please install missing dependencies.")
        print("\n📖 See README.md for installation instructions.")
        sys.exit(1)


if __name__ == "__main__":
    main()
