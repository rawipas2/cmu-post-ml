"""
Python 3.10 Version Checker for Windows
Checks if Python 3.10 is installed and provides installation guidance
"""
import sys
import subprocess
import platform


def check_python_310():
    """Check if Python 3.10 is installed"""
    print("="*80)
    print("🐍 Python 3.10 Version Checker")
    print("="*80)
    print()
    
    # Get current Python version
    current_version = sys.version_info
    current_version_str = f"{current_version.major}.{current_version.minor}.{current_version.micro}"
    
    print(f"📌 Current Python Version: {current_version_str}")
    print(f"   Path: {sys.executable}")
    print()
    
    # Check if it's Python 3.10
    if current_version.major == 3 and current_version.minor == 10:
        print("✅ Python 3.10 is installed and active!")
        print()
        return True
    else:
        print("❌ Python 3.10 is NOT active!")
        print()
        return False


def check_other_python_installations():
    """Check for other Python installations on Windows"""
    print("🔍 Searching for Python 3.10 installations on Windows...")
    print()
    
    # Check common installation paths
    import os
    from pathlib import Path
    
    possible_paths = [
        # Python Launcher paths
        Path(os.environ.get('LOCALAPPDATA', '')) / 'Programs' / 'Python' / 'Python310',
        Path(os.environ.get('LOCALAPPDATA', '')) / 'Programs' / 'Python' / 'Python310-64',
        # System-wide installations
        Path('C:/Python310'),
        Path('C:/Program Files/Python310'),
        Path('C:/Program Files (x86)/Python310'),
    ]
    
    # Add user profile paths
    user_profile = Path(os.environ.get('USERPROFILE', ''))
    possible_paths.extend([
        user_profile / 'AppData' / 'Local' / 'Programs' / 'Python' / 'Python310',
        user_profile / 'AppData' / 'Local' / 'Programs' / 'Python' / 'Python310-64',
    ])
    
    found_installations = []
    
    for path in possible_paths:
        python_exe = path / 'python.exe'
        if python_exe.exists():
            try:
                # Get version info
                result = subprocess.run(
                    [str(python_exe), '--version'],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                version_output = result.stdout.strip() or result.stderr.strip()
                found_installations.append((str(python_exe), version_output))
            except Exception as e:
                pass
    
    if found_installations:
        print("✅ Found Python 3.10 installation(s):")
        for exe_path, version in found_installations:
            print(f"   • {exe_path}")
            print(f"     Version: {version}")
        print()
        return True
    else:
        print("❌ No Python 3.10 installation found")
        print()
        return False


def show_installation_instructions():
    """Show instructions for installing Python 3.10"""
    print("="*80)
    print("📥 HOW TO INSTALL PYTHON 3.10 ON WINDOWS")
    print("="*80)
    print()
    print("Option 1: Download from Official Python Website (Recommended)")
    print("-" * 80)
    print("1. Visit: https://www.python.org/downloads/release/python-31011/")
    print("2. Scroll down to 'Files' section")
    print("3. Download one of these:")
    print("   • Windows installer (64-bit) - for most users")
    print("   • Windows installer (32-bit) - for older systems")
    print()
    print("4. Run the installer and IMPORTANT:")
    print("   ✅ Check 'Add Python 3.10 to PATH'")
    print("   ✅ Choose 'Install Now' or 'Customize installation'")
    print()
    print("5. After installation, restart your terminal/PowerShell")
    print("6. Verify with: python --version")
    print()
    
    print("Option 2: Using Windows Package Manager (winget)")
    print("-" * 80)
    print("Run in PowerShell or Command Prompt:")
    print("   winget install Python.Python.3.10")
    print()
    
    print("Option 3: Using Chocolatey")
    print("-" * 80)
    print("If you have Chocolatey installed:")
    print("   choco install python310")
    print()
    
    print("Option 4: Using Conda/Miniconda (For this project)")
    print("-" * 80)
    print("1. Install Miniconda: https://docs.conda.io/en/latest/miniconda.html")
    print("2. Open Anaconda Prompt or PowerShell")
    print("3. Create environment with Python 3.10:")
    print("   conda create -n thai-depression python=3.10 -y")
    print("4. Activate environment:")
    print("   conda activate thai-depression")
    print()
    
    print("="*80)
    print()


def check_py_launcher():
    """Check if Python launcher is available and can access Python 3.10"""
    print("🔧 Checking Python Launcher (py.exe)...")
    print()
    
    try:
        # Check if py.exe is available
        result = subprocess.run(
            ['py', '--list'],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        if result.returncode == 0:
            print("✅ Python Launcher is available")
            print("   Installed Python versions:")
            print()
            for line in result.stdout.split('\n'):
                if line.strip():
                    print(f"   {line}")
            print()
            
            # Check specifically for Python 3.10
            if '3.10' in result.stdout:
                print("✅ Python 3.10 is available via Python Launcher!")
                print("   You can use it with: py -3.10")
                print()
                
                # Test running Python 3.10
                test_result = subprocess.run(
                    ['py', '-3.10', '--version'],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                version_output = test_result.stdout.strip() or test_result.stderr.strip()
                print(f"   Test: {version_output}")
                print()
                return True
            else:
                print("❌ Python 3.10 not found in Python Launcher")
                print()
                return False
        else:
            print("⚠️  Python Launcher found but returned error")
            print()
            return False
            
    except FileNotFoundError:
        print("❌ Python Launcher (py.exe) not found")
        print("   Install Python from python.org to get the launcher")
        print()
        return False
    except Exception as e:
        print(f"⚠️  Error checking Python Launcher: {e}")
        print()
        return False


def main():
    """Main function"""
    is_windows = platform.system() == 'Windows'
    
    if not is_windows:
        print("⚠️  This script is designed for Windows only")
        print(f"   Your system: {platform.system()}")
        print()
        return
    
    # Check current Python version
    has_310_active = check_python_310()
    
    # Check for Python 3.10 via launcher
    has_310_launcher = check_py_launcher()
    
    # Check for Python 3.10 installations
    has_310_installed = check_other_python_installations()
    
    # Summary and recommendations
    print("="*80)
    print("📋 SUMMARY & RECOMMENDATIONS")
    print("="*80)
    print()
    
    if has_310_active:
        print("✅ READY TO USE!")
        print("   Python 3.10 is active and ready for the Thai Depression project")
        print()
        print("   Next steps:")
        print("   1. Run: python -m pip install --upgrade pip")
        print("   2. Run: .\\setup.ps1")
        print()
        
    elif has_310_launcher:
        print("⚠️  ACTION REQUIRED")
        print("   Python 3.10 is installed but not active by default")
        print()
        print("   Option A: Use Python Launcher")
        print("   • Replace 'python' with 'py -3.10' in commands")
        print("   • Example: py -3.10 train.py")
        print()
        print("   Option B: Create environment")
        print("   • Use conda: conda create -n thai-depression python=3.10")
        print("   • Or venv: py -3.10 -m venv venv")
        print()
        
    elif has_310_installed:
        print("⚠️  ACTION REQUIRED")
        print("   Python 3.10 is installed but not in PATH")
        print()
        print("   Option A: Add to PATH manually")
        print("   • Add the Python 3.10 installation directory to System PATH")
        print()
        print("   Option B: Create virtual environment")
        print("   • Navigate to Python 3.10 installation")
        print("   • Run: .\\python.exe -m venv C:\\path\\to\\thai-depression\\venv")
        print()
        
    else:
        print("❌ INSTALLATION NEEDED")
        print("   Python 3.10 is not installed on this system")
        print()
        show_installation_instructions()
    
    print("="*80)
    print()
    
    # Return exit code
    sys.exit(0 if has_310_active else 1)


if __name__ == "__main__":
    main()
