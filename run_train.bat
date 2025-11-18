@echo off
REM Quick run script for training
echo ================================================================================
echo Thai Depression Classification - Training
echo ================================================================================
echo.

REM ตรวจสอบว่าใช้ conda หรือ venv
if exist "%CONDA_PREFIX%" (
    echo Using conda environment: %CONDA_DEFAULT_ENV%
    echo.
    python train.py
) else if exist "venv\Scripts\activate.bat" (
    echo Activating virtual environment...
    call venv\Scripts\activate.bat
    echo.
    python train.py
) else (
    echo ERROR: No environment found!
    echo.
    echo Please run setup.bat or setup.ps1 first to install dependencies.
    echo.
    pause
    exit /b 1
)

echo.
pause
