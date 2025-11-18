@echo off
REM Quick check script
echo ================================================================================
echo Thai Depression Classification - System Check
echo ================================================================================
echo.

REM ตรวจสอบว่าใช้ conda หรือ venv
if exist "%CONDA_PREFIX%" (
    echo Using conda environment: %CONDA_DEFAULT_ENV%
    echo.
    python setup_check.py
) else if exist "venv\Scripts\activate.bat" (
    echo Activating virtual environment...
    call venv\Scripts\activate.bat
    echo.
    python setup_check.py
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
