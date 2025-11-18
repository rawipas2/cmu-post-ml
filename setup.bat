@echo off
REM Thai Depression Classification - Quick Setup for Windows
REM Double-click to run automatic setup

echo ================================================================================
echo Thai Depression Classification - Quick Setup
echo ================================================================================
echo.

REM ตรวจสอบ PowerShell execution policy
echo Checking PowerShell execution policy...
powershell -Command "Get-ExecutionPolicy"
echo.

echo This script will run setup.ps1 to install all dependencies.
echo.
echo Press any key to continue or Ctrl+C to cancel...
pause > nul

REM รัน PowerShell setup script
echo.
echo Running setup.ps1...
echo.

powershell -ExecutionPolicy Bypass -File "%~dp0setup.ps1"

echo.
echo ================================================================================
echo Setup script completed!
echo ================================================================================
echo.

pause
