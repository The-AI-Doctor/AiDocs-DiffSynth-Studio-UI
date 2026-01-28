@echo off
echo =========================================
echo DiffSynth-Studio UI Launcher
echo =========================================
echo.

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    pause
    exit /b 1
)

REM Create virtual environment if it doesn't exist
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo ERROR: Failed to create virtual environment
        pause
        exit /b 1
    )
    echo ✓ Virtual environment created
    echo.
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo ERROR: Failed to activate virtual environment
    pause
    exit /b 1
)

echo ✓ Virtual environment activated
echo.

REM Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip -q

REM Check for NVIDIA GPU and install appropriate PyTorch
echo.
echo Checking for NVIDIA GPU and installing appropriate PyTorch...
REM Try multiple paths where nvidia-smi might be located
where nvidia-smi >nul 2>&1
if errorlevel 1 (
    if exist "C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe" (
        set NVIDIA_FOUND=1
    ) else if exist "C:\Program Files (x86)\NVIDIA Corporation\NVSMI\nvidia-smi.exe" (
        set NVIDIA_FOUND=1
    ) else (
        set NVIDIA_FOUND=0
    )
) else (
    set NVIDIA_FOUND=1
)

if "%NVIDIA_FOUND%"=="1" (
    echo [OK] NVIDIA GPU detected
    REM Check if PyTorch with CUDA is already properly installed
    python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'" >nul 2>&1
    if errorlevel 1 (
        echo Installing PyTorch 2.7+ with CUDA 12.8 for RTX 5090...
        pip uninstall -y torch torchvision torchaudio >nul 2>&1
        pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
        echo [OK] PyTorch with CUDA installed
    ) else (
        echo [OK] PyTorch with CUDA already installed
    )
) else (
    echo [INFO] No NVIDIA GPU detected
    python -c "import torch; print('OK')" >nul 2>&1
    if errorlevel 1 (
        echo Installing PyTorch...
        pip install -q torch torchvision torchaudio
    ) else (
        echo [OK] PyTorch already installed
    )
)

echo.
echo Installing required packages...
pip install -q -r requirements.txt

if errorlevel 1 (
    echo ERROR: Failed to install dependencies
    pause
    exit /b 1
)

echo ✓ Dependencies installed
echo.
echo =========================================
echo Starting DiffSynth-Studio UI
echo =========================================
echo.
echo The application will be available at:
echo http://localhost:7860
echo.
echo Press Ctrl+C to stop the server
echo.

python app.py

pause
