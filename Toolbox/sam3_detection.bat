@echo off
setlocal

set ENV_DIR=%~dp0env_sam3

if not exist "%ENV_DIR%\Scripts\activate.bat" (
    echo ERROR: The SAM 3 environment was not found at "%ENV_DIR%".
    echo Build and pack it using requirements_sam3.yml first.
    pause
    exit /b 1
)

if not exist "%ENV_DIR%\__unpacked__.txt" (
    echo Running conda-unpack for the first time...
    "%ENV_DIR%\Scripts\conda-unpack.exe"
    echo done > "%ENV_DIR%\__unpacked__.txt"
)

echo Launching SAM 3 crop-mark segmentation...
call "%ENV_DIR%\Scripts\activate.bat"
python "%~dp0app\sam3_cropmarks_launcher.py"

pause
