@echo off
setlocal

set ENV_DIR=%~dp0env_pnn

if not exist "%ENV_DIR%\Scripts\activate.bat" (
    echo ERROR: The deep-learning pansharpening environment was not found at "%ENV_DIR%".
    echo Build and pack it using requirements_dlpan.yml first.
    pause
    exit /b 1
)

if not exist "%ENV_DIR%\__unpacked__.txt" (
    echo Running conda-unpack for the first time...
    "%ENV_DIR%\Scripts\conda-unpack.exe"
    echo done > "%ENV_DIR%\__unpacked__.txt"
)

echo Launching deep-learning pansharpening...
call "%ENV_DIR%\Scripts\activate.bat"
python "%~dp0app\pansharpen_cnn_launcher.py"
