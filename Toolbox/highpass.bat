@echo off
setlocal

set ENV_DIR=%~dp0env

:: --- Run conda-unpack once only ---
if not exist "%ENV_DIR%\__unpacked__.txt" (
    echo Running conda-unpack for the first time...
    "%ENV_DIR%\Scripts\conda-unpack.exe"
    echo done > "%ENV_DIR%\__unpacked__.txt"
)

:: --- Activate and run your tool ---
echo Launching the application...
call "%ENV_DIR%\Scripts\activate.bat"
python "%~dp0\app\highpass_launcher.py"