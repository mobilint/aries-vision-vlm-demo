@echo off
setlocal enabledelayedexpansion

set "ASSET_BUCKET_ID=%HF_ASSET_BUCKET_ID%"
if "%ASSET_BUCKET_ID%"=="" set "ASSET_BUCKET_ID=mobilint/aries-weapon-detection-demo-assets"
set "ASSET_BUCKET_URI=hf://buckets/%ASSET_BUCKET_ID%"

set "SCRIPT_DIR=%~dp0"
set "VENV_DIR=%HF_DOWNLOAD_VENV_DIR%"
if "%VENV_DIR%"=="" set "VENV_DIR=%SCRIPT_DIR%.hf_venv"
set "LOCAL_DIR=%SCRIPT_DIR%backend_vision\assets"

where uv >nul 2>nul
if errorlevel 1 (
  echo uv not found. Please install uv before running this script.
  echo See: https://docs.astral.sh/uv/getting-started/installation/
  exit /b 1
)

echo Preparing Hugging Face download environment: %VENV_DIR%
if not exist "%VENV_DIR%" (
  uv venv "%VENV_DIR%"
  if errorlevel 1 exit /b 1
)

set "VENV_PYTHON=%VENV_DIR%\Scripts\python.exe"
if not exist "%VENV_PYTHON%" (
  echo Cannot find venv python at %VENV_PYTHON%
  exit /b 1
)

uv pip install --python "%VENV_PYTHON%" huggingface-hub
if errorlevel 1 exit /b 1

set "HF_CLI=%VENV_DIR%\Scripts\hf.exe"
if not exist "%HF_CLI%" set "HF_CLI=%VENV_DIR%\Scripts\hf"
if not exist "%HF_CLI%" (
  echo Cannot find hf CLI in %VENV_DIR%\Scripts
  exit /b 1
)

if not exist "%LOCAL_DIR%" mkdir "%LOCAL_DIR%"

echo Downloading vision assets from Hugging Face bucket: %ASSET_BUCKET_URI%
"%HF_CLI%" buckets sync "%ASSET_BUCKET_URI%" "%LOCAL_DIR%" ^
  --include "config/*.yaml" ^
  --include "layout/*" ^
  --include "mxq/*" ^
  --include "video/**/*.mp4"
if errorlevel 1 exit /b 1

echo Vision assets downloaded to %LOCAL_DIR%
endlocal