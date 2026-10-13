@echo off
chcp 65001 >nul
title MiniOCR

cd /d "%~dp0"

if not exist "venv\Scripts\python.exe" (
    echo 正在建立 Python 虛擬環境 (venv)...
    python -m venv venv
    if errorlevel 1 (
        echo 建立 venv 失敗，請確認已安裝 Python 3.9+ 並勾選 "Add Python to PATH"
        pause
        exit /b 1
    )
    echo 正在安裝依賴套件 (PyQt6, rapidocr-onnxruntime, Pillow)...
    call venv\Scripts\pip install -r requirements.txt
    if errorlevel 1 (
        echo 套件安裝失敗，請檢查網路連線
        pause
        exit /b 1
    )
)

echo 啟動 MiniOCR...
venv\Scripts\python.exe app.py
if errorlevel 1 (
    pause
)
