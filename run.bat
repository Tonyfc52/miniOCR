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
    echo 正在安裝依賴套件 (PyQt6, RapidOCR, Pillow, OpenCC)...
    call venv\Scripts\pip install -r requirements.txt
    if errorlevel 1 (
        echo 套件安裝失敗，請檢查網路連線
        pause
        exit /b 1
    )
)

if not exist "models\PP-OCRv6_det_small.onnx" (
    echo 正在下載 PP-OCRv6 Small 離線模型 (約 30MB)...
    if not exist "models" mkdir models
    curl -L -o models\PP-OCRv6_det_small.onnx "https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.10.0/onnx/PP-OCRv6/det/PP-OCRv6_det_small.onnx"
    curl -L -o models\PP-OCRv6_rec_small.onnx "https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.10.0/onnx/PP-OCRv6/rec/PP-OCRv6_rec_small.onnx"
)

echo 啟動 MiniOCR (PP-OCRv6 Small 離線版)...
venv\Scripts\python.exe app.py
if errorlevel 1 (
    pause
)
