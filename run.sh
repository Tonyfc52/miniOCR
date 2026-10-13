#!/bin/bash
# MiniOCR 啟動腳本

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"

# 若 venv 不存在則自動建立與安裝
if [ ! -d "venv" ]; then
    echo "正在建立 Python 虛擬環境 (venv)..."
    python3 -m venv venv
    echo "正在安裝依賴套件 (PyQt6, rapidocr-onnxruntime, Pillow)..."
    ./venv/bin/pip install -r requirements.txt
fi

echo "啟動 MiniOCR..."
./venv/bin/python app.py
