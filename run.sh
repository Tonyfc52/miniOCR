#!/bin/bash
# MiniOCR 啟動腳本

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"

# 若 venv 不存在則自動建立與安裝
if [ ! -d "venv" ]; then
    echo "正在建立 Python 虛擬環境 (venv)..."
    python3 -m venv venv
    echo "正在安裝依賴套件 (PyQt6, RapidOCR, Pillow, OpenCC)..."
    ./venv/bin/pip install -r requirements.txt
fi

# 確保離線模型存在
if [ ! -f "models/PP-OCRv6_det_small.onnx" ]; then
    echo "正在下載 PP-OCRv6 Small 離線模型 (約 30MB)..."
    mkdir -p models
    curl -L -o models/PP-OCRv6_det_small.onnx "https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.10.0/onnx/PP-OCRv6/det/PP-OCRv6_det_small.onnx"
    curl -L -o models/PP-OCRv6_rec_small.onnx "https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.10.0/onnx/PP-OCRv6/rec/PP-OCRv6_rec_small.onnx"
fi

echo "啟動 MiniOCR (PP-OCRv6 Small 離線版)..."
./venv/bin/python app.py
