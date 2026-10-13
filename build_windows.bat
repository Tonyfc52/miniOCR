@echo off
chcp 65001 >nul
title MiniOCR - Windows 執行檔編譯器

cd /d "%~dp0"

echo ======================================================
echo          MiniOCR - Windows (.exe) 打包編譯程式
echo ======================================================
echo.

:: 1. 確保虛擬環境存在
if not exist "venv\Scripts\python.exe" (
    echo [1/3] 正在建立虛擬環境...
    python -m venv venv
    if errorlevel 1 (
        echo [錯誤] 找不到 Python，請確認已安裝 Python 3.9+ 並勾選 Add to PATH
        pause
        exit /b 1
    )
)

:: 2. 安裝核心依賴與 PyInstaller
echo [2/3] 檢查並安裝必要套件 (PyInstaller, PyQt6, RapidOCR)...
call venv\Scripts\pip install -r requirements.txt
call venv\Scripts\pip install pyinstaller

if errorlevel 1 (
    echo [錯誤] 套件安裝失敗，請檢查網路連線
    pause
    exit /b 1
)

:: 3. 開始編譯為單一 .exe 執行檔
echo.
echo [3/3] 正在使用 PyInstaller 進行打包（此步驟大約需要 1~2 分鐘）...
echo 注意：正在將 ONNX 模型檔與 PyQt6 完整打包進 exe...
echo.

venv\Scripts\pyinstaller.exe ^
    --name "MiniOCR" ^
    --windowed ^
    --onefile ^
    --clean ^
    --noconfirm ^
    --collect-all rapidocr_onnxruntime ^
    --collect-all onnxruntime ^
    app.py

if errorlevel 1 (
    echo.
    echo [錯誤] 編譯過程發生錯誤！
    pause
    exit /b 1
)

echo.
echo ======================================================
echo                 編譯打包成功！
echo ======================================================
echo 您的執行檔已生成於：
echo    dist\MiniOCR.exe
echo.
echo 該檔案為獨立可執行檔，無需安裝 Python 即可直接在其他 Windows 電腦上運行！
echo ======================================================
echo.
pause
