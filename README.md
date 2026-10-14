# MiniOCR 輕量級圖片文字辨識工具 (PP-OCRv6 Small 離線輕巧版)

專為 8GB RAM 無 GPU 的電腦與一般辦公室電腦環境設計，核心搭載 **PP-OCRv6 Small (多語種/簡繁一體輕量高精度模型)**，100% 本地離線運算，無需連網。

---

## ✨ 核心特色
1. **PP-OCRv6 Small 離線輕巧模型**：
   - 模型總體積僅約 **30 MB**（`models/PP-OCRv6_det_small.onnx` 約 9.5MB、`models/PP-OCRv6_rec_small.onnx` 約 20MB），推論速度大幅提升，在低階辦公室電腦上也能極速秒開秒辨識。
2. **原生繁體中文與多語種支援**：
   - PP-OCRv6 字典庫原生涵蓋繁體字、簡體字、英文與數字，繁體字直接輸出繁體，無額外轉換損耗與語義偏誤。
3. **辨識超時防呆「停止」按鈕**：
   - 新增 **`⏹️ 停止`** 按鈕（亦可按快捷鍵 **`Esc`**），當遇到超大圖片或辨識過慢時，可隨時一鍵強制中斷，UI 絕不卡死凍結。
4. **防漏字調校**：
   - 預設開啟「小字增強 / 防漏字模式」，並擴展字框邊界（`unclip_ratio: 1.85`、`min_height: 5`），避免截圖邊緣字被削掉或小字漏字。
5. **靈活的圖片輸入方式**：
   - 📋 **剪貼簿貼上**：直接使用快捷鍵 `Cmd + V` (或 `Ctrl + V`) 貼上截圖。
   - 📁 **檔案上傳**：點擊「上傳圖片」或快捷鍵 `Cmd + O` 選取檔案。
   - 🖱️ **拖曳上傳**：直接將圖片檔案拖曳至左側預覽區。
6. **一鍵複製與匯出**：
   - 右側結果框可直接選取、編輯與複製。
   - 提供「一鍵複製」按鈕與「另存為文字檔 (.txt)」。
   - 支援「合併為單一段落」選項，方便複製文章。

---

## 🚀 啟動方式

本專案支援 **跨平台 (Windows / macOS / Linux)**。

### 🪟 Windows 環境
1. **方式一（最簡單）：** 
   - 直接滑鼠左鍵點擊兩下執行 **`run.bat`** 腳本。
   - （腳本會自動檢查 Python、建立虛擬環境、下載模型並啟動程式）。
2. **方式二（終端機 Command Prompt / PowerShell）：**
   ```cmd
   python app.py
   ```
   *註：已內建自動環境重定向，若全域已裝好套件或已有專案 `venv`，皆能自動識別並啟動。*

---

### 🍎 macOS / Linux 環境
在終端機進入專案目錄，執行以下任一指令：
```bash
./run.sh
# 或
python3 app.py
```

---

## ⌨️ 跨平台快捷鍵
| 功能 | macOS | Windows / Linux |
|---|---|---|
| 從剪貼簿貼上圖片 | `Cmd + V` | `Ctrl + V` |
| 開啟選取圖片檔案 | `Cmd + O` | `Ctrl + O` |
| 開始文字辨識 | `Cmd + Return` / `Ctrl + Return` | `Ctrl + Enter` |
| 停止/中斷辨識 | `Esc` | `Esc` |
| 一鍵複製右側全部文字 | `Cmd + Shift + C` | `Ctrl + Shift + C` |

---

## 📦 打包成 Windows 獨立執行檔 (.exe)

本程式支援使用 **PyInstaller** 打包成無需安裝 Python 即可執行的綠色免安裝 `.exe` 檔：

### 方式一：一鍵打包（最推薦）
在 Windows 上直接滑鼠雙擊執行 **`build_windows.bat`**。
它會自動：
1. 下載並安裝 `pyinstaller`
2. 自動搜集並封裝 PP-OCRv6 Small 的 ONNX 模型權重與配置檔
3. 編譯完成後輸出於 **`dist\MiniOCR.exe`**。

### 方式二：手動指令打包
在 Windows 命令提示字元中執行：
```cmd
pip install pyinstaller
pyinstaller --name "MiniOCR" --windowed --onefile --clean --add-data "models;models" --collect-all rapidocr --collect-all onnxruntime app.py
```
生成的 `dist\MiniOCR.exe` 即可直接複製到其他沒有安裝 Python 的 Windows 電腦上使用！

---

## 📜 開源許可與模型授權聲明 (License & Attribution)

本專案遵循開源社群規範，依據 **Apache License 2.0** 進行合法歸屬聲明：

### 1. PP-OCRv6 核心模型授權
- **上游原始作品**：[PaddleOCR (Baidu)](https://github.com/PaddlePaddle/PaddleOCR)
- **模型格式轉換與適配**：[RapidOCR (RapidAI)](https://github.com/RapidAI/RapidOCR)
- **授權條款**：本專案使用之 **PP-OCRv6** ONNX 模型權重（`PP-OCRv6_det_small.onnx`、`PP-OCRv6_rec_small.onnx` 等）源自百度 PaddleOCR 衍生作品，並由 RapidOCR 團隊進行 ONNX 格式轉換適配。模型權重及轉換製品均依據 **[Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0)** 授權開源。
- **使用範圍**：依據 Apache-2.0 協議，使用者可自由進行學術研究、個人使用、修改與二次分發（包含商業應用），但必須完整保留原始版權歸屬聲明。

### 2. 第三方元件致謝
- **[RapidOCR](https://github.com/RapidAI/RapidOCR)**：提供跨平台高效 ONNX 推理架構 (Apache-2.0)。
- **[PyQt6](https://www.riverbankcomputing.com/software/pyqt/)**：提供跨平台桌面圖形使用者介面框架。

### 3. 免責與商標聲明
- 本專案與百度（Baidu）、飛槳（PaddlePaddle）或 RapidAI 官方無直接隸屬關係。提及上述名稱僅用於客觀描述模型來源與技術依賴，不代表上述機構為本專案背書。
- 軟體均以「現狀（AS IS）」方式提供，不附帶任何明示或暗示之擔保。
