# MiniOCR 輕量級圖片文字辨識工具 (PP-OCRv6 Medium 離線版)

專為 8GB RAM 無 GPU 的電腦環境設計，核心搭載 **PP-OCRv6 Medium (多語種/簡繁一體高精度模型)**，100% 本地離線運算，無需連網。

---

## ✨ 核心特色
1. **PP-OCRv6 Medium 離線模型**：
   - 包含多語種簡繁一體高精度模型檔（`models/PP-OCRv6_det_medium.onnx` 與 `models/PP-OCRv6_rec_medium.onnx`），免連網、零隱私疑慮。
2. **解決繁體字與漏字痛點**：
   - 原生支援繁體字庫，不再強制轉簡中。
   - 整合 **OpenCC（臺灣正體校正）** 下拉選單，可即時切換臺灣正體、港澳繁體或簡體。
   - 深度調校文字偵測門檻與文字框擴展（`unclip_ratio: 1.85`、`min_height: 5`），徹底杜絕首尾削字與小字漏字。
3. **8GB RAM 輕量低負載**：
   - 模型推論記憶體佔用僅約 200~300MB，CPU 運算流暢，完全不需要 GPU。
4. **靈活的圖片輸入方式**：
   - 📋 **剪貼簿貼上**：直接使用快捷鍵 `Cmd + V` (或 `Ctrl + V`) 貼上截圖。
   - 📁 **檔案上傳**：點擊「上傳圖片」或快捷鍵 `Cmd + O` 選取檔案。
   - 🖱️ **拖曳上傳**：直接將圖片檔案拖曳至左側預覽區。
5. **一鍵複製與匯出**：
   - 右側結果框可直接選取、編輯與複製。
   - 提供「一鍵複製」按鈕與「另存為文字檔 (.txt)」。
   - 支援「合併為單一段落」選項，方便複製文章。

---

## 🚀 啟動方式

本專案支援 **跨平台 (Windows / macOS / Linux)**。

### 🪟 Windows 環境
1. **方式一（最簡單）：** 
   - 直接滑鼠左鍵點擊兩下執行 **`run.bat`** 腳本。
   - （腳本會自動檢查 Python、建立虛擬環境、安裝套件並啟動程式）。
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

## 📦 打包成 Windows 獨立執行檔 (.exe)

本程式支援使用 **PyInstaller** 打包成無需安裝 Python 即可執行的綠色免安裝 `.exe` 檔：

### 方式一：一鍵打包（最推薦）
在 Windows 上直接滑鼠雙擊執行 **`build_windows.bat`**。
它會自動：
1. 下載並安裝 `pyinstaller`
2. 自動搜集並封裝 RapidOCR 的 ONNX 模型權重與配置檔（`--collect-all rapidocr_onnxruntime`）
3. 編譯完成後輸出於 **`dist\MiniOCR.exe`**。

### 方式二：手動指令打包
在 Windows 命令提示字元中執行：
```cmd
pip install pyinstaller
pyinstaller --name "MiniOCR" --windowed --onefile --clean --add-data "models;models" --collect-all rapidocr --collect-all onnxruntime --collect-all opencc_python_reimplemented app.py
```
生成的 `dist\MiniOCR.exe` 即可直接複製到其他沒有安裝 Python 的 Windows 電腦上使用！

---

## 📜 開源許可與模型授權聲明 (License & Attribution)

本專案遵循開源社群規範，依據 **Apache License 2.0** 進行合法歸屬聲明：

### 1. PP-OCRv6 核心模型授權
- **上游原始作品**：[PaddleOCR (Baidu)](https://github.com/PaddlePaddle/PaddleOCR)
- **模型格式轉換與適配**：[RapidOCR (RapidAI)](https://github.com/RapidAI/RapidOCR)
- **授權條款**：本專案使用之 **PP-OCRv6** ONNX 模型權重（`PP-OCRv6_det_medium.onnx`、`PP-OCRv6_rec_medium.onnx` 等）源自百度 PaddleOCR 衍生作品，並由 RapidOCR 團隊進行 ONNX 格式轉換適配。模型權重及轉換製品均依據 **[Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0)** 授權開源。
- **使用範圍**：依據 Apache-2.0 協議，使用者可自由進行學術研究、個人使用、修改與二次分發（包含商業應用），但必須完整保留原始版權歸屬聲明。

### 2. 第三方元件致謝
- **[RapidOCR](https://github.com/RapidAI/RapidOCR)**：提供跨平台高效 ONNX 推理架構 (Apache-2.0)。
- **[OpenCC (開放中文轉換)](https://github.com/BYVoid/OpenCC)**：提供高品質繁體中文臺灣正體轉換校正 (Apache-2.0)。
- **[PyQt6](https://www.riverbankcomputing.com/software/pyqt/)**：提供跨平台桌面圖形使用者介面框架。

### 3. 免責與商標聲明
- 本專案與百度（Baidu）、飛槳（PaddlePaddle）或 RapidAI 官方無直接隸屬關係。提及上述名稱僅用於客觀描述模型來源與技術依賴，不代表上述機構為本專案背書。
- 軟體均以「現狀（AS IS）」方式提供，不附帶任何明示或暗示之擔保。
