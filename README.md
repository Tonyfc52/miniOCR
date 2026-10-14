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
pyinstaller --name "MiniOCR" --windowed --onefile --clean --collect-all rapidocr_onnxruntime --collect-all onnxruntime app.py
```
生成的 `dist\MiniOCR.exe` 即可直接複製到其他沒有安裝 Python 的 Windows 電腦上使用！
