#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MiniOCR - 輕量級中英文圖片文字辨識工具 (PyQt6 + RapidOCR)
適合 8GB RAM 無 GPU 的電腦環境，提供極速、低記憶體佔用的 OCR 體驗。
"""

import os
import sys
from pathlib import Path

# 若為已編譯的獨立執行檔 (PyInstaller frozen)，則直接執行；否則自動切換至專案虛擬環境
if not getattr(sys, "frozen", False):
    _project_dir = Path(__file__).resolve().parent
    _candidates = [
        _project_dir / "venv" / "Scripts" / "python.exe",  # Windows
        _project_dir / "venv" / "bin" / "python",         # macOS / Linux
    ]
    for _candidate in _candidates:
        if _candidate.exists() and sys.executable.lower() != str(_candidate).lower():
            os.execv(str(_candidate), [str(_candidate)] + sys.argv)
            break

# 壓制 ONNX Runtime 的除錯雜訊
os.environ["ORT_LOGGING_LEVEL"] = "3"

import io
import time
from typing import Optional, List, Tuple

import numpy as np
from PIL import Image

from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QBuffer, QIODevice
from PyQt6.QtGui import QPixmap, QImage, QKeySequence, QShortcut, QAction, QFont, QColor, QPalette
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QPlainTextEdit, QFileDialog, QMessageBox,
    QFrame, QSplitter, QStatusBar, QProgressBar, QCheckBox, QComboBox, QSizePolicy
)

# OpenCC 繁簡轉換引擎
try:
    from opencc import OpenCC
    _cc_tw = OpenCC("s2twp")  # 繁體中文（臺灣正體與用語校正）
    _cc_hk = OpenCC("s2t")    # 繁體中文（標準繁體）
    _cc_cn = OpenCC("t2s")    # 簡體中文
except Exception:
    _cc_tw = None
    _cc_hk = None
    _cc_cn = None

# 支援的圖片副檔名
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tiff", ".tif", ".gif"}


class OCRWorker(QThread):
    """在背景線程執行 OCR 運算，採用防漏字與高靈敏度參數"""
    finished = pyqtSignal(list, float)  # (ocr_results, elapsed_seconds)
    error = pyqtSignal(str)

    _engine = None  # 靜態共享模型實例，只需初始化一次

    def __init__(self, image_array: np.ndarray, enhance_small_text: bool = True):
        super().__init__()
        self.image_array = image_array
        self.enhance_small_text = enhance_small_text

    @classmethod
    def get_engine(cls):
        if cls._engine is None:
            # 延遲載入 RapidOCR 引擎（基於 ONNX Runtime CPU，防漏字調優）
            from rapidocr_onnxruntime import RapidOCR
            cls._engine = RapidOCR(
                min_height=5,            # 原預設 30 會漏掉小於 30px 的截圖文字，降至 5 徹底避免漏行
                width_height_ratio=35,   # 原預設 8 會漏掉超長句子，放寬至 35
                text_score=0.35,         # 靈敏度門檻，捕捉較淡或筆畫複雜之文字
                det_model_path=None,     # 維持內建模型
                det_thresh=0.20,         # 原 0.3，降低二值化門檻以捕捉低對比或淺色字
                det_box_thresh=0.40,     # 原 0.5，降低文字框過濾門檻，減少漏框
                det_unclip_ratio=1.85,   # 原 1.6，適度擴大文字框邊界，避免首尾字元被削邊
                det_limit_side_len=1800, # 原 736，避免高解析度截圖被強行縮小導致小字模糊
            )
        return cls._engine

    def run(self):
        try:
            start_time = time.time()
            engine = self.get_engine()

            # 智慧小字放大增強：若圖片高度小於 150px 或寬度小於 300px，進行高品質放大
            processed_img = self.image_array
            if self.enhance_small_text:
                h, w = processed_img.shape[:2]
                if h < 160 or w < 320:
                    pil_img = Image.fromarray(processed_img)
                    scale = min(3.0, max(2.0, 360.0 / max(h, 1)))
                    new_w, new_h = int(w * scale), int(h * scale)
                    pil_img = pil_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
                    processed_img = np.array(pil_img)

            results, elapse = engine(processed_img)
            elapsed = time.time() - start_time
            # results 為 [ [box, text, score], ... ]
            formatted_results = []
            if results:
                for item in results:
                    text = item[1]
                    score = float(item[2])
                    formatted_results.append((text, score))
            self.finished.emit(formatted_results, elapsed)
        except Exception as e:
            self.error.emit(str(e))


class DropAreaLabel(QLabel):
    """支援拖曳圖片、剪貼簿貼上與縮放預覽的圖片顯示區域"""
    image_dropped = pyqtSignal(QImage)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumSize(320, 360)
        self._current_pixmap: Optional[QPixmap] = None
        self.show_placeholder()

    def show_placeholder(self):
        self._current_pixmap = None
        mod_key = "Cmd" if sys.platform == "darwin" else "Ctrl"
        self.setText(
            f"📷 請上傳圖片或從剪貼簿貼上\n\n"
            f"• 支援快捷鍵 {mod_key} + V 直接貼上截圖\n"
            f"• 支援拖曳圖檔至此處\n"
            f"• 支援繁簡中文、英文與數字辨識"
        )
        self.setStyleSheet("""
            QLabel {
                border: 2px dashed #B0B8C1;
                border-radius: 12px;
                background-color: #F8F9FA;
                color: #6C757D;
                font-size: 14px;
                line-height: 1.6;
                padding: 24px;
            }
        """)

    def set_display_pixmap(self, pixmap: QPixmap):
        self._current_pixmap = pixmap
        self.setStyleSheet("""
            QLabel {
                border: 1px solid #D0D7DE;
                border-radius: 12px;
                background-color: #FFFFFF;
                padding: 8px;
            }
        """)
        self.update_scaled_pixmap()

    def update_scaled_pixmap(self):
        if self._current_pixmap and not self._current_pixmap.isNull():
            # 依目前區域縮放，保持長寬比
            target_size = self.size() - QtCore.QSize(20, 20)
            if target_size.width() > 10 and target_size.height() > 10:
                scaled = self._current_pixmap.scaled(
                    target_size,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation
                )
                self.setPixmap(scaled)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._current_pixmap:
            self.update_scaled_pixmap()

    def dragEnterEvent(self, event: QtGui.QDragEnterEvent):
        mime = event.mimeData()
        if mime.hasUrls() or mime.hasImage():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event: QtGui.QDropEvent):
        mime = event.mimeData()
        if mime.hasImage():
            img = qimage_from_mime(mime)
            if img and not img.isNull():
                self.image_dropped.emit(img)
                event.acceptProposedAction()
                return

        if mime.hasUrls():
            for url in mime.urls():
                file_path = url.toLocalFile()
                p = Path(file_path)
                if p.suffix.lower() in IMAGE_EXTENSIONS:
                    loaded_img = QImage(file_path)
                    if not loaded_img.isNull():
                        self.image_dropped.emit(loaded_img)
                        event.acceptProposedAction()
                        return
        event.ignore()


def qimage_from_mime(mime_data: QtCore.QMimeData) -> Optional[QImage]:
    """從 QMimeData 提取 QImage，優先嘗試 native image，再嘗試 URL"""
    if mime_data.hasImage():
        qimg = mime_data.imageData()
        if isinstance(qimg, QImage) and not qimg.isNull():
            return qimg
        elif isinstance(qimg, QPixmap) and not qimg.isNull():
            return qimg.toImage()

    if mime_data.hasUrls():
        for url in mime_data.urls():
            path = url.toLocalFile()
            if Path(path).suffix.lower() in IMAGE_EXTENSIONS:
                img = QImage(path)
                if not img.isNull():
                    return img
    return None


def qimage_to_numpy(qimg: QImage) -> np.ndarray:
    """安全地將 QImage 轉為 NumPy RGB 陣列（適用於 RapidOCR）"""
    buffer = QBuffer()
    buffer.open(QIODevice.OpenModeFlag.WriteOnly)
    qimg.save(buffer, "PNG")
    bytes_data = bytes(buffer.data())
    pil_image = Image.open(io.BytesIO(bytes_data)).convert("RGB")
    return np.array(pil_image)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MiniOCR - 輕量中英文圖片文字辨識")
        self.resize(1000, 680)
        self.setMinimumSize(800, 520)

        self.current_image: Optional[QImage] = None
        self.raw_ocr_items: List[Tuple[str, float]] = []
        self.worker: Optional[OCRWorker] = None

        self.init_ui()
        self.setup_shortcuts()
        self.apply_theme()

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(18, 16, 18, 12)
        main_layout.setSpacing(12)

        # 頂部標題與說明
        header_layout = QHBoxLayout()
        title_label = QLabel("⚡ MiniOCR 圖片文字辨識")
        title_font = QFont()
        title_font.setPointSize(16)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setStyleSheet("color: #1F2328;")

        desc_label = QLabel("輕量高效 • 支援中英文 • 8GB RAM 流暢運行")
        desc_label.setStyleSheet("color: #656D76; font-size: 13px;")

        header_layout.addWidget(title_label)
        header_layout.addSpacing(10)
        header_layout.addWidget(desc_label)
        header_layout.addStretch()

        main_layout.addLayout(header_layout)

        # 左右主分割視窗 (QSplitter)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(8)

        # ====== 左側：圖片區域 ======
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 8, 0)
        left_layout.setSpacing(10)

        # 左側工具列（按鈕）
        left_tools = QHBoxLayout()
        left_tools.setSpacing(8)

        mod_key = "Cmd" if sys.platform == "darwin" else "Ctrl"

        self.btn_upload = QPushButton("📁 上傳圖片")
        self.btn_upload.setToolTip(f"從電腦中選取圖片檔案 ({mod_key}+O)")
        self.btn_upload.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_upload.clicked.connect(self.action_upload_image)

        self.btn_paste = QPushButton("📋 從剪貼簿貼上")
        self.btn_paste.setToolTip(f"貼上截圖或已複製的圖片 ({mod_key}+V)")
        self.btn_paste.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_paste.clicked.connect(self.action_paste_image)

        self.btn_clear_img = QPushButton("🗑️ 清除")
        self.btn_clear_img.setToolTip("清除當前圖片")
        self.btn_clear_img.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_clear_img.clicked.connect(self.action_clear_image)

        left_tools.addWidget(self.btn_upload)
        left_tools.addWidget(self.btn_paste)
        left_tools.addWidget(self.btn_clear_img)
        left_tools.addStretch()

        left_layout.addLayout(left_tools)

        # 圖片預覽區域
        self.drop_label = DropAreaLabel()
        self.drop_label.image_dropped.connect(self.load_image_object)
        left_layout.addWidget(self.drop_label, 1)

        # 圖片資訊標籤
        self.img_info_label = QLabel("未載入圖片")
        self.img_info_label.setStyleSheet("color: #656D76; font-size: 12px;")
        left_layout.addWidget(self.img_info_label)

        # 辨識選項
        opt_layout = QHBoxLayout()
        self.chk_enhance = QCheckBox("🔍 小字增強 / 防漏字模式")
        self.chk_enhance.setChecked(True)
        self.chk_enhance.setToolTip("針對截圖小字進行智慧放大並開啟高靈敏偵測，防止漏字漏行")
        opt_layout.addWidget(self.chk_enhance)
        opt_layout.addStretch()
        left_layout.addLayout(opt_layout)

        # 開始辨識按鈕
        self.btn_recognize = QPushButton("🔍 開始辨識 (Recognize)")
        self.btn_recognize.setFixedHeight(44)
        self.btn_recognize.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_recognize.setStyleSheet("""
            QPushButton {
                background-color: #0969DA;
                color: #FFFFFF;
                font-size: 15px;
                font-weight: bold;
                border-radius: 8px;
                border: none;
            }
            QPushButton:hover {
                background-color: #0856B7;
            }
            QPushButton:pressed {
                background-color: #06408A;
            }
            QPushButton:disabled {
                background-color: #8C959F;
                color: #EAEEF2;
            }
        """)
        self.btn_recognize.clicked.connect(self.action_start_ocr)
        left_layout.addWidget(self.btn_recognize)

        splitter.addWidget(left_widget)

        # ====== 右側：辨識文字結果 ======
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(8, 0, 0, 0)
        right_layout.setSpacing(10)

        # 右側頂部工具列
        right_tools = QHBoxLayout()
        right_tools.setSpacing(8)

        self.btn_copy = QPushButton("📋 一鍵複製")
        self.btn_copy.setToolTip("複製右側所有文字至剪貼簿 (Cmd+Shift+C)")
        self.btn_copy.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_copy.setStyleSheet("""
            QPushButton {
                background-color: #1A7F37;
                color: #FFFFFF;
                font-weight: bold;
                border-radius: 6px;
                padding: 6px 14px;
                border: none;
            }
            QPushButton:hover {
                background-color: #15662C;
            }
            QPushButton:disabled {
                background-color: #8C959F;
            }
        """)
        self.btn_copy.clicked.connect(self.action_copy_text)

        self.btn_save = QPushButton("💾 另存文字")
        self.btn_save.setToolTip("將辨識結果儲存為 .txt")
        self.btn_save.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_save.clicked.connect(self.action_save_txt)

        self.btn_clear_text = QPushButton("🧹 清空")
        self.btn_clear_text.setToolTip("清空文字框")
        self.btn_clear_text.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_clear_text.clicked.connect(self.action_clear_text)

        right_tools.addWidget(self.btn_copy)
        right_tools.addWidget(self.btn_save)
        right_tools.addWidget(self.btn_clear_text)
        right_tools.addStretch()

        # 語言輸出選項下拉選單
        self.combo_lang = QComboBox()
        self.combo_lang.addItems([
            "🇹🇼 繁體中文 (臺灣正體)",
            "🇭🇰 繁體中文 (標準繁體)",
            "🇨🇳 簡體中文",
            "🔤 保持原始識別",
        ])
        self.combo_lang.setToolTip("選擇文字轉換格式（切換後即時生效，不需重新辨識）")
        self.combo_lang.currentIndexChanged.connect(self.refresh_text_display)
        right_tools.addWidget(self.combo_lang)

        # 合併段落選項
        self.chk_merge_lines = QCheckBox("合併為單一段落")
        self.chk_merge_lines.setToolTip("勾選後將文字行串接成段落（適合複製文章），取消則保留每行辨識位置換行")
        self.chk_merge_lines.toggled.connect(self.refresh_text_display)
        right_tools.addWidget(self.chk_merge_lines)

        right_layout.addLayout(right_tools)

        # 文字顯示與編輯框
        self.text_edit = QPlainTextEdit()
        self.text_edit.setPlaceholderText("辨識出的文字將會顯示在此處，可直接編輯、選取與複製...")
        font = QFont()
        font.setFamilies(["PingFang SC", "Microsoft JhengHei", "Microsoft YaHei", "Segoe UI", "sans-serif"])
        font.setPointSize(14)
        self.text_edit.setFont(font)
        self.text_edit.textChanged.connect(self.update_char_stats)
        right_layout.addWidget(self.text_edit, 1)

        # 文字統計
        self.stats_label = QLabel("共 0 個字元 / 0 行")
        self.stats_label.setStyleSheet("color: #656D76; font-size: 12px;")
        right_layout.addWidget(self.stats_label)

        splitter.addWidget(right_widget)

        # 預設 50% / 50% 比例分配
        splitter.setSizes([480, 520])
        main_layout.addWidget(splitter, 1)

        # 狀態列
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)  # 不確定進度動畫
        self.progress_bar.setFixedWidth(130)
        self.progress_bar.setFixedHeight(14)
        self.progress_bar.setVisible(False)
        self.status_bar.addPermanentWidget(self.progress_bar)

        self.status_bar.showMessage("就緒：請貼上或上傳圖片")

    def setup_shortcuts(self):
        """配置跨平台快捷鍵 (macOS 與 Windows 均適用)"""
        # 貼上圖片 (Cmd+V on Mac / Ctrl+V on Windows)
        paste_sc = QShortcut(QKeySequence.StandardKey.Paste, self)
        paste_sc.activated.connect(self.action_paste_image)

        # 開啟圖片檔案 (Cmd+O on Mac / Ctrl+O on Windows)
        open_sc = QShortcut(QKeySequence.StandardKey.Open, self)
        open_sc.activated.connect(self.action_upload_image)

        # 開始辨識 (Ctrl+Return 或 Cmd+Return)
        run_sc1 = QShortcut(QKeySequence("Ctrl+Return"), self)
        run_sc1.activated.connect(self.action_start_ocr)
        if sys.platform == "darwin":
            run_sc2 = QShortcut(QKeySequence("Meta+Return"), self)
            run_sc2.activated.connect(self.action_start_ocr)

        # 一鍵複製文字 (Ctrl+Shift+C 或 Cmd+Shift+C)
        copy_all_sc1 = QShortcut(QKeySequence("Ctrl+Shift+C"), self)
        copy_all_sc1.activated.connect(self.action_copy_text)
        if sys.platform == "darwin":
            copy_all_sc2 = QShortcut(QKeySequence("Meta+Shift+C"), self)
            copy_all_sc2.activated.connect(self.action_copy_text)

    def apply_theme(self):
        """設定精美且易閱讀的扁平化介面樣式"""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #F6F8FA;
            }
            QPushButton {
                background-color: #FFFFFF;
                color: #24292F;
                border: 1px solid #D0D7DE;
                border-radius: 6px;
                padding: 6px 12px;
                font-size: 13px;
            }
            QPushButton:hover {
                background-color: #F3F4F6;
                border-color: #B0B8C1;
            }
            QPushButton:pressed {
                background-color: #E7E9EC;
            }
            QPlainTextEdit {
                background-color: #FFFFFF;
                color: #1F2328;
                border: 1px solid #D0D7DE;
                border-radius: 8px;
                padding: 10px;
                selection-background-color: #0969DA;
                selection-color: #FFFFFF;
                line-height: 1.5;
            }
            QCheckBox {
                color: #424A53;
                font-size: 13px;
            }
            QComboBox {
                background-color: #FFFFFF;
                color: #24292F;
                border: 1px solid #D0D7DE;
                border-radius: 6px;
                padding: 4px 8px;
                font-size: 13px;
                min-width: 170px;
            }
            QComboBox:hover {
                background-color: #F3F4F6;
                border-color: #B0B8C1;
            }
            QStatusBar {
                color: #57606A;
                background-color: #FFFFFF;
                border-top: 1px solid #D8DEE4;
                font-size: 12px;
            }
            QSplitter::handle {
                background-color: #E1E4E8;
                border-radius: 3px;
            }
        """)

    # ================= 動作回調 (Actions) =================

    def action_upload_image(self):
        """選擇本機圖檔"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "選取要辨識的圖片",
            "",
            "圖片檔案 (*.png *.jpg *.jpeg *.bmp *.webp *.tiff);;所有檔案 (*.*)"
        )
        if file_path:
            img = QImage(file_path)
            if img.isNull():
                QMessageBox.warning(self, "讀取失敗", "無法載入此圖片檔案，請確認格式是否正確。")
                return
            self.load_image_object(img, source_info=Path(file_path).name)

    def action_paste_image(self):
        """從系統剪貼簿讀取圖片"""
        clipboard = QApplication.clipboard()
        mime_data = clipboard.mimeData()

        img = qimage_from_mime(mime_data)
        if img and not img.isNull():
            self.load_image_object(img, source_info="剪貼簿圖片")
            self.status_bar.showMessage("已成功從剪貼簿載入圖片！按「開始辨識」進行提取", 4000)
        else:
            self.status_bar.showMessage("剪貼簿內目前沒有圖片資料（可使用截圖快捷鍵後再按此處）", 4000)

    def load_image_object(self, img: QImage, source_info: str = "載入圖片"):
        """載入 QImage 物件至介面"""
        self.current_image = img
        pixmap = QPixmap.fromImage(img)
        self.drop_label.set_display_pixmap(pixmap)
        w, h = img.width(), img.height()
        self.img_info_label.setText(f"來源：{source_info} | 尺寸：{w} × {h} px")
        self.status_bar.showMessage(f"圖片已就緒 ({w}x{h})，點擊「開始辨識」開始提取文字")

    def action_clear_image(self):
        """清除圖片預覽"""
        self.current_image = None
        self.drop_label.show_placeholder()
        self.img_info_label.setText("未載入圖片")
        self.status_bar.showMessage("已清除圖片")

    def action_clear_text(self):
        """清空文字結果"""
        self.raw_ocr_items = []
        self.text_edit.clear()
        self.status_bar.showMessage("已清空文字框")

    def action_copy_text(self):
        """複製辨識結果至剪貼簿"""
        text = self.text_edit.toPlainText().strip()
        if not text:
            self.status_bar.showMessage("目前沒有可複製的文字！", 3000)
            return

        clipboard = QApplication.clipboard()
        clipboard.setText(text)
        self.status_bar.showMessage(f"✅ 已成功複製 {len(text)} 個字元至剪貼簿！", 4000)

    def action_save_txt(self):
        """將辨識出的文字存成檔案"""
        text = self.text_edit.toPlainText().strip()
        if not text:
            QMessageBox.information(self, "提示", "目前沒有文字可供儲存。")
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "另存文字檔",
            "ocr_result.txt",
            "文字檔案 (*.txt);;所有檔案 (*.*)"
        )
        if file_path:
            try:
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(text)
                self.status_bar.showMessage(f"✅ 已儲存至：{file_path}", 4000)
            except Exception as e:
                QMessageBox.critical(self, "儲存失敗", f"寫入檔案時發生錯誤：\n{e}")

    def update_char_stats(self):
        """更新字數與行數統計"""
        text = self.text_edit.toPlainText()
        char_count = len(text)
        lines = len(text.splitlines()) if char_count > 0 else 0
        self.stats_label.setText(f"共 {char_count} 個字元 / {lines} 行")

    def refresh_text_display(self):
        """依據語言選項與「合併段落」選項重新組織文字呈現"""
        if not self.raw_ocr_items:
            return

        raw_lines = [item[0] for item in self.raw_ocr_items]

        # 根據下拉選單進行語言轉換
        lang_idx = self.combo_lang.currentIndex()
        converted_lines = []
        for line in raw_lines:
            if lang_idx == 0 and _cc_tw:    # 臺灣正體
                converted_lines.append(_cc_tw.convert(line))
            elif lang_idx == 1 and _cc_hk:  # 標準繁體
                converted_lines.append(_cc_hk.convert(line))
            elif lang_idx == 2 and _cc_cn:  # 簡體中文
                converted_lines.append(_cc_cn.convert(line))
            else:
                converted_lines.append(line)

        if self.chk_merge_lines.isChecked():
            # 合併為段落（處理中英文空白間隔）
            merged = ""
            for i, line in enumerate(converted_lines):
                if not merged:
                    merged = line
                else:
                    # 若上一行結尾是英數字且此行開頭是英數字，補空格
                    if merged[-1].isalnum() and line[0].isalnum():
                        merged += " " + line
                    else:
                        merged += line
            self.text_edit.setPlainText(merged)
        else:
            self.text_edit.setPlainText("\n".join(converted_lines))

    def action_start_ocr(self):
        """觸發 OCR 辨識"""
        if self.current_image is None or self.current_image.isNull():
            QMessageBox.information(self, "提示", "請先上傳圖片或從剪貼簿貼上圖片再進行辨識。")
            return

        # 鎖定按鈕，防止重複點擊
        self.btn_recognize.setEnabled(False)
        self.btn_recognize.setText("⏳ 辨識中，請稍候...")
        self.progress_bar.setVisible(True)
        self.status_bar.showMessage("正在載入輕量模型並辨識中，請稍候...")

        # 轉換 QImage 為 NumPy 陣列
        img_array = qimage_to_numpy(self.current_image)

        # 啟動背景 Worker 線程（包含防漏字小字增強設定）
        enhance = self.chk_enhance.isChecked()
        self.worker = OCRWorker(img_array, enhance_small_text=enhance)
        self.worker.finished.connect(self.on_ocr_finished)
        self.worker.error.connect(self.on_ocr_error)
        self.worker.start()

    def on_ocr_finished(self, results: list, elapsed: float):
        """OCR 完成回調"""
        self.btn_recognize.setEnabled(True)
        self.btn_recognize.setText("🔍 開始辨識 (Recognize)")
        self.progress_bar.setVisible(False)

        self.raw_ocr_items = results
        if not results:
            self.text_edit.setPlainText("（未偵測到任何文字）")
            self.status_bar.showMessage(f"辨識完成，耗時 {elapsed:.2f} 秒（未找到文字）", 4000)
            return

        self.refresh_text_display()
        total_lines = len(results)
        self.status_bar.showMessage(f"✅ 辨識完成！耗時 {elapsed:.2f} 秒，共提取 {total_lines} 行文字", 5000)

    def on_ocr_error(self, err_msg: str):
        """OCR 錯誤回調"""
        self.btn_recognize.setEnabled(True)
        self.btn_recognize.setText("🔍 開始辨識 (Recognize)")
        self.progress_bar.setVisible(False)
        self.status_bar.showMessage(f"辨識過程出錯：{err_msg}", 6000)
        QMessageBox.critical(self, "辨識失敗", f"發生例外狀況：\n{err_msg}")


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("MiniOCR")

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
