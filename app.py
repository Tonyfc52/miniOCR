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
import onnxruntime  # 明確引入，確保 PyInstaller 與靜態打包能完整納入 onnxruntime 與 C++ DLL

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
    QFrame, QSplitter, QStatusBar, QProgressBar, QCheckBox, QSizePolicy
)

# 支援的圖片副檔名
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tiff", ".tif", ".gif"}


def get_offline_v6_models() -> Tuple[Optional[Path], Optional[Path]]:
    """取得本地離線 PP-OCRv6 Small 模型路徑（體積極小、推論極快）"""
    if getattr(sys, "frozen", False):
        base_dir = Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))
    else:
        base_dir = Path(__file__).resolve().parent

    det_path = base_dir / "models" / "PP-OCRv6_det_small.onnx"
    rec_path = base_dir / "models" / "PP-OCRv6_rec_small.onnx"
    if det_path.exists() and rec_path.exists():
        return det_path, rec_path
    return None, None


class OCRWorker(QThread):
    """在背景線程執行 OCR 運算，優先載入本地離線 PP-OCRv6 Small 輕巧模型"""
    finished = pyqtSignal(list, float)  # (ocr_results, elapsed_seconds)
    error = pyqtSignal(str)

    _engine = None  # 靜態共享模型實例
    _engine_type = "v6_small"  # "v6_small" 或 "v3_legacy"

    def __init__(self, image_array: np.ndarray, enhance_small_text: bool = True):
        super().__init__()
        self.image_array = image_array
        self.enhance_small_text = enhance_small_text

    @classmethod
    def get_engine(cls):
        if cls._engine is None:
            det_v6, rec_v6 = get_offline_v6_models()
            if det_v6 and rec_v6:
                # 載入 PP-OCRv6 Small 離線模型（極速、原生支援繁體中文與多語種）
                from rapidocr import RapidOCR
                cls._engine = RapidOCR(params={
                    "Global.log_level": "error",
                    "Global.use_cls": False,
                    "Global.text_score": 0.35,
                    "Global.min_height": 5,
                    "Global.width_height_ratio": 35,
                    "Det.model_path": str(det_v6),
                    "Det.unclip_ratio": 1.85,
                    "Det.limit_side_len": 1800,
                    "Det.thresh": 0.20,
                    "Det.box_thresh": 0.40,
                    "Rec.model_path": str(rec_v6),
                })
                cls._engine_type = "v6_small"
            else:
                # 後備：載入預設 rapidocr-onnxruntime
                from rapidocr_onnxruntime import RapidOCR
                cls._engine = RapidOCR(
                    min_height=5,
                    width_height_ratio=35,
                    text_score=0.35,
                    det_model_path=None,
                    det_thresh=0.20,
                    det_box_thresh=0.40,
                    det_unclip_ratio=1.85,
                    det_limit_side_len=1800,
                )
                cls._engine_type = "v3_legacy"
        return cls._engine

    def run(self):
        try:
            start_time = time.time()
            engine = self.get_engine()

            # 智慧小字放大增強：若圖片高度小於 160px 或寬度小於 320px，進行高品質插值放大
            processed_img = self.image_array
            if self.enhance_small_text:
                h, w = processed_img.shape[:2]
                if h < 160 or w < 320:
                    pil_img = Image.fromarray(processed_img)
                    scale = min(3.0, max(2.0, 360.0 / max(h, 1)))
                    new_w, new_h = int(w * scale), int(h * scale)
                    pil_img = pil_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
                    processed_img = np.array(pil_img)

            formatted_results = []
            if "v6" in self._engine_type:
                out = engine(processed_img)
                elapsed = out.elapse if (hasattr(out, "elapse") and out.elapse) else (time.time() - start_time)
                if out and getattr(out, "txts", None):
                    for text, score in zip(out.txts, out.scores):
                        formatted_results.append((text, float(score)))
            else:
                results, elapse = engine(processed_img)
                elapsed = time.time() - start_time
                if results:
                    for item in results:
                        formatted_results.append((item[1], float(item[2])))

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

        desc_label = QLabel("🚀 PP-OCRv6 Small 離線版 • 原生繁中多語種 • 辦公極速流暢")
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

        # 開始辨識與停止按鈕列
        action_btn_layout = QHBoxLayout()
        action_btn_layout.setSpacing(8)

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

        self.btn_stop = QPushButton("⏹️ 停止")
        self.btn_stop.setFixedHeight(44)
        self.btn_stop.setFixedWidth(85)
        self.btn_stop.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_stop.setEnabled(False)
        self.btn_stop.setToolTip("若辨識過慢，點擊可立即中斷 (Esc)")
        self.btn_stop.setStyleSheet("""
            QPushButton {
                background-color: #CF222E;
                color: #FFFFFF;
                font-size: 14px;
                font-weight: bold;
                border-radius: 8px;
                border: none;
            }
            QPushButton:hover {
                background-color: #A40E26;
            }
            QPushButton:disabled {
                background-color: #E1E4E8;
                color: #8C959F;
            }
        """)
        self.btn_stop.clicked.connect(self.action_stop_ocr)

        action_btn_layout.addWidget(self.btn_recognize, 1)
        action_btn_layout.addWidget(self.btn_stop, 0)
        left_layout.addLayout(action_btn_layout)

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

        # 停止辨識 (Esc)
        stop_sc = QShortcut(QKeySequence("Esc"), self)
        stop_sc.activated.connect(self.action_stop_ocr)

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
        """依據「合併段落」選項重新組織文字呈現（原生支援繁體中文與多語種）"""
        if not self.raw_ocr_items:
            return

        raw_lines = [item[0] for item in self.raw_ocr_items]

        if self.chk_merge_lines.isChecked():
            # 合併為段落（處理中英文空白間隔）
            merged = ""
            for i, line in enumerate(raw_lines):
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
            self.text_edit.setPlainText("\n".join(raw_lines))

    def action_start_ocr(self):
        """觸發 OCR 辨識"""
        if self.current_image is None or self.current_image.isNull():
            QMessageBox.information(self, "提示", "請先上傳圖片或從剪貼簿貼上圖片再進行辨識。")
            return

        # 鎖定辨識按鈕，啟用停止按鈕
        self.btn_recognize.setEnabled(False)
        self.btn_recognize.setText("⏳ 辨識中...")
        self.btn_stop.setEnabled(True)
        self.progress_bar.setVisible(True)
        self.status_bar.showMessage("正在使用 PP-OCRv6 Small 快速辨識中...")

        # 轉換 QImage 為 NumPy 陣列
        img_array = qimage_to_numpy(self.current_image)

        # 啟動背景 Worker 線程（包含防漏字小字增強設定）
        enhance = self.chk_enhance.isChecked()
        self.worker = OCRWorker(img_array, enhance_small_text=enhance)
        self.worker.finished.connect(self.on_ocr_finished)
        self.worker.error.connect(self.on_ocr_error)
        self.worker.start()

    def action_stop_ocr(self):
        """手動強制停止辨識線程"""
        if self.worker and self.worker.isRunning():
            self.worker.terminate()
            self.worker.wait(300)
            self.btn_recognize.setEnabled(True)
            self.btn_recognize.setText("🔍 開始辨識 (Recognize)")
            self.btn_stop.setEnabled(False)
            self.progress_bar.setVisible(False)
            self.status_bar.showMessage("⏹️ 已停止辨識作業", 4000)

    def on_ocr_finished(self, results: list, elapsed: float):
        """OCR 完成回調"""
        self.btn_recognize.setEnabled(True)
        self.btn_recognize.setText("🔍 開始辨識 (Recognize)")
        self.btn_stop.setEnabled(False)
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
        self.btn_stop.setEnabled(False)
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
