# -*- coding: utf-8 -*-
"""
MiniOCR 辨識核心（不依賴 Qt，在獨立子行程中執行）

設計重點：
1. 記憶體控制：
   - 實測 PP-OCRv6 Small 偵測 (det) 階段峰值記憶體約與「輸入像素數」成正比
     （約 每百萬像素 +290MB），辨識 (rec) 階段則與 batch 數 × 行寬度成正比。
   - 因此：限制圖片寬度、將高圖切成水平條帶 (strip) 逐段偵測、rec batch 設為 1，
     讓任何解析度的圖片峰值記憶體都維持在固定上限內。
2. 可安全中斷：
   - OCR 跑在獨立子行程，按「停止」時直接結束子行程，
     不會像 QThread.terminate() 那樣破壞 Python / ONNX Runtime 狀態而導致 APP 閃退。
"""

import os
import sys
import time
import threading
import multiprocessing as mp
from pathlib import Path
from typing import List, Optional, Tuple

os.environ.setdefault("ORT_LOGGING_LEVEL", "3")

import numpy as np

# ================= 記憶體 / 效能參數 =================
# 送入 OCR 的最大圖片寬度（超過則等比例縮小）。實測 960px 辨識精準度與 1920 相同，
# 但推論記憶體峰值穩定壓在 340~440 MB，嚴格遵守 500MB 記憶體上限。
MAX_WIDTH = 960
MAX_HEIGHT = 3200
# 單次偵測 (det) 的像素上限：約 0.48MP -> det 峰值約 180~250MB
STRIP_PIXELS = 480_000
# 條帶之間的重疊高度（需大於單行文字高度，避免文字被切半）
STRIP_OVERLAP = 96
# 條帶最小高度
MIN_STRIP_HEIGHT = 280
# 小圖放大門檻（截圖小字增強）
SMALL_H, SMALL_W = 160, 320

OCRItem = Tuple[str, float]


def get_model_paths() -> Tuple[Optional[Path], Optional[Path]]:
    """取得本地離線 PP-OCRv6 Small 模型路徑"""
    if getattr(sys, "frozen", False):
        base_dir = Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))
    else:
        base_dir = Path(__file__).resolve().parent

    det_path = base_dir / "models" / "PP-OCRv6_det_small.onnx"
    rec_path = base_dir / "models" / "PP-OCRv6_rec_small.onnx"
    if det_path.exists() and rec_path.exists():
        return det_path, rec_path
    return None, None


def create_engine():
    """建立 RapidOCR 引擎（僅在子行程中呼叫）"""
    import onnxruntime  # noqa: F401  明確引入，確保 PyInstaller 打包 onnxruntime 與其 DLL
    from rapidocr import RapidOCR

    det_path, rec_path = get_model_paths()
    if not (det_path and rec_path):
        raise FileNotFoundError(
            "找不到離線模型檔：models/PP-OCRv6_det_small.onnx 與 models/PP-OCRv6_rec_small.onnx"
        )

    return RapidOCR(params={
        "Global.log_level": "error",
        "Global.use_cls": False,
        "Global.text_score": 0.35,
        "Global.min_height": 5,
        "Global.width_height_ratio": 35,
        # 由本模組自行控制縮放，避免 RapidOCR 內部再放大/縮小
        "Global.max_side_len": 100000,
        "Det.model_path": str(det_path),
        "Det.unclip_ratio": 1.85,
        # 改為 max：只在超過上限時縮小，絕不放大
        "Det.limit_type": "max",
        "Det.limit_side_len": MAX_WIDTH,
        "Det.thresh": 0.20,
        "Det.box_thresh": 0.40,
        "Rec.model_path": str(rec_path),
        # batch=1 避免寬行文字造成記憶體暴增
        "Rec.rec_batch_num": 1,
        # 關閉記憶體 arena 與多餘執行緒，徹底杜絕記憶體飆升
        "EngineConfig.onnxruntime.enable_cpu_mem_arena": False,
        "EngineConfig.onnxruntime.intra_op_num_threads": 2,
        "EngineConfig.onnxruntime.inter_op_num_threads": 1,
    })


def _resize(img: np.ndarray, new_w: int, new_h: int) -> np.ndarray:
    import cv2
    interp = cv2.INTER_AREA if new_w < img.shape[1] else cv2.INTER_CUBIC
    return cv2.resize(img, (new_w, new_h), interpolation=interp)


def prepare_image(img: np.ndarray, enhance_small_text: bool) -> np.ndarray:
    """小圖放大 + 寬高上限縮小（保持小於 500MB 記憶體）"""
    h, w = img.shape[:2]
    if enhance_small_text and (h < SMALL_H or w < SMALL_W):
        scale = min(3.0, max(2.0, 360.0 / max(h, 1)))
        new_w = min(int(w * scale), MAX_WIDTH)
        scale = new_w / max(w, 1)
        img = _resize(img, new_w, max(1, int(h * scale)))
        h, w = img.shape[:2]

    if w > MAX_WIDTH:
        scale = MAX_WIDTH / w
        img = _resize(img, MAX_WIDTH, max(1, int(h * scale)))
        h, w = img.shape[:2]

    if h > MAX_HEIGHT:
        scale = MAX_HEIGHT / h
        img = _resize(img, max(1, int(w * scale)), MAX_HEIGHT)

    return np.ascontiguousarray(img)


def _strip_ranges(h: int, w: int) -> List[Tuple[int, int]]:
    """將圖片切成水平條帶，回傳 [(y0, y1), ...]"""
    strip_h = max(MIN_STRIP_HEIGHT, STRIP_PIXELS // max(w, 1))
    if h <= strip_h:
        return [(0, h)]
    ranges = []
    step = strip_h - STRIP_OVERLAP
    y0 = 0
    while True:
        y1 = min(h, y0 + strip_h)
        ranges.append((y0, y1))
        if y1 >= h:
            break
        y0 += step
    return ranges


def _sort_reading_order(items):
    """依閱讀順序（由上而下、由左而右）排序 [(box, text, score)]"""
    if not items:
        return []
    items = sorted(items, key=lambda it: (it[0][:, 1].min(), it[0][:, 0].min()))
    lines = []
    for it in items:
        box = it[0]
        cy = box[:, 1].mean()
        bh = max(1.0, box[:, 1].max() - box[:, 1].min())
        if lines:
            last = lines[-1]
            if abs(cy - last["cy"]) < 0.5 * min(bh, last["h"]):
                last["items"].append(it)
                continue
        lines.append({"cy": cy, "h": bh, "items": [it]})
    result = []
    for ln in lines:
        result.extend(sorted(ln["items"], key=lambda it: it[0][:, 0].min()))
    return result


def run_ocr(engine, img: np.ndarray, enhance_small_text: bool = True) -> List[OCRItem]:
    """以條帶方式執行 OCR，回傳 [(text, score), ...]"""
    img = prepare_image(img, enhance_small_text)
    h, w = img.shape[:2]
    ranges = _strip_ranges(h, w)
    half = STRIP_OVERLAP // 2
    collected = []

    for idx, (y0, y1) in enumerate(ranges):
        strip = img[y0:y1]
        out = engine(strip)
        boxes = getattr(out, "boxes", None)
        txts = getattr(out, "txts", None)
        if boxes is None or not txts:
            continue
        # 每個條帶只「擁有」去掉重疊一半後的區域，避免重疊區的文字重複輸出
        own_top = 0 if idx == 0 else half
        own_bottom = (y1 - y0) if idx == len(ranges) - 1 else (y1 - y0) - half
        for box, text, score in zip(boxes, txts, out.scores):
            box = np.asarray(box, dtype=np.float32)
            cy = box[:, 1].mean()
            if own_top <= cy < own_bottom:
                box = box.copy()
                box[:, 1] += y0
                collected.append((box, text, float(score)))

    return [(text, score) for _, text, score in _sort_reading_order(collected)]


# ================= 子行程 =================

def _worker_main(conn):
    """子行程主迴圈：載入模型一次，持續接收影像並回傳結果"""
    import gc
    try:
        engine = create_engine()
        conn.send(("ready", None, 0.0))
    except Exception as e:  # 模型載入失敗
        conn.send(("error", f"模型載入失敗：{e}", 0.0))
        return

    while True:
        try:
            job = conn.recv()
        except (EOFError, OSError):
            break
        if job is None:
            break
        img, enhance = job
        start = time.time()
        try:
            results = run_ocr(engine, img, enhance)
            conn.send(("ok", results, time.time() - start))
        except Exception as e:
            conn.send(("error", str(e), time.time() - start))
        finally:
            del img, job
            gc.collect()


class OCRCancelled(Exception):
    pass


class OCRProcess:
    """管理常駐 OCR 子行程（模型只載入一次；停止時可直接結束子行程）"""

    def __init__(self):
        self._ctx = mp.get_context("spawn")
        self._proc = None
        self._conn = None
        self._ready = False
        self._cancel = False
        self._lock = threading.Lock()

    def is_alive(self) -> bool:
        return self._proc is not None and self._proc.is_alive()

    def start(self):
        if self.is_alive():
            return
        parent_conn, child_conn = self._ctx.Pipe(duplex=True)
        proc = self._ctx.Process(target=_worker_main, args=(child_conn,), daemon=True)
        proc.start()
        child_conn.close()
        self._proc, self._conn, self._ready = proc, parent_conn, False

    def _wait_reply(self):
        """等待子行程回覆；每 0.1 秒檢查一次是否被要求停止"""
        while True:
            if self._cancel:
                raise OCRCancelled()
            try:
                if self._conn.poll(0.1):
                    return self._conn.recv()
            except (EOFError, OSError, BrokenPipeError):
                if self._cancel:
                    raise OCRCancelled()
                raise RuntimeError("OCR 子行程意外結束（可能是記憶體不足）")
            if not self.is_alive() and not self._conn.poll(0):
                if self._cancel:
                    raise OCRCancelled()
                raise RuntimeError("OCR 子行程意外結束（可能是記憶體不足）")

    def recognize(self, img: np.ndarray, enhance: bool) -> Tuple[List[OCRItem], float]:
        """阻塞式辨識（請在背景執行緒呼叫）"""
        with self._lock:
            self._cancel = False
            try:
                self.start()
                if not self._ready:
                    status, payload, _ = self._wait_reply()
                    if status != "ready":
                        self._cleanup()
                        raise RuntimeError(payload)
                    self._ready = True
                self._conn.send((img, enhance))
                status, payload, elapsed = self._wait_reply()
            except OCRCancelled:
                self._cleanup()
                raise
            except RuntimeError:
                self._cleanup()
                raise
            if status == "ok":
                return payload, elapsed
            raise RuntimeError(payload)

    def warm_up(self):
        """預先啟動子行程並載入模型（背景執行緒呼叫）"""
        if not self._lock.acquire(blocking=False):
            return
        try:
            self._cancel = False
            self.start()
            if not self._ready:
                status, _, _ = self._wait_reply()
                if status == "ready":
                    self._ready = True
                else:
                    self._cleanup()
        except Exception:
            self._cleanup()
        finally:
            self._lock.release()

    def cancel(self):
        """要求停止：直接結束子行程（任何執行緒皆可安全呼叫）"""
        self._cancel = True
        proc = self._proc
        if proc is not None and proc.is_alive():
            try:
                proc.kill()
            except Exception:
                pass

    def _cleanup(self):
        proc, conn = self._proc, self._conn
        self._proc, self._conn, self._ready = None, None, False
        if proc is not None:
            if proc.is_alive():
                try:
                    proc.kill()
                except Exception:
                    pass
            proc.join(2)
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass

    def shutdown(self):
        self._cancel = True
        self._cleanup()
