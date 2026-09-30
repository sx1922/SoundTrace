"""测内存占用，给"老电脑也能跑"这个目标提供数据。

用外部进程读工作集，避免 Python 自己看不到自己。
"""

from __future__ import annotations

import ctypes
import subprocess
import sys
import threading
import time
from ctypes import wintypes
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


class _PMC(ctypes.Structure):
    _fields_ = [
        ("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
        ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
        ("_a", ctypes.c_size_t), ("_b", ctypes.c_size_t),
        ("_c", ctypes.c_size_t), ("_d", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t),
    ]


_ps = ctypes.windll.psapi
_k = ctypes.windll.kernel32
_ps.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(_PMC), wintypes.DWORD]


def rss_mb(pid: int) -> float | None:
    h = _k.OpenProcess(0x1000, False, pid)   # PROCESS_QUERY_LIMITED_INFORMATION
    if not h:
        return None
    c = _PMC()
    c.cb = ctypes.sizeof(c)
    ok = _ps.GetProcessMemoryInfo(h, ctypes.byref(c), c.cb)
    _k.CloseHandle(h)
    return c.WorkingSetSize / 1e6 if ok else None


CHILD = r'''
import os, sys, time
os.environ["QT_QPA_PLATFORM"] = "offscreen"
ROOT = r"{root}"
sys.path.insert(0, ROOT)
import threading
from pathlib import Path
from whisper_bridge import WhisperEngine

stage = Path(ROOT) / ".memstage"
if stage.exists():
    stage.unlink()

def mark(name):
    stage.write_text(name, encoding="utf-8")

mark("imports")
from PySide6.QtWidgets import QApplication
app = QApplication([])
w = None
import app as appmod
w = appmod.MainWindow()
mark("ui")
import numpy as np
e = WhisperEngine(Path(ROOT) / "models" / "{model}", use_gpu={gpu},
                  n_threads=6, vendor_dir=Path(ROOT) / "vendor" / "{vendor}" / "Release")
e.load()
e.silence_logs()
mark("model")
x = np.zeros(16000 * 3, dtype=np.int16)
e.transcribe(x, language="zh")
mark("first_run")
time.sleep(20)
'''


def run(model: str, gpu: bool, vendor: str) -> None:
    stage = ROOT / ".memstage"
    code = CHILD.format(root=ROOT, model=model, gpu=gpu, vendor=vendor)
    p = subprocess.Popen([sys.executable, "-u", "-c", code])
    readings: list[tuple[str, float]] = []
    t0 = time.time()
    while time.time() - t0 < 26:
        time.sleep(0.4)
        m = rss_mb(p.pid)
        name = stage.read_text(encoding="utf-8").strip() if stage.exists() else "?"
        if m and (not readings or readings[-1][0] != name):
            readings.append((name, m))
        if name == "first_run" and time.time() - t0 > 8:
            break
    p.terminate()
    p.wait()
    stage.unlink(missing_ok=True)

    label = {"imports": "导入依赖", "ui": "建主窗口", "model": "加载模型",
             "first_run": "首次识别后"}.copy()
    print(f"\n--- {model} · {'GPU' if gpu else 'CPU'} ---")
    prev = 0.0
    for name, m in readings:
        print(f"  {label.get(name, name):<12} {m:7.1f} MB   ({m - prev:+.1f})")
        prev = m


if __name__ == "__main__":
    for model, gpu, vendor in (("ggml-small.bin", False, "whisper"),
                               ("ggml-base.bin", False, "whisper")):
        run(model, gpu, vendor)
