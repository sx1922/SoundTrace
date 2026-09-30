"""在进程内读工作集内存。给 bench_memory 用。"""

from __future__ import annotations

import ctypes
from ctypes import wintypes


class PMC(ctypes.Structure):
    _fields_ = [
        ("cb", wintypes.DWORD),
        ("PageFaultCount", wintypes.DWORD),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("_a", ctypes.c_size_t), ("_b", ctypes.c_size_t),
        ("_c", ctypes.c_size_t), ("_d", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t),
    ]


def rss_mb() -> float:
    """当前进程工作集，单位 MB。"""
    ps = ctypes.windll.psapi
    k = ctypes.windll.kernel32
    ps.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(PMC), wintypes.DWORD]
    c = PMC()
    c.cb = ctypes.sizeof(c)
    h = k.GetCurrentProcess()
    ok = ps.GetProcessMemoryInfo(h, ctypes.byref(c), c.cb)
    return c.WorkingSetSize / 1e6 if ok else 0.0
