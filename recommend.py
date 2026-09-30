"""按机器情况推荐模型。

用户的目标是"老电脑也能流畅跑"。实测数据（Windows，无 GPU，2 线程）：

    模型   内存      RTF     字符错误率
    base   287 MB    0.16    24.65%
    small  701 MB    0.58    20.00%
    medium 1.5 GB    --      20.47%  ← 比 small 还差，别用

base 内存是 small 的 40%，代价是错误率多 4.6 个百分点；medium 体积三倍
却没有任何收益。所以只有两档有意义：内存紧张用 base，否则用 small。

界面上还应该给出占用估计，让用户知道自己在跑什么。
"""

from __future__ import annotations

import ctypes
import os
from dataclasses import dataclass
from ctypes import wintypes
from pathlib import Path

# 实测值（Windows，纯 CPU，含 whisper 推理缓冲，不含 Qt 界面）
MODEL_PROFILE: dict[str, tuple[float, float]] = {
    # 模型名: (推理内存 MB, 2 线程 CPU 上的 RTF)
    "ggml-tiny.bin": (150.0, 0.10),
    "ggml-base.bin": (287.0, 0.16),
    "ggml-small.bin": (701.0, 0.58),
}

UI_OVERHEAD_MB = 95.0   # Qt 界面 + numpy + 依赖的实测值
# 留给系统和其他程序的余量。原来设 900MB 太激进：4GB 的老机器可用内存
# 通常只有 1.2GB 左右，扣掉 900 就只剩 300，base 反而选不上、直接掉到
# tiny，白丢精度。500MB 是实测下来比较合理的线。
RESERVE_MB = 500.0


class _MemStatus(ctypes.Structure):
    _fields_ = [
        ("dwLength", wintypes.DWORD),
        ("dwMemoryLoad", wintypes.DWORD),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]


def available_mb() -> int:
    """当前可用物理内存（MB）。拿不到时返回一个保守值。"""
    if os.name != "nt":
        try:
            info = {}
            for line in Path("/proc/meminfo").read_text().splitlines():
                k, _, v = line.partition(":")
                info[k] = int(v.strip().split()[0]) // 1024
            return int(info.get("MemAvailable", 2048))
        except Exception:
            return 2048
    try:
        m = _MemStatus()
        m.dwLength = ctypes.sizeof(_MemStatus)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)):
            return int(m.ullAvailPhys / 1e6)
    except Exception:
        pass
    return 2048


@dataclass
class Recommendation:
    model: str
    reason: str
    available_mb: int
    estimated_mb: float
    tight: bool


def recommend(preferred: str | None = None,
              have: list[str] | None = None) -> Recommendation:
    """给一个默认模型建议。

    Args:
        preferred: 用户在 config 里指定过的模型，存在且装了就用它，不改。
        have: 本机已下载的模型名列表。
    """
    have = have or []
    avail = available_mb()
    budget = avail - RESERVE_MB

    if preferred and preferred in have:
        mem, _ = MODEL_PROFILE.get(preferred, (0, 0))
        tight = mem > budget
        reason = ("配置里指定的模型" +
                  ("（但当前内存紧张，可能变慢）" if tight else ""))
        return Recommendation(preferred, reason, avail, mem + UI_OVERHEAD_MB, tight)

    if not have:
        return Recommendation("ggml-small.bin", "还没有下载任何模型", avail,
                             MODEL_PROFILE["ggml-small.bin"][0] + UI_OVERHEAD_MB, False)

    # 从精度最高往下降级，选第一个内存装得下的。反过来（从小往大选）会
    # 永远停在 tiny——tiny 总是装得下，等于把精度白白丢掉。
    for name in ("ggml-small.bin", "ggml-base.bin", "ggml-tiny.bin"):
        if name not in have:
            continue
        mem, _ = MODEL_PROFILE[name]
        need = mem + UI_OVERHEAD_MB
        if need <= budget:
            why = "内存够用" if name == "ggml-small.bin" else "内存有限，已降级"
            return Recommendation(name, f"{why}（约 {need:.0f}MB）", avail, need, False)

    # 一个都装不下：给最小的，让用户至少能用，再提示内存紧张
    for name in ("ggml-tiny.bin", "ggml-base.bin"):
        if name in have:
            mem, _ = MODEL_PROFILE[name]
            need = mem + UI_OVERHEAD_MB
            return Recommendation(
                name, f"内存非常紧张（需约 {need:.0f}MB，可用 {avail:.0f}MB），"
                      f"已选最小的模型", avail, need, True)

    mem, _ = MODEL_PROFILE.get(have[0], (300.0, 0.2))
    return Recommendation(have[0], "没有已知配置的模型", avail,
                         mem + UI_OVERHEAD_MB, True)


def describe(model: str) -> str:
    """给界面用的一行说明。"""
    mem, rtf = MODEL_PROFILE.get(model, (0.0, 0.0))
    if not mem:
        return model
    return f"约 {mem:.0f}MB 内存 · 纯 CPU 约 {1 / rtf:.0f} 倍实时" if rtf else f"约 {mem:.0f}MB"
