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

# 实测值（Windows，纯 CPU，2 线程，53 秒音频，取加载+推理后的 WorkingSet 净增）
# 测量方法见 tools/bench_memory.py。
# 注意 RTF 与音频长度强相关：whisper 的编码器恒定按 30 秒窗口算，短音频的
# RTF 会被这项固定开销主导而虚高（6 秒音频上 small 是 1.05，53 秒上是 0.30）。
# 所以表里的 RTF 只对"长音频"有意义，别拿它预测单句延迟。
MODEL_PROFILE: dict[str, tuple[float, float]] = {
    # 模型名: (推理内存 MB, 2 线程 CPU 上的 RTF)
    "ggml-tiny.bin": (150.0, 0.10),          # 本机未装，按比例推算
    "ggml-base.bin": (283.0, 0.08),
    "ggml-small.bin": (677.0, 0.30),
    "ggml-medium.bin": (1816.0, 0.90),
    "ggml-large-v3-turbo.bin": (1783.0, 1.31),
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
    """当前可用物理内存（MB）。拿不到时返回一个保守值。

    单位统一用十进制 MB（1e6 字节），和 MODEL_PROFILE 里的实测值一致。
    /proc/meminfo 给的是 KiB，直接 //1024 得到的是 MiB，和 Windows 分支的
    1e6 口径差了约 4.9%——两个分支喂给同一个 budget 预算，不该有偏差。
    """
    if os.name != "nt":
        try:
            info = {}
            for line in Path("/proc/meminfo").read_text().splitlines():
                k, _, v = line.partition(":")
                vals = v.strip().split()
                if not vals:
                    continue
                info[k] = int(vals[0]) * 1024 / 1e6     # KiB -> MB
            return int(info.get("MemAvailable", 2048))
        except Exception:
            return 2048
    try:
        m = _MemStatus()
        m.dwLength = ctypes.sizeof(_MemStatus)
        ctypes.windll.kernel32.GlobalMemoryStatusEx.restype = ctypes.c_int
        ctypes.windll.kernel32.GlobalMemoryStatusEx.argtypes = [ctypes.c_void_p]
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


def _weight_mb(name: str) -> float:
    """权重文件本身的体积（MB），作为没有实测值时的下限估计。

    ggml 系列的推理内存以权重为主，所以文件大小是个像样的下限。
    原来这里直接取 0，于是任何没进 MODEL_PROFILE 的模型都被报成
    "约 95MB"——界面会给一个 1.5GB 的模型显示"内存占用 95MB"。
    """
    try:
        p = Path(__file__).resolve().parent / "models" / name
        return p.stat().st_size / 1e6
    except OSError:
        return 0.0


def _mem_for(name: str) -> tuple[float, bool]:
    """返回 (内存 MB, 是否有实测值)。"""
    if name in MODEL_PROFILE:
        return MODEL_PROFILE[name][0], True
    return _weight_mb(name), False


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
        mem, known = _mem_for(preferred)
        need = mem + UI_OVERHEAD_MB
        # tight 要拿"总需求"去比预算。只比 mem 会漏掉界面自身的开销——
        # 原来 mem=701 恰好比 budget=700 大，蒙对了；换成实测的 677 就再也
        # 不会触发，内存明明紧张却不提示。
        tight = need > budget
        reason = "配置里指定的模型"
        if not known:
            # 没实测过就只说文件多大，别把猜测讲成实测
            reason += f"（未实测占用，权重文件 {mem:.0f}MB）"
        elif tight:
            reason += "（但当前内存紧张，可能变慢）"
        return Recommendation(preferred, reason, avail, need, tight)

    if not have:
        return Recommendation("ggml-small.bin", "还没有下载任何模型", avail,
                             MODEL_PROFILE["ggml-small.bin"][0] + UI_OVERHEAD_MB, False)

    # 从精度最高往下降级，选第一个内存装得下的。反过来（从小往大选）会
    # 永远停在 tiny——tiny 总是装得下，等于把精度白白丢掉。
    # 降级链里含 medium/large：实测它们在 2 线程 CPU 上 RTF 分别是 0.90
    # 和 1.31，后者已经慢于实时，放进去只会让用户等。
    for name in ("ggml-small.bin", "ggml-base.bin", "ggml-tiny.bin"):
        if name not in have:
            continue
        mem, _ = MODEL_PROFILE[name]
        need = mem + UI_OVERHEAD_MB
        if need <= budget:
            why = "内存够用" if name == "ggml-small.bin" else "内存有限，已降级"
            return Recommendation(name, f"{why}（约 {need:.0f}MB）", avail, need, False)

    # 一个都装不下：给权重文件最小的那个，让用户至少能用，再提示内存紧张。
    # 不能只在小模型里找——用户可能只装了 medium/large，降级链够不着它们。
    best = None
    for name in have:
        mem, _ = _mem_for(name)
        if best is None or mem < best[1]:
            best = (name, mem)
    if best is not None:
        name, mem = best
        need = mem + UI_OVERHEAD_MB
        return Recommendation(
            name, f"内存非常紧张（需约 {need:.0f}MB，可用 {avail:.0f}MB），"
                  f"已选占用最小的已装模型", avail, need, True)

    return Recommendation("ggml-small.bin", "没有可用模型", avail,
                         MODEL_PROFILE["ggml-small.bin"][0] + UI_OVERHEAD_MB, True)


def describe(model: str) -> str:
    """给界面用的一行说明。"""
    mem, rtf = MODEL_PROFILE.get(model, (0.0, 0.0))
    if mem:
        if not rtf:
            return f"约 {mem:.0f}MB 内存"
        # RTF = 推理耗时 / 音频时长。0.08 表示只用 8% 的时间，是快于实时；
        # 大于 1 才是慢于实时。写反了会正好把最该提醒的模型说成最快。
        if rtf > 1.0:
            return (f"约 {mem:.0f}MB 内存 · 纯 CPU 2 线程慢于实时"
                    f"（10 秒音频约需 {rtf * 10:.0f} 秒）")
        return f"约 {mem:.0f}MB 内存 · 纯 CPU 2 线程约 {1 / rtf:.0f} 倍实时"
    # 没实测过的模型只报文件大小，别把估算讲成实测
    w = _weight_mb(model)
    return f"权重 {w:.0f}MB · 占用未实测" if w else model
