"""推理设备探测。

关键点：光看有没有 N 卡是不够的。whisper.cpp 的 CPU 版和 CUDA 版是两套
DLL，同一个 vendor 目录里只能放一套。所以这里要判断的是"CUDA 构建是否已
就位"，而不是"显卡是否存在"。

CUDA 构建（tools/fetch_assets.py --cuda）会把 DLL 解到 vendor/whisper_cuda，
两者共存，运行时按配置选目录。
"""

from __future__ import annotations

import ctypes
import os
import platforms
import subprocess
from dataclasses import dataclass
from pathlib import Path

ROOT = platforms.app_root()
CPU_DIR = ROOT / "vendor" / "whisper" / "Release"
CUDA_DIR = ROOT / "vendor" / "whisper_cuda" / "Release"


@dataclass
class DeviceInfo:
    name: str          # "cuda" / "cpu"
    vendor_dir: Path
    use_gpu: bool
    detail: str        # 给界面看的一句话说明

    @property
    def label(self) -> str:
        return "GPU 加速" if self.use_gpu else "CPU"


def has_cuda_build() -> bool:
    """CUDA 构建是否已经下载解压过。"""
    plat = platforms.current()
    if not (CUDA_DIR / plat.whisper_path_name).is_file():
        return False
    pat = f"*{plat.cuda_backend}.{plat.lib_ext}"
    return bool(list(CUDA_DIR.glob(pat))) or bool(list(CUDA_DIR.glob("cudart*")))


def nvidia_present() -> bool:
    if os.name == "nt":
        pass
    elif os.name == "posix":
        return _nvidia_posix()
    else:
        return False
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
        )
        return out.returncode == 0 and out.stdout.strip() != ""
    except Exception:
        return False


def _nvidia_posix() -> bool:
    import shutil
    return shutil.which("nvidia-smi") is not None


def _cudart_dlls_present(d: Path) -> bool:
    """CUDA 运行时 DLL 是否随包提供。

    whisper.cpp 的 cublas 包不含 cudart，需要机器上装 CUDA Toolkit，
    除非另外下载了 cudart-llama 包。没有它 CUDA 构建照样起不来。
    """
    plat = platforms.current()
    ext = plat.lib_ext
    if any(d.glob(f"cudart*.{ext}")) or any(d.glob(f"cublas*.{ext}")):
        return True
    if plat.key != "windows":
        # macOS 上 Metal 是系统自带的，Linux 上 CUDA 一般装在系统路径里
        return plat.key == "macos"
    # Windows：可能装过 CUDA Toolkit，DLL 在系统 PATH 里
    import ctypes
    for name in ("cudart64_12.dll", "cublas64_12.dll"):
        try:
            ctypes.WinDLL(name)
            return True
        except OSError:
            continue
    return False


def detect(preference: str = "auto") -> DeviceInfo:
    """按配置决定用 CPU 还是 CUDA。

    Args:
        preference: "auto" / "cpu" / "gpu"
    """
    if preference == "cpu":
        return DeviceInfo("cpu", CPU_DIR, False, "强制使用 CPU")

    if not has_cuda_build():
        detail = "未安装 CUDA 构建"
        if nvidia_present():
            detail += "（检测到 N 卡，可运行 tools/fetch_assets.py --cuda 启用）"
        else:
            detail += "（未检测到 N 卡）"
        return DeviceInfo("cpu", CPU_DIR, False, detail)

    if not _cudart_dlls_present(CUDA_DIR):
        return DeviceInfo("cpu", CPU_DIR, False,
                          "CUDA 构建缺少 CUDA 运行时，已回退 CPU")

    if not nvidia_present():
        return DeviceInfo("cpu", CPU_DIR, False,
                          "安装了 CUDA 构建但没检测到 N 卡，已回退 CPU")

    return DeviceInfo("cuda", CUDA_DIR, True, "CUDA 加速")


def default_threads() -> int:
    n = os.cpu_count() or 4
    return max(1, n - 1)


def cpu_hint() -> str:
    """给界面显示的 CPU 名字。"""
    if os.name == "nt":
        try:
            out = subprocess.run(
                ["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor).Name"],
                capture_output=True, text=True, timeout=8,
            )
            if out.returncode == 0 and out.stdout.strip():
                return out.stdout.strip().splitlines()[0].strip()
        except Exception:
            pass
    return platform_processor()


def platform_processor() -> str:
    import platform

    return platform.processor() or platform.machine()
