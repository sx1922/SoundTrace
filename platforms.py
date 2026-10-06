"""平台差异集中在这里。

本项目只在 Windows 上完整验证过。macOS / Linux 的路径是按 whisper.cpp
官方 release 的资产命名推出来的，逻辑上成立但没有实机跑过——用之前请
先在目标系统上执行 `python tools/selfcheck.py`。

差异有三处：
1. 动态库后缀：.dll / .dylib / .so
2. 加载方式：Windows 要 os.add_dll_directory 挂搜索路径，
   macOS/Linux 用 dlopen 且依赖 @loader_path 一类的相对写法
3. 官方 release 只提供 Windows 和 macOS 的预编译包，Linux 要自己编译
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Platform:
    key: str            # windows / macos / linux
    lib_ext: str        # 动态库后缀，不含点
    main_lib: str       # whisper 主库文件名
    ggml_lib: str       # ggml 加载器文件名
    cpu_backend: str    # x86-64 CPU 后端文件名
    cuda_backend: str   # CUDA 后端文件名（没有则为空串）
    official_builds: bool   # 官方是否提供该平台的预编译包
    build_hint: str

    @property
    def whisper_path_name(self) -> str:
        return f"{self.main_lib}.{self.lib_ext}"


WINDOWS = Platform(
    key="windows",
    lib_ext="dll",
    main_lib="whisper",
    ggml_lib="ggml",
    cpu_backend="ggml-cpu-haswell",
    cuda_backend="ggml-cuda",
    official_builds=True,
    build_hint="官方提供 whisper-bin-x64.zip（含 CUDA 版）",
)

# Apple Silicon / Intel 通用：release 里是 xcframework 打包，
# 实际会解出 .framework 或 .dylib，文件名按最常见的形态取
# 典型结构：
#   whisper-b5130-xcframework.zip
#   └── whisper.xcframework/
#       ├── macos-arm64/
#       │   └── whisper.framework/
#       │       └── whisper
#       └── macos-x86_64/
#           └── whisper.framework/
#           └── whisper
# 实际加载时需要找到对应架构的 .framework 内的 dylib
MACOS = Platform(
    key="macos",
    lib_ext="dylib",
    main_lib="libwhisper",
    ggml_lib="libggml",
    cpu_backend="ggml-cpu",
    cuda_backend="ggml-metal",
    official_builds=True,
    build_hint="官方提供 whisper-bXXXX-xcframework.zip（Universal）",
)

LINUX = Platform(
    key="linux",
    lib_ext="so",
    main_lib="libwhisper",
    ggml_lib="libggml",
    cpu_backend="ggml-cpu",
    cuda_backend="ggml-cuda",
    official_builds=False,
    build_hint="官方不提供 Linux 预编译包，需要用 CMake 自行构建",
)


def _find_macos_framework_lib(vendor_dir: Path, lib_name: str) -> Path | None:
    """在 macOS xcframework 解压目录中查找对应架构的 dylib。

    xcframework 结构通常为：
      vendor/whisper/whisper.xcframework/
        macos-arm64/whisper.framework/whisper
        macos-x86_64/whisper.framework/whisper

    返回找到的 dylib 路径，找不到返回 None。
    """
    import platform as _platform
    arch = _platform.machine().lower()
    # 归一化架构名
    if arch in ("arm64", "aarch64"):
        arch_dir = "macos-arm64"
    elif arch in ("x86_64", "amd64"):
        arch_dir = "macos-x86_64"
    else:
        # 兜底：尝试两个都找找
        for candidate in ("macos-arm64", "macos-x86_64"):
            fw = vendor_dir / "whisper" / "whisper.xcframework" / candidate / f"{lib_name}.framework" / lib_name
            if fw.is_file():
                return fw
        return None

    fw = vendor_dir / "whisper" / "whisper.xcframework" / arch_dir / f"{lib_name}.framework" / lib_name
    if fw.is_file():
        return fw

    # 兜底：另一个架构
    other = "macos-x86_64" if arch_dir == "macos-arm64" else "macos-arm64"
    fw = vendor_dir / "whisper" / "whisper.xcframework" / other / f"{lib_name}.framework" / lib_name
    if fw.is_file():
        return fw

    return None


def is_runtime_present(vendor_root: Path) -> bool:
    """检查 whisper.cpp 运行时是否已就绪（跨平台）。"""
    vendor_root = _as_vendor_root(vendor_root)
    plat = current()
    if plat.key == "windows":
        return (vendor_root / "whisper" / "Release" / plat.whisper_path_name).is_file()
    if plat.key == "macos":
        # macOS 用 xcframework，查找 .framework 内的 dylib
        main_lib = _find_macos_framework_lib(vendor_root, "whisper")
        return main_lib is not None and main_lib.is_file()
    if plat.key == "linux":
        # Linux 通常编译后放在 vendor/whisper/Release/ 或 vendor/whisper/
        return (vendor_root / "whisper" / "Release" / plat.whisper_path_name).is_file() or \
               (vendor_root / "whisper" / plat.whisper_path_name).is_file()
    return False


def _as_vendor_root(vendor_root: Path) -> Path:
    """归一化入参：既接受 vendor 根目录，也接受已经指到 vendor/whisper/Release 的路径。

    whisper_bridge.py 的 VENDOR_DIR 是 vendor/whisper/Release（DLL 搜索路径
    必须指向 dll 所在目录），而 is_runtime_present 的调用方传的是 vendor/。
    这里统一退回 vendor 根，下面的拼接逻辑就不用区分谁传的。
    """
    if vendor_root.name == "Release" and vendor_root.parent.name == "whisper":
        return vendor_root.parent.parent
    return vendor_root


def get_library_path(vendor_root: Path, lib_name: str = "whisper") -> Path | None:
    """获取动态库的实际路径（跨平台）。

    用于 whisper_bridge.py 等需要精确路径加载库的场景。
    """
    vendor_root = _as_vendor_root(vendor_root)
    plat = current()
    if plat.key == "windows":
        return vendor_root / "whisper" / "Release" / f"{lib_name}.{plat.lib_ext}"
    if plat.key == "macos":
        return _find_macos_framework_lib(vendor_root, lib_name)
    if plat.key == "linux":
        p = vendor_root / "whisper" / "Release" / f"{lib_name}.{plat.lib_ext}"
        if p.is_file():
            return p
        p = vendor_root / "whisper" / f"{lib_name}.{plat.lib_ext}"
        if p.is_file():
            return p
        return None
    return None


def app_root() -> Path:
    """程序可写根目录：源码运行是项目根，打包后是 exe 所在目录。

    打包成 PyInstaller one-folder 时，__file__ 会落在 <exe同级>/_internal/，
    直接用 Path(__file__).parent 会把 config.json、models/、vendor/
    全部指到 _internal 里去——配置写不对位置、模型找不到。
    所以统一走这里。
    """
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


def is_frozen() -> bool:
    return bool(getattr(sys, "frozen", False))


def current() -> Platform:
    if os.name == "nt":
        return WINDOWS
    if sys.platform == "darwin":
        return MACOS
    return LINUX


def is_supported() -> bool:
    """明确说清楚哪些平台是"能跑"、哪些只是"代码路径存在"。

    Linux 目前会直接拒绝：官方没有预编译包，而让用户自己编译
    whisper.cpp 需要 CMake + 编译器，超出了"下载即用"的范围。
    与其给一个必然失败的引导，不如直说。
    """
    return current().key in ("windows", "macos")


# -- 加载 ------------------------------------------------------------------


def load_library(path: Path):
    """按当前平台加载动态库，返回 ctypes 的库对象。

    macOS 的 .framework 需要直接加载 framework 内的 dylib 文件，
    而不是加载 .framework 目录本身。
    """
    import ctypes

    p = current()
    if p.key == "windows":
        return ctypes.WinDLL(str(path))
    # macOS 和 Linux 都用 CDLL，但 macOS 传入的是 .framework 内的 dylib 文件路径
    return ctypes.CDLL(str(path))


def add_vendor_to_search_path(vendor_dir: Path):
    """让动态链接器能在 vendor 目录里找到依赖。

    Windows 必须显式挂路径，否则 ggml-cuda.dll 找不到同目录的
    cublas/cudart。macOS/Linux 用的是 RPATH/ldconfig，代码上不需要
    做什么，返回 None 表示无需处理。
    """
    if current().key == "windows" and hasattr(os, "add_dll_directory"):
        return os.add_dll_directory(str(vendor_dir))
    return None


def release_asset_name(p: Platform, want_cuda: bool) -> str | None:
    """对应的官方 release 资产名。Linux 没有预编译包，返回 None。"""
    if p.key == "windows":
        return "whisper-cublas-12.4.0-bin-x64.zip" if want_cuda else "whisper-bin-x64.zip"
    if p.key == "macos":
        return None  # 资产名带 build 号（whisper-b5130-xcframework.zip），由调用方拼
    return None
