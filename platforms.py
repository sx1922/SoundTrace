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


# 运行时按"族"分目录：CPU 主包在 vendor/whisper/，CUDA 包在 vendor/whisper_cuda/。
# device.py 的 CPU_DIR / CUDA_DIR、assetdl 与 fetch_assets 的检查都围绕这两族。
RUNTIME_FAMILIES = ("whisper", "whisper_cuda")


def _split_vendor(path: Path) -> tuple[Path, str]:
    """把调用方给的任意一层路径归一化成 (vendor 根目录, 运行时族名)。

    历史上有多种形态都在传，不能只认一种：
      vendor/                        -> (vendor, "whisper")
      vendor/whisper/                -> (vendor, "whisper")
      vendor/whisper/Release         -> (vendor, "whisper")
      vendor/whisper_cuda/Release    -> (vendor, "whisper_cuda")

    4b33c58 的 _as_vendor_root 只认 whisper/Release 这一种形态，CUDA 目录
    原样通过后又被拼上 whisper/Release，得到
    vendor/whisper_cuda/Release/whisper/Release/whisper.dll —— 装了 CUDA
    构建、device=auto 的机器（GPU 用户的标准形态）模型加载必然失败。
    """
    p = path
    family = "whisper"
    if p.name == "Release":
        p = p.parent
    if p.name in RUNTIME_FAMILIES:
        family = p.name
        p = p.parent
    return p, family


def _find_macos_framework_lib(family_dir: Path, lib_name: str) -> Path | None:
    """在 macOS xcframework 解压目录中查找对应架构的库。

    family_dir 是 vendor/whisper 这一层（xcframework 解压在它下面）：
      whisper.xcframework/
        macos-arm64/whisper.framework/whisper
        macos-x86_64/whisper.framework/whisper

    返回找到的 dylib 路径，找不到返回 None。未经真机验证。
    """
    import platform as _platform
    arch = _platform.machine().lower()
    # 归一化架构名；认不出的架构两个都试
    if arch in ("arm64", "aarch64"):
        archs = ("macos-arm64", "macos-x86_64")
    elif arch in ("x86_64", "amd64"):
        archs = ("macos-x86_64", "macos-arm64")
    else:
        archs = ("macos-arm64", "macos-x86_64")

    xcf = family_dir / f"{lib_name}.xcframework"
    if not xcf.is_dir():
        return None
    for arch_dir in archs:
        fw = xcf / arch_dir / f"{lib_name}.framework" / lib_name
        if fw.is_file():
            return fw
    return None


def is_runtime_present(vendor_root: Path, family: str | None = None) -> bool:
    """检查 whisper.cpp 运行时是否已就绪（跨平台）。

    family 不传时从路径推断（如 vendor/whisper_cuda/Release -> "whisper_cuda"），
    显式传入可覆盖推断。
    """
    root, inferred = _split_vendor(vendor_root)
    fam = family or inferred
    plat = current()
    main = plat.whisper_path_name
    if plat.key == "windows":
        return (root / fam / "Release" / main).is_file()
    if plat.key == "macos":
        if fam != "whisper":
            return False    # Metal 随主包，没有单独的 CUDA 族
        lib = _find_macos_framework_lib(root / fam, "whisper")
        return lib is not None and lib.is_file()
    if plat.key == "linux":
        # Linux 通常编译后放在 vendor/whisper/Release/ 或 vendor/whisper/
        return (root / fam / "Release" / main).is_file() or \
               (root / fam / main).is_file()
    return False


def get_library_path(vendor_root: Path, lib_name: str = "whisper") -> Path | None:
    """获取动态库的实际路径（跨平台）。

    lib_name 是逻辑名（whisper / ggml / ggml-cuda / ggml-cpu-haswell），
    Windows 下文件名即 <lib_name>.dll。运行时族从 vendor_root 推断：
    device.CUDA_DIR（vendor/whisper_cuda/Release）解析出的是 CUDA 目录
    下的库，而不是被错误拼上 whisper/Release。

    用于 whisper_bridge.py 等需要精确路径加载库的场景。
    """
    root, family = _split_vendor(vendor_root)
    plat = current()
    if plat.key == "windows":
        return root / family / "Release" / f"{lib_name}.{plat.lib_ext}"
    if plat.key == "macos":
        fam_dir = root / family
        # 官方 release 形态：主库在 xcframework 的 .framework 里
        p = _find_macos_framework_lib(fam_dir, lib_name)
        if p is not None:
            return p
        # 自编译 / ggml 后端形态：平铺 dylib，带或不带 lib 前缀
        for base in (fam_dir, fam_dir / "Release"):
            for stem in (f"lib{lib_name}", lib_name):
                cand = base / f"{stem}.{plat.lib_ext}"
                if cand.is_file():
                    return cand
        return None
    if plat.key == "linux":
        for base in (root / family / "Release", root / family):
            for stem in (lib_name, f"lib{lib_name}"):
                cand = base / f"{stem}.{plat.lib_ext}"
                if cand.is_file():
                    return cand
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
