#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""下载 whisper.cpp 运行时和 GGML 模型。

用法:
    python tools/fetch_assets.py              # 仅下载 CPU 运行时和默认模型
    python tools/fetch_assets.py --cuda       # 额外下载 CUDA/Metal 运行时
    python tools/fetch_assets.py --model base # 指定模型
    python tools/fetch_assets.py --all        # 下载全部模型
"""

from __future__ import annotations

import argparse
import sys
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

# 这段代码必须在 import 前执行，解决 GitHub Actions Windows runner 的 cp1252 编码问题
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import platforms  # noqa: E402
import device  # noqa: E402

VENDOR = ROOT / "vendor"
MODELS = ROOT / "models"

# 挑一个带预编译包的 tag。b5454 的资产名较 b5130 有变：
# whisper-cublas-11.8.0-bin-x64.zip -> whisper-bin-win-cuda-11.8.0-x64.zip
RELEASE_TAG = "b5454"

# 官方 release 的资产名按平台不同。Linux 没有预编译包，需要自行编译。
# CUDA 按变体选：老卡 + 老驱动跑不动 CUDA 12.4 运行时，11.8 可以——
# 选哪个由 device.preferred_cuda_variant() 按算力和驱动定。
_ASSETS = {
    "windows": {
        "cpu": "whisper-bin-x64.zip",
        "cuda": {
            "12.4": "whisper-bin-win-cuda-12.4.0-x64.zip",
            "11.8": "whisper-bin-win-cuda-11.8.0-x64.zip",
        },
        # 不声明独立 cudart 资产：whisper.cpp 的 release 里没有这个包，
        # CUDA 包已内置 cudart。就绪核验在 fetch_runtime 里做。
        "dest": "whisper",
    },
    "macos": {
        # 资产名带 build 号，和 RELEASE_TAG 一致
        "cpu": f"whisper-{RELEASE_TAG}-xcframework.zip",
        "cuda": None,          # macOS 走 Metal，随主包一起
        "cudart": None,
        "dest": "whisper",
    },
    "linux": {"cpu": None, "cuda": None, "cudart": None, "dest": "whisper"},
}

BASE = f"https://github.com/ggml-org/whisper.cpp/releases/download/{RELEASE_TAG}"
HF = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main"
# 主站部分地区直连不通，失败时切这条镜像再试（路径结构一致）
HF_MIRROR = "https://hf-mirror.com/ggerganov/whisper.cpp/resolve/main"

MODELS_AVAILABLE = {
    "tiny": "ggml-tiny.bin",
    "base": "ggml-base.bin",
    "small": "ggml-small.bin",
    "medium": "ggml-medium.bin",
    "large-v3-turbo": "ggml-large-v3-turbo.bin",
}
DEFAULT_MODEL = "small"


def human(n: int) -> str:
    return f"{n / 1e6:.1f}MB" if n < 1e9 else f"{n / 1e9:.2f}GB"


def download(url: str, dst: Path, desc: str = "") -> bool:
    """带进度显示的下载。已完整存在则跳过。"""
    if dst.is_file():
        print(f"  已存在 {dst.name} ({human(dst.stat().st_size)})，跳过")
        return True
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".part")
    label = desc or dst.name
    print(f"  下载 {label} …")
    try:
        with urllib.request.urlopen(url, timeout=60) as r:
            total = int(r.headers.get("Content-Length") or 0)
            done = 0
            last = 0
            with open(tmp, "wb") as f:
                while True:
                    block = r.read(1 << 20)
                    if not block:
                        break
                    f.write(block)
                    done += len(block)
                    if total and done - last > total // 20:
                        last = done
                        pct = done * 100 / total
                        print(f"\r    {pct:5.1f}%  {human(done)} / {human(total)}",
                              end="", flush=True)
        print(f"\r    完成 {human(done)}{' ' * 24}")
        tmp.replace(dst)
        return True
    except Exception as e:
        print(f"\n  下载失败: {e}")
        print(f"  可以手动下载后放到: {dst}")
        if tmp.is_file():
            tmp.unlink()
        return False


def _extract_macos_xcframework(zip_path: Path, dest: Path) -> None:
    """解压 macOS xcframework 包。

    xcframework 结构通常为：
      whisper-{tag}-xcframework.zip
      └── whisper.xcframework/
          ├── macos-arm64/
          │   └── whisper.framework/
          │       └── whisper
          └── macos-x86_64/
              └── whisper.framework/
                  └── whisper

    我们需要把它解压到 dest/whisper.xcframework/，让结构自然落位。
    """
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(dest)


def _extract_linux(zip_path: Path, dest: Path) -> None:
    """解压 Linux 自编译包（如果将来有官方包的话）。

    暂时保持和 Windows 一样的解压逻辑。
    """
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(dest)


def _extract(zip_path: Path, dest: Path) -> None:
    """根据平台分发解压逻辑。

    Windows/Linux: 直接解压到 dest，让 "Release/" 这一层自然落在正确位置。
    macOS: 解压 xcframework 到 dest，结构为 whisper.xcframework/...。
    """
    plat = platforms.current()
    if plat.key == "macos":
        _extract_macos_xcframework(zip_path, dest)
    elif plat.key == "linux":
        _extract_linux(zip_path, dest)
    else:
        dest.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(dest)


def _require(path: Path, what: str) -> None:
    """解压后确认关键文件真的落地了。

    之前这里无条件打印"已安装"，结果解压到了错误的目录，
    后面每一步都找不到运行时，排查起来很费时间。
    """
    if not path.is_file():
        raise SystemExit(f"{what} 解压异常：预期位置没有 {path.name}（{path}）")
    print(f"  已安装到 {path.parent}")


def fetch_runtime(cuda: bool, cuda_variant: str | None = None) -> bool:
    """返回 True 表示 CPU 运行时确实就位。

    运行时是硬依赖，缺了后面一步都跑不了，所以失败必须中断，
    不能像之前那样打完"完成"字样再让用户去 app.py 里撞报错。
    """
    plat = platforms.current()
    # 资产名都从这张表取。4b33c58 重写本文件时把定义弄丢了，A 成了
    # 未定义名字：新机器首次下载和 --cuda 分支必崩 NameError，
    # 只有"一切都已装好"的机器碰巧不触发。
    A = _ASSETS[plat.key]
    print(f"\n[1/2] whisper.cpp 运行时 ({plat.key})")
    dest = VENDOR / "whisper"

    # 跨平台检查是否已就绪
    if platforms.is_runtime_present(VENDOR):
        print(f"  CPU 运行时已就绪 ({plat.key})")
    elif download(f"{BASE}/{A['cpu']}", VENDOR / "whisper-bin.zip", "CPU 运行时"):
        print("  解压中 …")
        _extract(VENDOR / "whisper-bin.zip", dest)
        (VENDOR / "whisper-bin.zip").unlink(missing_ok=True)
        # 跨平台验证
        if not platforms.is_runtime_present(VENDOR):
            raise SystemExit(f"CPU 运行时解压异常：预期位置没有找到库文件 ({dest})")
        print(f"  CPU 运行时已安装到 {dest}")
    else:
        return False

    if not cuda:
        return True

    print(f"\n[2/3] CUDA/Metal 运行时 ({plat.key})")
    cuda_dest = VENDOR / "whisper_cuda"

    cuda_variants = A.get("cuda")
    if not cuda_variants:
        print(f"  当前平台 {plat.key} 无需单独 CUDA 运行时（{plat.build_hint}）")
        return True

    # 变体按算力 + 驱动选：老卡/老驱动给 11.8，其余 12.4；都不合适就跳过。
    # --cuda-variant 可强制指定（排查问题时用）。
    variant = cuda_variant or device.preferred_cuda_variant()
    if not variant:
        print("  未检测到可用的 CUDA 环境（无 N 卡 / 算力过低 / 缺 nvidia-smi）。")
        print("  跳过 CUDA 构建——核显与老显卡走 CPU 路径即可正常使用。")
        return True
    cuda_asset = cuda_variants.get(variant)
    print(f"  选定 CUDA 构建 {variant}（按算力/驱动自动选择）")

    if platforms.is_runtime_present(VENDOR / "whisper_cuda"):
        print("  CUDA/Metal 运行时已就绪")
    elif download(f"{BASE}/{cuda_asset}", VENDOR / "whisper-cuda.zip",
                  f"CUDA {variant} 构建（较大）"):
        print("  解压中 …")
        _extract(VENDOR / "whisper-cuda.zip", cuda_dest)
        (VENDOR / "whisper-cuda.zip").unlink(missing_ok=True)
        # 跨平台验证
        if not platforms.is_runtime_present(VENDOR / "whisper_cuda"):
            raise SystemExit(f"CUDA/Metal 运行时解压异常：预期位置没有找到库文件 ({cuda_dest})")
        print(f"  CUDA/Metal 运行时已安装到 {cuda_dest}")
    else:
        return False

    # CUDA runtime（cudart）核验。旧版在这里下载 "cudart-llama-win-x64.zip"，
    # 但这个名字在 whisper.cpp 的任何 release 里都不存在（b5130 全部 12 个
    # 资产实测无此包，恒 404）。实测 b5130 的 whisper-cublas-12.4.0 包已把
    # cudart64_12.dll / nvrtc / nvblas 打进包里（本机 CUDA 目录与 whisper.dll
    # 同批解出、CPU 目录无这些文件即证），所以解压后核验就位即可；
    # 真缺了给出手动补齐指引，而不是请求幽灵 URL。
    if plat.key == "windows":
        if list(cuda_dest.rglob("cudart*.dll")):
            print("  CUDA runtime (cudart) 已随 CUDA 构建包就绪")
        else:
            print("  [警告] CUDA 构建包解压后没有 cudart DLL，CUDA 起不来。")
            print("         请安装 CUDA Toolkit 12.x，把 cudart64_12.dll 复制到")
            print(f"         {cuda_dest / 'Release'}")
    return True


def fetch_models(which: list[str], step: str) -> None:
    print(f"\n[{step}] 模型")
    for key in which:
        fname = MODELS_AVAILABLE[key]
        if download(f"{HF}/{fname}", MODELS / fname, f"{key} 模型"):
            continue
        print("  主站失败，切换 hf-mirror 重试 …")
        download(f"{HF_MIRROR}/{fname}", MODELS / fname, f"{key} 模型（镜像）")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL, choices=sorted(MODELS_AVAILABLE))
    ap.add_argument("--cuda", action="store_true", help="额外下载 CUDA/Metal 运行时")
    ap.add_argument("--cuda-variant", choices=("12.4", "11.8"), default=None,
                    help="强制指定 CUDA 变体（默认按算力/驱动自动选）")
    ap.add_argument("--all", action="store_true", help="下载全部模型")
    args = ap.parse_args()

    plat = platforms.current()
    if not platforms.is_supported():
        print(f"提示: {plat.key} 尚未提供开箱即用的预编译包。")

    print(f"whisper.cpp 构建版本: {RELEASE_TAG}")
    if not fetch_runtime(args.cuda, args.cuda_variant):
        print("\n运行时没能就位，后面装什么都不管用。先解决上面的下载问题再重试。")
        return 1

    # 步数按实际要走的步骤算，跳过 CUDA 时不该显示 3/3
    fetch_models(list(MODELS_AVAILABLE) if args.all else [args.model],
                 "3/3" if args.cuda else "2/2")

    print("\n完成。现在可以运行:  python app.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
