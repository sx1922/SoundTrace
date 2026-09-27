"""首次运行：下载 whisper.cpp 运行时和 GGUF 模型。

用法:
    python tools/fetch_assets.py            # CPU 运行时 + 默认模型
    python tools/fetch_assets.py --model medium
    python tools/fetch_assets.py --cuda     # 额外下载 CUDA 构建（需要 N 卡）
    python tools/fetch_assets.py --all      # 下所有模型
"""

from __future__ import annotations

import argparse
import sys
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import platforms

VENDOR = ROOT / "vendor"
MODELS = ROOT / "models"

# 挑一个带预编译包的 tag
RELEASE_TAG = "b5130"

# 官方 release 的资产名按平台不同。Linux 没有预编译包，需要自行编译。
_ASSETS = {
    "windows": {
        "cpu": "whisper-bin-x64.zip",
        "cuda": "whisper-cublas-12.4.0-bin-x64.zip",
        "cudart": "cudart-llama-win-x64.zip",
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


def _extract(zip_path: Path, dest: Path) -> None:
    """把 release 包解到 dest 下。

    包内路径是 "Release/xxx.dll"，直接解到 vendor/ 会得到 vendor/Release/，
    而代码期望的是 vendor/whisper/Release/（CPU）和
    vendor/whisper_cuda/Release/（CUDA）。所以解到目标目录本身，
    让 "Release/" 这一层自然落在正确的位置。
    """
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


def fetch_runtime(cuda: bool) -> bool:
    """返回 True 表示 CPU 运行时确实就位。

    运行时是硬依赖，缺了后面一步都跑不了，所以失败必须中断，
    不能像之前那样打完"完成"字样再让用户去 app.py 里撞报错。
    """
    print("\n[1/2] whisper.cpp 运行时")
    dest = VENDOR / "whisper"
    if (dest / "Release" / "whisper.dll").is_file():
        print("  CPU 运行时已就绪")
    elif download(f"{BASE}/{A['cpu']}", VENDOR / "whisper-bin-x64.zip", "CPU 运行时"):
        print("  解压中 …")
        _extract(VENDOR / "whisper-bin-x64.zip", dest)
        (VENDOR / "whisper-bin-x64.zip").unlink(missing_ok=True)
        _require(dest / "Release" / "whisper.dll", "CPU 运行时")
    else:
        return False

    if not cuda:
        return True

    print("\n[2/3] CUDA 运行时")
    cuda_dest = VENDOR / "whisper_cuda"
    if (cuda_dest / "Release" / "whisper.dll").is_file():
        print("  CUDA 运行时已就绪")
    elif download(f"{BASE}/{A['cuda']}", VENDOR / "whisper-cuda.zip", "CUDA 构建（较大）"):
        print("  解压中 …")
        _extract(VENDOR / "whisper-cuda.zip", cuda_dest)
        (VENDOR / "whisper-cuda.zip").unlink(missing_ok=True)
        _require(cuda_dest / "Release" / "whisper.dll", "CUDA 运行时")

    # CUDA 构建不含 cudart，得单独补，否则驱动再新也起不来
    print("  补齐 CUDA runtime DLL …")
    if download(f"{BASE}/{A['cudart']}", VENDOR / "cudart.zip", "cudart"):
        with zipfile.ZipFile(VENDOR / "cudart.zip") as z:
            z.extractall(cuda_dest / "Release")
        (VENDOR / "cudart.zip").unlink(missing_ok=True)
        print("  已安装 cudart")
    return True


def fetch_models(which: list[str], step: str) -> None:
    print(f"\n[{step}] 模型")
    for key in which:
        fname = MODELS_AVAILABLE[key]
        download(f"{HF}/{fname}", MODELS / fname, f"{key} 模型")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL, choices=sorted(MODELS_AVAILABLE))
    ap.add_argument("--cuda", action="store_true", help="额外下载 CUDA 构建")
    ap.add_argument("--all", action="store_true", help="下载全部模型")
    args = ap.parse_args()

    plat = platforms.current()
    if not platforms.is_supported():
        print(f"提示: {plat.key} 尚未提供开箱即用的预编译包。")

    print(f"whisper.cpp 构建版本: {RELEASE_TAG}")
    if not fetch_runtime(args.cuda):
        print("\n运行时没能就位，后面装什么都不管用。先解决上面的下载问题再重试。")
        return 1
    # 步数按实际要走的步骤算，跳过 CUDA 时不该显示 3/3
    fetch_models(list(MODELS_AVAILABLE) if args.all else [args.model],
                 "3/3" if args.cuda else "2/2")

    print("\n完成。现在可以运行:  python app.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
