"""首次运行的资源下载。

打包后的 exe 只有 6MB 左右，whisper.cpp 运行时（40MB）和模型（466MB）
必须单独下载。源码运行时用 tools/fetch_assets.py 就行，但 exe 里没有
那个脚本，所以下载逻辑要能在程序内跑。

放在独立模块而不是塞进 app.py，是为了让它在 PyInstaller 下也能被打包，
并且能独立测试。
"""

from __future__ import annotations

import threading
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import platforms
import device

RELEASE_TAG = "b5454"
BASE = f"https://github.com/ggml-org/whisper.cpp/releases/download/{RELEASE_TAG}"
HF = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main"

# 官方 release 资产名按平台不同。CUDA 按变体选（12.4 新卡新驱动 /
# 11.8 老卡老驱动），由 device.preferred_cuda_variant() 按算力与驱动定。
ASSETS = {
    "windows": {
        "runtime": "whisper-bin-x64.zip",
        "cuda": {
            "12.4": "whisper-bin-win-cuda-12.4.0-x64.zip",
            "11.8": "whisper-bin-win-cuda-11.8.0-x64.zip",
        },
        # 不声明独立的 cudart 资产：whisper.cpp 的 release 里从来没有
        # "cudart-llama-win-x64.zip"（恒 404），CUDA 包已内置 cudart。
        # 缺 cudart 的核验在 _fetch_runtime 里做。
    },
    "macos": {"runtime": f"whisper-{RELEASE_TAG}-xcframework.zip",
              "cuda": None},
    "linux": {"runtime": None, "cuda": None},
}

MODELS = {
    "tiny": "ggml-tiny.bin",
    "base": "ggml-base.bin",
    "small": "ggml-small.bin",
    "medium": "ggml-medium.bin",
    "large-v3-turbo": "ggml-large-v3-turbo.bin",
}


@dataclass
class Step:
    desc: str
    total: int = 0
    done: int = 0

    @property
    def fraction(self) -> float:
        return (self.done / self.total) if self.total else 0.0


class AssetFetcher:
    """后台下载 + 进度回调。调用方在 UI 线程通过 signal 接进度。"""

    def __init__(self, root: Path, model: str = "small", with_cuda: bool = False):
        self.root = Path(root)
        self.model = model
        self.with_cuda = with_cuda
        self.step = Step("准备中")
        self.error = ""
        self.done = False

    def has_runtime(self) -> bool:
        return platforms.is_runtime_present(self.root / "vendor")

    def has_model(self) -> bool:
        return (self.root / "models" / MODELS.get(self.model, "")).is_file()

    # -- 实际执行（跑在后台线程）----------------------------------------

    def run(self) -> None:
        try:
            if not self.has_runtime():
                self._fetch_runtime()
            if not self.has_model():
                self._fetch_model()
            self.done = True
            self.step = Step("完成")
        except Exception as e:
            self.error = str(e)
            self.step = Step("失败")

    def _download(self, url: str, dest: Path, desc: str) -> None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(dest.suffix + ".part")
        self.step = Step(desc)
        with urllib.request.urlopen(url, timeout=60) as r:
            total = int(r.headers.get("Content-Length") or 0)
            done = 0
            with open(tmp, "wb") as f:
                while True:
                    block = r.read(1 << 20)
                    if not block:
                        break
                    f.write(block)
                    done += len(block)
                    self.step = Step(desc, total, done)
        tmp.replace(dest)

    def _fetch_runtime(self) -> None:
        plat = platforms.current()
        a = ASSETS.get(plat.key, {})
        name = a.get("runtime")
        if not name:
            raise RuntimeError(
                f"{plat.key} 没有官方预编译包：{plat.build_hint}")

        zip_path = self.root / "vendor" / "whisper-bin.zip"
        self._download(f"{BASE}/{name}", zip_path, "下载 whisper.cpp 运行时")
        self.step = Step("解压运行时")
        dest = self.root / "vendor" / "whisper"
        dest.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(dest)
        zip_path.unlink(missing_ok=True)

        # 跨平台验证运行时是否就绪
        if not platforms.is_runtime_present(self.root / "vendor"):
            raise RuntimeError("解压后没找到主库，资产结构可能变了")

        if self.with_cuda and a.get("cuda"):
            # 变体按算力 + 驱动选：老卡/老驱动给 11.8（CUDA 12.4 运行时
            # 要求驱动 >= 552），其余 12.4。选不出（无 N 卡/算力过低/缺
            # nvidia-smi）就跳过 CUDA 下载，别白拉 700MB——核显与老显卡
            # 走 CPU 路径即可正常使用。
            variant = device.preferred_cuda_variant()
            cuda_asset = a["cuda"].get(variant) if variant else None
            if not cuda_asset:
                self.step = Step("跳过 CUDA（无可用 CUDA 环境，走 CPU）")
                return
            czip = self.root / "vendor" / "whisper-cuda.zip"
            self._download(f"{BASE}/{cuda_asset}", czip,
                           f"下载 CUDA {variant} 构建（较大）")
            cdest = self.root / "vendor" / "whisper_cuda"
            cdest.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(czip) as z:
                z.extractall(cdest)
            czip.unlink(missing_ok=True)

            # CUDA runtime（cudart）核验。旧版把 cudart 当独立资产下载，
            # 但 "cudart-llama-win-x64.zip" 这个名字在 whisper.cpp 的 release
            # 里根本不存在（b5130 全部 12 个资产实测无此包），新机器装 CUDA
            # 必炸 404。实测 b5130 的 whisper-cublas-12.4.0 包已内置
            # cudart64_12.dll / nvrtc / nvblas（本机 vendor/whisper_cuda/Release
            # 与 whisper.dll 同批解出、CPU 目录无这些文件即为证据），
            # 所以解压后直接核验就位情况。cudart 缺失的失败是静默的——
            # ggml 只在日志里留一行 "no GPU found" 就退回 CPU，用户会以为
            # 是驱动问题——必须在这里拦住。
            ext = platforms.current().lib_ext
            if not list(cdest.rglob(f"cudart*.{ext}")):
                raise RuntimeError(
                    "CUDA 构建包解压后没有 cudart DLL，CUDA 无法启用（缺失时"
                    "程序会静默退回 CPU）。请安装 CUDA Toolkit 12.x，把 "
                    f"cudart64_12.dll 复制到 {cdest / 'Release'} 后重启程序。"
                )

    def _fetch_model(self) -> None:
        fname = MODELS.get(self.model)
        if not fname:
            raise RuntimeError(f"未知模型 {self.model}")
        self._download(f"{HF}/{fname}", self.root / "models" / fname,
                       f"下载 {self.model} 模型（约 500MB）")


def fetch_background(root: Path, model: str, with_cuda: bool,
                     on_progress, on_done) -> AssetFetcher:
    """起一个后台下载线程。on_progress(fetcher) 会在下载中被反复调用。"""
    f = AssetFetcher(root, model, with_cuda)

    def loop():
        last = 0.0
        # 进度回调很频繁，自己节流一下，别把 UI 线程淹了
        def tick():
            nonlocal last
            import time
            now = time.monotonic()
            if now - last > 0.2:
                last = now
                on_progress(f)
        f._tick = tick
        _run_with_progress(f, tick)
        on_done(f)

    t = threading.Thread(target=loop, daemon=True)
    t.start()
    return f


def _run_with_progress(f: AssetFetcher, tick) -> None:
    # AssetFetcher.run 内部更新 self.step，这里包一层定期上报
    import time
    stop = threading.Event()

    def watcher():
        while not stop.is_set():
            tick()
            stop.wait(0.15)

    w = threading.Thread(target=watcher, daemon=True)
    w.start()
    try:
        f.run()
    finally:
        stop.set()
