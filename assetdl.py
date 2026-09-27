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

RELEASE_TAG = "b5130"
BASE = f"https://github.com/ggml-org/whisper.cpp/releases/download/{RELEASE_TAG}"
HF = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main"

# 官方 release 资产名按平台不同
ASSETS = {
    "windows": {
        "runtime": "whisper-bin-x64.zip",
        "cuda": "whisper-cublas-12.4.0-bin-x64.zip",
        "cudart": "cudart-llama-win-x64.zip",
    },
    "macos": {"runtime": f"whisper-{RELEASE_TAG}-xcframework.zip",
              "cuda": None, "cudart": None},
    "linux": {"runtime": None, "cuda": None, "cudart": None},
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
        plat = platforms.current()
        d = self.root / "vendor" / "whisper" / "Release"
        return (d / plat.whisper_path_name).is_file()

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

        if not (dest / "Release" / platforms.current().whisper_path_name).is_file():
            raise RuntimeError("解压后没找到主库，资产结构可能变了")

        if self.with_cuda and a.get("cuda"):
            czip = self.root / "vendor" / "whisper-cuda.zip"
            self._download(f"{BASE}/{a['cuda']}", czip, "下载 CUDA 构建（较大）")
            cdest = self.root / "vendor" / "whisper_cuda"
            cdest.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(czip) as z:
                z.extractall(cdest)
            czip.unlink(missing_ok=True)

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
