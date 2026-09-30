"""对比不同模型 / 推理参数下的内存占用。

用法: python tools/bench_memory2.py [model] [cpu|gpu] [n_max_text_ctx]
"""

from __future__ import annotations

import ctypes
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from memprobe import rss_mb  # noqa: E402
from whisper_bridge import WhisperEngine  # noqa: E402


def main() -> int:
    model = sys.argv[1] if len(sys.argv) > 1 else "ggml-small.bin"
    use_gpu = (sys.argv[2] if len(sys.argv) > 2 else "cpu") == "gpu"
    nctx = int(sys.argv[3]) if len(sys.argv) > 3 else 0

    vendor = "whisper_cuda" if use_gpu else "whisper"
    before = rss_mb()

    e = WhisperEngine(ROOT / "models" / model, use_gpu=use_gpu, n_threads=6,
                      vendor_dir=ROOT / "vendor" / vendor / "Release")
    e.load()
    e.silence_logs()
    loaded = rss_mb()

    d = e._dll
    p = d.whisper_full_default_params(0)
    if nctx:
        p.n_max_text_ctx = nctx
    p.n_threads = 6
    p.language = b"zh"
    p.no_timestamps = True
    p.single_segment = True
    p.temperature = 0.0
    p.temperature_inc = 0.0
    p.greedy.best_of = 1
    pcm = np.zeros(16000 * 5, dtype=np.float32)
    d.whisper_full(e._ctx, p, pcm.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), pcm.size)
    ran = rss_mb()
    e.free()

    tag = f"{model.replace('ggml-','').replace('.bin',''):<7} {'GPU' if use_gpu else 'CPU'}"
    if nctx:
        tag += f" nctx={nctx}"
    print(f"{tag:<24} 启动 {before:6.0f}  加载后 {loaded:7.0f}  推理后 {ran:7.0f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
