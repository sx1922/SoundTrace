"""参数扫描：找出哪一项 decode 参数在吃字。

已知事实（tools/ab_single_segment.py 验证过）：
  - 四个模型（base/small/medium/large-v3-turbo）在同一处丢同样的词
  - single_segment 开关无影响
  - 幻觉/碎片过滤净效果为 0
  - 样本音频时长与参考字数比一致，TTS 没截断

所以问题只可能出在 whisper_bridge 里被我们改过的 decode 参数上。
这个脚本逐项把参数恢复成 whisper 默认，看哪一项能救回尾部。

用法：
    python tools/sweep_params.py --model small
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import ctypes


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="small")
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    import device
    import make_bench as mb
    import whisper_bridge as wb
    from transcribe import Transcriber

    cases = [c for c in mb.load_cases() if c.wav.is_file()]
    info = device.detect("cpu" if args.cpu else "auto")
    print(f"模型 {args.model} · 设备 {info.label}\n")

    tr = Transcriber(
        ROOT / "models" / f"ggml-{args.model}.bin",
        use_gpu=info.use_gpu, n_threads=device.default_threads(),
        language="zh", vendor_dir=info.vendor_dir,
    )
    tr.load_blocking()
    eng = tr._engine

    # 预先把每条的整段音频取出来，避免重复做 VAD
    payloads = []
    for c in cases:
        s = mb.read_wav(c.wav)
        payloads.append((c, np.ascontiguousarray(s, dtype=np.int16)))

    def decode(pcm_i16, **over):
        pcm = pcm_i16.astype(np.float32) / 32768.0
        params = eng._dll.whisper_full_default_params(wb.WHISPER_SAMPLING_GREEDY)
        params.n_threads = eng.n_threads
        params.language = b"zh"
        params.translate = False
        params.no_timestamps = True
        params.single_segment = True
        params.print_progress = False
        params.print_realtime = False
        params.print_special = False
        params.print_timestamps = False
        params.suppress_blank = True
        params.no_speech_thold = 0.6
        params.temperature = 0.0
        params.temperature_inc = 0.0
        params.greedy.best_of = 1
        for k, v in over.items():
            setattr(params, k, v)
        rc = eng._dll.whisper_full(
            eng._ctx, params,
            pcm.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), pcm.size)
        if rc != 0:
            return f"<rc={rc}>"
        parts = []
        for i in range(eng._dll.whisper_full_n_segments(eng._ctx)):
            raw = eng._dll.whisper_full_get_segment_text(eng._ctx, i)
            parts.append((raw or b"").decode("utf-8", "replace"))
        return "".join(parts).strip()

    variants = {
        "现状（全部显式设定）": {},
        "timesteps 开启": dict(no_timestamps=False),
        "suppress_blank 关": dict(suppress_blank=False),
        "best_of=5": dict(greedy=None) if False else {},
        "温度回退 inc=0.2": dict(temperature_inc=0.2),
        "beam_size=5": None,     # 特殊处理
        "全默认": None,          # 特殊处理
    }

    def decode_beam(pcm_i16):
        pcm = pcm_i16.astype(np.float32) / 32768.0
        params = eng._dll.whisper_full_default_params(wb.WHISPER_SAMPLING_BEAM_SEARCH)
        params.n_threads = eng.n_threads
        params.language = b"zh"
        params.translate = False
        params.no_timestamps = True
        params.single_segment = True
        params.print_progress = False
        params.print_realtime = False
        params.beam_search.beam_size = 5
        rc = eng._dll.whisper_full(
            eng._ctx, params,
            pcm.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), pcm.size)
        if rc != 0:
            return f"<rc={rc}>"
        parts = []
        for i in range(eng._dll.whisper_full_n_segments(eng._ctx)):
            raw = eng._dll.whisper_full_get_segment_text(eng._ctx, i)
            parts.append((raw or b"").decode("utf-8", "replace"))
        return "".join(parts).strip()

    def decode_default(pcm_i16):
        pcm = pcm_i16.astype(np.float32) / 32768.0
        params = eng._dll.whisper_full_default_params(wb.WHISPER_SAMPLING_GREEDY)
        params.n_threads = eng.n_threads
        params.language = b"zh"
        params.translate = False
        rc = eng._dll.whisper_full(
            eng._ctx, params,
            pcm.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), pcm.size)
        if rc != 0:
            return f"<rc={rc}>"
        parts = []
        for i in range(eng._dll.whisper_full_n_segments(eng._ctx)):
            raw = eng._dll.whisper_full_get_segment_text(eng._ctx, i)
            parts.append((raw or b"").decode("utf-8", "replace"))
        return "".join(parts).strip()

    print(f"{'变体':24s} {'CER':>8s}")
    print("-" * 60)
    for label, over in variants.items():
        if label == "beam_size=5":
            fn = decode_beam
        elif label == "全默认":
            fn = decode_default
        else:
            fn = lambda p, _o=over: decode(p, **_o)
        ed = ref = 0
        for c, pcm in payloads:
            hyp = fn(pcm)
            ed += c.errors(hyp)
            ref += c.ref_len()
        print(f"{label:24s} {ed / ref * 100:7.2f}%  ({ed}/{ref})")

    print("\n--- 关键样本 mixed 的尾部（参考尾部：…延迟控制在三百毫秒以内）---")
    for label, over in variants.items():
        if label == "beam_size=5":
            fn = decode_beam
        elif label == "全默认":
            fn = decode_default
        else:
            fn = lambda p, _o=over: decode(p, **_o)
        for c, pcm in payloads:
            if c.name == "mixed":
                print(f"{label:24s} {fn(pcm)}")
                break
    tr.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())