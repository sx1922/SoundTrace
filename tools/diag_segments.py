"""诊断：逐段 dump 识别结果，定位内容是在哪一步丢的。

make_bench 只给最终 CER，看不出"是切段丢了还是识别丢了"。
这里把每段的时长和文本都打出来，能一眼看出边界处的问题。

用法：
    python tools/diag_segments.py --model small --case numbers
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="small")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--case", default="numbers")
    args = ap.parse_args()

    import device
    import make_bench as mb
    from config import Config
    from transcribe import Transcriber
    from vad import VADConfig, Segmenter

    cases = {c.name: c for c in mb.load_cases()}
    case = cases.get(args.case)
    if case is None:
        print(f"没有样本 {args.case}，可选：{list(cases)}")
        return 1

    info = device.detect("cpu" if args.cpu else "auto")
    tr = Transcriber(
        ROOT / "models" / f"ggml-{args.model}.bin",
        use_gpu=info.use_gpu, n_threads=device.default_threads(),
        language="zh", vendor_dir=info.vendor_dir,
    )
    tr.load_blocking()

    cfg = Config.load()
    vcfg = VADConfig.from_config(cfg)
    print(f"参考：{case.reference}")
    print(f"VAD：end_frames={vcfg.end_frames} "
          f"(≈{vcfg.end_frames * 20}ms) "
          f"preroll={vcfg.preroll_ms}ms "
          f"min_segment={vcfg.min_segment_ms}ms "
          f"max_segment={vcfg.max_segment_s}s "
          f"aggr={vcfg.aggressiveness}\n")

    samples = mb.read_wav(case.wav)
    print(f"音频：{samples.size / 16000:.2f}s")

    seg = Segmenter(vcfg)
    n = 0
    got = []

    def handle(s, why):
        nonlocal n
        n += 1
        r = tr._engine.transcribe(s.samples, language="zh")
        text = tr.normalizer(r.text)
        dropped = ""
        from text_norm import is_hallucination, is_suspicious
        if is_hallucination(text) or is_suspicious(text, s.duration_ms):
            dropped = "  <-- 被过滤"
        print(f"  #{n} [{why}] {s.start_sample / 16000:6.2f}s-"
              f"{s.end_sample / 16000:6.2f}s ({s.duration_ms:5d}ms, "
              f"{s.samples.size}样本) {text}{dropped}")
        if text and not dropped:
            got.append(text)

    for i in range(0, samples.size, 1024):
        for s in seg.feed(samples[i : i + 1024]):
            handle(s, "feed")
    for s in seg.flush():
        handle(s, "flush")

    hyp = "".join(got)
    print(f"\n合计：{hyp}")
    print(f"CER = {case.errors(hyp) / case.ref_len() * 100:.2f}% "
          f"({case.errors(hyp)}/{case.ref_len()})")
    tr.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())