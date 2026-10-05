"""A/B：whisper 的 single_segment 开关对漏字的影响。

起因：评测里所有模型都在同一处截断（"…延迟"后面"控制在三百毫秒以内"
整段消失），这不像模型能力问题，像是管线把内容吃了。
嫌疑是 whisper_bridge 里 params.single_segment = True——它让 whisper
把整段音频当成"一段话"来解码，而模型对超长单段的处理方式是压缩
而不是逐字转写。

这个脚本固定其它所有参数，只切这一个开关，跑同一套样本比 CER。

用法：
    python tools/ab_single_segment.py --model small
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="small")
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    import device
    import make_bench as mb
    from config import Config
    from text_norm import is_hallucination, is_suspicious
    from transcribe import Transcriber
    from vad import VADConfig, Segmenter

    cases = [c for c in mb.load_cases() if c.wav.is_file()]
    if not cases:
        print("没有测试集")
        return 1

    info = device.detect("cpu" if args.cpu else "auto")
    print(f"模型 {args.model} · 设备 {info.label}\n")

    tr = Transcriber(
        ROOT / "models" / f"ggml-{args.model}.bin",
        use_gpu=info.use_gpu, n_threads=device.default_threads(),
        language="zh", vendor_dir=info.vendor_dir,
    )
    tr.load_blocking()

    vcfg = VADConfig.from_config(Config.load())

    def run_all(single: bool):
        ed = ref = 0
        detail = []
        for c in cases:
            samples = mb.read_wav(c.wav)
            seg = Segmenter(vcfg)
            out: list[str] = []

            def one(s):
                r = tr._engine.transcribe(s.samples, language="zh",
                                          single_segment=single)
                t = tr.normalizer(r.text)
                if t and not is_hallucination(t) and not is_suspicious(t, s.duration_ms):
                    out.append(t)

            for i in range(0, samples.size, 1024):
                for s in seg.feed(samples[i : i + 1024]):
                    one(s)
            for s in seg.flush():
                one(s)

            hyp = "".join(out)
            e = c.errors(hyp)
            r = c.ref_len()
            ed += e
            ref += r
            detail.append((c.name, e, r, hyp))
        return ed, ref, detail

    totals = {}
    for label, single in (("single_segment=True （现状）", True),
                          ("single_segment=False", False)):
        ed, ref, detail = run_all(single)
        totals[label] = (ed, ref, detail)
        print(f"--- {label} ---")
        for name, e, r, hyp in detail:
            print(f"  {name:10s} {e:3d}/{r:3d}  {hyp}")
        print(f"  CER = {ed / ref * 100:.2f}%  ({ed}/{ref})\n")

    (e1, r1, d1) = totals["single_segment=True （现状）"]
    (e2, r2, d2) = totals["single_segment=False"]
    print("--- 逐条对比（False 相对 True）---")
    by_name = {n: e for n, e, _, _ in d2}
    for name, e, r, _ in d1:
        e_other = by_name.get(name)
        if e_other is not None and e_other != e:
            verdict = "False 更好" if e_other < e else "True 更好"
            print(f"  {name:10s} {e:3d} -> {e_other:3d}   {verdict}")

    print(f"\n总计：True {e1}/{r1} = {e1 / r1 * 100:.2f}%   "
          f"False {e2}/{r2} = {e2 / r2 * 100:.2f}%")
    print(f"净变化：{e2 - e1:+d} 字（负数=更好）")
    tr.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())