"""诊断：幻觉/碎片过滤到底是在帮忙还是在吃字。

make_bench 报的 CER 是"用户实际看到"的数字，但它把过滤器的损耗
和模型识别误差混在一起了。这里把同一套样本跑两遍——一遍照常过滤，
一遍绕过 text_norm 的两个判定——看净收益。

用法：
    python tools/diag_filters.py --model small
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
    args = ap.parse_args()

    import device
    import text_norm
    from transcribe import Transcriber

    import make_bench as mb

    cases = mb.load_cases()
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

    # 两遍：一次用真过滤器，一次把它们换成永远放行
    orig_h, orig_s = text_norm.is_hallucination, text_norm.is_suspicious
    text_norm.is_hallucination = lambda t: False
    text_norm.is_suspicious = lambda t, d: False
    import transcribe as _tr_mod
    _tr_mod.is_hallucination = text_norm.is_hallucination
    _tr_mod.is_suspicious = text_norm.is_suspicious

    tot = {}
    per_case: dict[str, dict[str, tuple[int, int, str]]] = {}
    for label, bypass in (("带过滤", False), ("不过滤", True)):
        if not bypass:
            text_norm.is_hallucination = orig_h
            text_norm.is_suspicious = orig_s
            _tr_mod.is_hallucination = orig_h
            _tr_mod.is_suspicious = orig_s
        ed = ref = 0
        rows = []
        per_case[label] = {}
        for c in cases:
            if not c.wav.is_file():
                continue
            hyp = mb.transcribe_case(tr, c, False)
            e = c.errors(hyp)
            r = c.ref_len()
            ed += e
            ref += r
            rows.append((c.name, e, r, hyp))
            per_case[label][c.name] = (e, r, hyp)
        tot[label] = (ed, ref)
        print(f"--- {label} ---")
        for name, e, r, hyp in rows:
            print(f"  {name:10s} {e:3d}/{r:3d}  {hyp}")
        print(f"  CER = {ed / ref * 100:.2f}%  ({ed}/{ref})\n")

    text_norm.is_hallucination = orig_h
    text_norm.is_suspicious = orig_s
    _tr_mod.is_hallucination = orig_h
    _tr_mod.is_suspicious = orig_s

    (e1, r1), (e2, r2) = tot["带过滤"], tot["不过滤"]
    print("--- 逐条对比（过滤让哪条变差了）---")
    for name, (e_f, r_f, h_f) in per_case["带过滤"].items():
        e_n, r_n, h_n = per_case["不过滤"].get(name, (0, 0, ""))
        if e_f != e_n:
            verdict = "过滤反而更差" if e_f > e_n else "过滤有帮助"
            print(f"  {name:10s} {e_n:3d} -> {e_f:3d}  {verdict}")
            if e_f > e_n:
                print(f"      带过滤: {h_f}")
                print(f"      不过滤: {h_n}")

    d = e2 - e1
    print(f"\n过滤器的净效果：{'少错' if d < 0 else '多错'} {abs(d)} 字"
          f"（带过滤 {e1}/{r1}，不过滤 {e2}/{r2}）")
    tr.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())