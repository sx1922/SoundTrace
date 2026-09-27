"""生成带标准答案的中文测试集，用于量化识别质量。

用系统 TTS 合成若干段不同风格的语音，保存 wav 和标准答案，
之后可以算字符错误率（CER）来对比不同参数/模型/改动的效果。

用法:
    python tools/make_bench.py           # 生成（或补齐）测试集
    python tools/make_bench.py --eval    # 跑当前模型，算出 CER
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # 脚本在 tools/ 下，要能 import 到项目模块
BENCH_DIR = ROOT / "tests" / "bench"
SAMPLE_RATE = 16000

# 覆盖不同场景：短句、连说、带数字、英文混排、疑问语气
SAMPLES: list[tuple[str, str]] = [
    ("meeting", "今天下午三点，我们开个短会，讨论一下语音转文字工具的优化方向。"),
    ("numbers", "会议纪要：第一项，成本控制，目标是把预算降低百分之十五。第二项，进度管理，本月底之前完成交付。"),
    ("mixed", "这个方案的吞吐量大概是每秒两千条，准确率能到百分之九十五以上，延迟控制在三百毫秒以内。"),
    ("question", "你有没有考虑过，用户在弱网环境下上传大文件的时候，会不会中途失败？这个问题该怎么解决？"),
    ("longform", "接下来我说三点。第一，模型要更小，部署成本才能降下来。第二，断句要更自然，不能一句一断。第三，导出格式要支持字幕，方便后期剪辑。这三点都很关键，请大家认真讨论。"),
    ("short", "好的，收到。"),
]

_PUNCT = re.compile(r"[\s，。！？、；：""''（）,.!?;:\"'()\-—…·]")


@dataclass
class BenchCase:
    name: str
    wav: Path
    reference: str

    def ref_len(self) -> int:
        return len(_PUNCT.sub("", self.reference))

    def errors(self, hypothesis: str) -> int:
        """去掉标点后的编辑距离（错字个数）。

        标点不计入：它由规则生成，不反映模型能力；繁体已在上游统一。
        """
        ref = _PUNCT.sub("", self.reference)
        hyp = _PUNCT.sub("", hypothesis)
        if not ref:
            return 0
        return _levenshtein(ref, hyp)


def _levenshtein(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def _synth(text: str, dest: Path) -> bool:
    raw = dest.with_suffix(".raw.wav")
    ps = (
        "Add-Type -AssemblyName System.Speech; "
        "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
        "$v = $s.GetInstalledVoices() | "
        "Where-Object { $_.VoiceInfo.Culture -like 'zh*' } | "
        "Select-Object -First 1; "
        "if (-not $v) { exit 1 }; "
        f"$s.SelectVoice($v.VoiceInfo.Name); "
        f"$s.SetOutputToWaveFile('{raw}'); "
        f"$s.Speak('{text}'); $s.SetOutputToNull()"
    )
    r = subprocess.run(["powershell", "-NoProfile", "-Command", ps],
                       capture_output=True, text=True, timeout=180)
    if r.returncode != 0 or not raw.is_file():
        return False
    _to_16k(raw, dest)
    raw.unlink(missing_ok=True)
    return True


def _to_16k(src: Path, dst: Path) -> None:
    with wave.open(str(src), "rb") as w:
        rate = w.getframerate()
        x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32)
    if rate == SAMPLE_RATE:
        y = x.astype(np.int16)
    else:
        t = np.arange(len(x) * SAMPLE_RATE / rate, dtype=np.float32)
        y = np.interp(t, np.arange(len(x), dtype=np.float32), x).astype(np.int16)
    with wave.open(str(dst), "wb") as o:
        o.setnchannels(1)
        o.setsampwidth(2)
        o.setframerate(SAMPLE_RATE)
        o.writeframes(y.tobytes())


def build() -> int:
    BENCH_DIR.mkdir(parents=True, exist_ok=True)
    manifest = BENCH_DIR / "manifest.json"
    made = 0
    for name, text in SAMPLES:
        wav = BENCH_DIR / f"{name}.wav"
        if wav.is_file():
            continue
        if _synth(text, wav):
            made += 1
            print(f"  生成 {name}.wav")
        else:
            print("系统没有中文 TTS 语音，无法生成测试集。")
            return 1
    manifest.write_text(
        json.dumps([{"name": n, "reference": t} for n, t in SAMPLES],
                   ensure_ascii=False, indent=2),
        encoding="utf-8")
    print(f"测试集就绪（共 {len(SAMPLES)} 条，新生成 {made} 条）")
    return 0


def load_cases() -> list[BenchCase]:
    manifest = BENCH_DIR / "manifest.json"
    if not manifest.is_file():
        return []
    data = json.loads(manifest.read_text(encoding="utf-8"))
    return [BenchCase(d["name"], BENCH_DIR / f"{d['name']}.wav", d["reference"])
            for d in data]


def read_wav(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as w:
        return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)


def evaluate(model: str, gpu: bool, use_context: bool) -> int:
    """跑完整管线并算 CER。

    刻意走 Transcriber 而不是直接调引擎：繁简统一、标点规整、幻觉过滤
    都是用户实际会看到的处理，绕过去测出来的数字没有意义
    （之前直接调引擎测出 41.5% CER，一半是繁体字造成的假错误）。
    """
    import device
    from transcribe import Transcriber

    cases = load_cases()
    if not cases:
        print("没有测试集，先运行: python tools/make_bench.py")
        return 1

    info = device.detect("gpu" if gpu else "auto")
    print(f"模型 {model} · 设备 {info.label} · 段间上下文 {'开' if use_context else '关'}\n")

    tr = Transcriber(
        ROOT / "models" / f"ggml-{model}.bin",
        use_gpu=info.use_gpu, n_threads=device.default_threads(),
        language="zh", vendor_dir=info.vendor_dir,
    )
    tr.load_blocking()

    total_ref = total_ed = 0
    for c in cases:
        if not c.wav.is_file():
            continue
        hyp = transcribe_case(tr, c, use_context)
        ed = c.errors(hyp)
        ref_len = c.ref_len()
        cer = ed / ref_len if ref_len else 0.0
        total_ref += ref_len
        total_ed += ed
        mark = "ok " if cer <= 0.05 else ("~  " if cer <= 0.2 else "差 ")
        print(f"{mark} {c.name:10s} CER={cer * 100:5.1f}%  {hyp}")

    if total_ref:
        print(f"\n总体 CER = {total_ed / total_ref * 100:.2f}%  "
              f"（{total_ed} / {total_ref} 字）")
    tr.stop()
    return 0


def transcribe_case(tr, case: BenchCase, use_context: bool) -> str:
    """走和正式程序一样的分段与识别逻辑。

    完整结果和"喂给下一段的上下文"是两回事：即使关掉上下文，
    所有段落的识别结果也必须拼起来，否则只剩最后一段。
    """
    from config import Config
    from vad import VADConfig, Segmenter

    samples = read_wav(case.wav)
    seg = Segmenter(VADConfig.from_config(Config.load()))
    out: list[str] = []
    ctx = ""

    def handle(s):
        nonlocal ctx
        text = run_one(tr, s.samples, ctx if use_context else "")
        if text:
            out.append(text)
            ctx = (ctx + text)[-160:]

    for i in range(0, samples.size, 1024):
        for s in seg.feed(samples[i : i + 1024]):
            handle(s)
    for s in seg.flush():
        handle(s)
    return "".join(out)


def run_one(tr, samples: np.ndarray, context: str) -> str:
    """识别一段并做后处理，幻觉/碎片过滤与正式程序一致。

    Returns:
        归一化后的文本；被过滤掉时返回空串。
    """
    from text_norm import is_hallucination, is_suspicious

    r = tr._engine.transcribe(
        samples, language="zh", no_speech_thold=tr.no_speech_thold,
        initial_prompt=context or None,
    )
    text = tr.normalizer(r.text)
    if not text or is_hallucination(text) or is_suspicious(text, len(samples) * 1000 // 16000):
        return ""
    return text


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", action="store_true", help="跑评测而不是生成测试集")
    ap.add_argument("--model", default="small")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--context", action="store_true", help="段间携带上一段作为上下文")
    args = ap.parse_args()
    if not args.eval:
        return build()
    return evaluate(args.model, not args.cpu, args.context)


if __name__ == "__main__":
    raise SystemExit(main())
