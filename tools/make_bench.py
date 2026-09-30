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

# 控制台编码：GitHub Actions 的 Windows runner 是 cp1252，编不了中文，
# 脚本里的中文提示会直接抛 UnicodeEncodeError。本地中文系统是 GBK 不会
# 暴露这个问题，所以必须在这里兜住。
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
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
        return len(_canon(self.reference))

    def errors(self, hypothesis: str) -> int:
        """去掉标点后的编辑距离（错字个数）。

        标点不计入：它由规则生成，不反映模型能力；繁体已在上游统一。
        """
        return _levenshtein(_canon(self.reference), _canon(hypothesis))


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


def _cn_to_int(s: str):
    """中文数字转阿拉伯数字。处理到万位，评测语料够用。"""
    digits = {"零": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4,
              "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
    units = {"十": 10, "百": 100, "千": 1000, "万": 10000}
    total = section = number = 0
    for ch in s:
        if ch in digits:
            number = digits[ch]
        elif ch in units:
            u = units[ch]
            if u == 10000:
                total += (section + number or 1) * u
                section = number = 0
            else:
                if number == 0:
                    number = 1          # "十五" -> 15
                section += number * u
                number = 0
    return total + section + number


def _canon(s: str) -> str:
    """归一到可比形式：去标点、繁转简、数字写法统一。

    whisper 会把"百分之十五"归一化成 "95%"、"两千条"成 "2000条"，这是模型
    的正确行为而非错字。之前当错字算，严重高估了错误率——实测同一条样本宽松
    评分 5.6%、严格评分 27.8%，差距几乎全在数字格式上。
    """
    from text_norm import TextNormalizer

    s = _PUNCT.sub("", TextNormalizer()(s))
    s = re.sub(r"百分之([零一二两三四五六七八九十百千万]+|[0-9]+)",
               lambda m: str(_cn_to_int(m.group(1))), s)
    s = re.sub(r"([0-9]+) ?%", lambda m: m.group(1), s)
    s = re.sub(r"([零一二两三四五六七八九十百千万]+)(条|毫秒|秒|分钟|小时|天|个|次)",
               lambda m: str(_cn_to_int(m.group(1))) + m.group(2), s)
    return s


def _synth(text: str, dest: Path) -> bool:
    """合成一段语音。

    必须按句切分逐句合成再拼接。实测 SAPI 对长文本（>30 字）合成不可靠：
    尾部要么没念完（whisper 识别出的内容比参考短一大截），要么音量渐弱到
    模型听不清。之前整个评测体系都建在这上面，测出来的 CER 31.96% 里有一大
    部分是语料缺陷，不是模型能力——把 medium 误判成"无收益"就是这个后果。

    逐句合成每句都很短（<20 字），质量稳定，拼接后内容完整。
    """
    parts = [t for t in re.split(r"[。；！？]", text) if t.strip()]
    chunks = []
    for part in parts:
        part = part.strip("，、 ")
        if part:
            chunks.append(part)
    wavs = []
    try:
        for i, part in enumerate(chunks):
            w = dest.with_name(f"{dest.stem}_p{i}.wav")
            if not _synth_one(part, w):
                return False
            wavs.append(w)
        _join(wavs, dest)
        return True
    finally:
        for w in wavs:
            w.unlink(missing_ok=True)


def _join(parts: list[Path], dest: Path) -> None:
    """按 300ms 静音拼接，并在每段前后留白，避免边界被误切。"""
    gap = np.zeros(int(0.3 * SAMPLE_RATE), dtype=np.int16)
    out = []
    for i, p in enumerate(parts):
        with wave.open(str(p), "rb") as w:
            out.append(np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16))
        if i < len(parts) - 1:
            out.append(gap)
    data = np.concatenate(out)
    with wave.open(str(dest), "wb") as o:
        o.setnchannels(1)
        o.setsampwidth(2)
        o.setframerate(SAMPLE_RATE)
        o.writeframes(data.tobytes())


def _synth_one(text: str, dest: Path) -> bool:
    raw = dest.with_suffix(".raw.wav")
    ps = (
        "Add-Type -AssemblyName System.Speech; "
        "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
        "$v = $s.GetInstalledVoices() | "
        "Where-Object { $_.VoiceInfo.Culture -like 'zh*' } | "
        "Select-Object -First 1; "
        "if (-not $v) { exit 1 }; "
        f"$s.SelectVoice($v.VoiceInfo.Name); $s.Rate=-1; "
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
