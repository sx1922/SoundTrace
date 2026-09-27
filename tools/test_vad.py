"""离线验证 VAD 分段：读一段 wav，走完整分段 -> 识别链路。

用法: python tools/test_vad.py [wav路径]
"""

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
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from transcribe import Transcriber  # noqa: E402
from config import Config  # noqa: E402
from vad import SpeechSegment, VADConfig, Segmenter  # noqa: E402

SAMPLE_RATE = 16000


def read_wav(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as w:
        rate, width, ch = w.getframerate(), w.getsampwidth(), w.getnchannels()
        if width != 2 or ch != 1 or rate != SAMPLE_RATE:
            raise SystemExit(f"需要 16kHz/单声道/16bit，实际 {rate}/{ch}/{width}")
        return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)


def main() -> int:
    wav = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "tests" / "jfk.wav"

    print("== VAD 分段验证 ==")
    samples = read_wav(wav)
    seg = Segmenter(VADConfig.from_config(Config.load()))
    segments: list[SpeechSegment] = []
    # 模拟实时：每次喂 1024 样本，和 PortAudio 的块大小一致
    for i in range(0, samples.size, 1024):
        segments.extend(seg.feed(samples[i : i + 1024]))
    segments.extend(seg.flush())
    print(f"输入 {samples.size / SAMPLE_RATE:.2f}s，切出 {len(segments)} 段：")
    for s in segments:
        print(f"  {s.start_sample / SAMPLE_RATE:6.2f}s -> {s.end_sample / SAMPLE_RATE:6.2f}s "
              f"({s.duration_ms:5d}ms)")
    if not segments:
        print("失败：一段都没切出来")
        return 1

    print("\n== 识别验证 ==")
    t = Transcriber(ROOT / "models" / "ggml-small.bin", n_threads=10, language="en",
                    verbose=True)
    v = t.load_blocking()
    print(f"whisper.cpp {v}")
    for s in segments:
        r = t._engine.transcribe(s.samples, language="en")
        print(f"  [{s.duration_ms:5d}ms] RTF={r.rtf:.2f}  {r.text}")
    t.stop()

    print("\n== 静音幻觉检查 ==")
    # 纯静音最容易触发幻觉，VAD 应该根本切不出段
    silence = np.zeros(SAMPLE_RATE * 5, dtype=np.int16)
    seg2 = Segmenter(VADConfig.from_config(Config.load()))
    fake = seg2.feed(silence)
    fake.extend(seg2.flush())
    print(f"5s 纯静音 -> 切出 {len(fake)} 段（期望 0）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
