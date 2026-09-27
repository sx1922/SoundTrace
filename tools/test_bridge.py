"""验证 ctypes -> whisper.dll 绑定是否正确。

用法: python tools/test_bridge.py [wav路径]
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from whisper_bridge import WhisperEngine  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def read_wav(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as w:
        assert w.getsampwidth() == 2, "只支持 16bit wav"
        assert w.getnchannels() == 1, "只支持单声道"
        rate = w.getframerate()
        assert rate == 16000, f"需要 16kHz，实际 {rate}"
        data = w.readframes(w.getnframes())
    return np.frombuffer(data, dtype=np.int16)


def main() -> int:
    wav = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "tests" / "jfk.wav"
    model = ROOT / "models" / "ggml-small.bin"

    samples = read_wav(wav)
    print(f"音频: {wav.name}  {samples.size / 16000:.2f}s  {samples.size} samples")

    engine = WhisperEngine(model, use_gpu=False, n_threads=10, verbose=True)
    engine.load()
    engine.silence_logs()
    print(f"whisper.cpp 版本: {engine.version}")
    print(f"模型: {model.name}  use_gpu={engine.use_gpu}  threads={engine.n_threads}")

    result = engine.transcribe(samples, language="en")
    print("-" * 60)
    print("识别结果:", repr(result.text))
    print(f"lang_id={result.lang_id}  耗时={result.elapsed_s:.2f}s  "
          f"音频={result.audio_s:.2f}s  RTF={result.rtf:.3f}")
    for seg in result.segments:
        print(f"  [{seg.t0_ms:6d} -> {seg.t1_ms:6d}] "
              f"no_speech={seg.no_speech_prob:.3f}  {seg.text}")
    print("-" * 60)

    expected = "ask not what your country can do for you"
    ok = expected in result.text.lower()
    print("绑定验证:", "通过" if ok else "失败（结果与预期不符）")
    engine.free()
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
