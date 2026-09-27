"""生成中文测试音频（需要 Windows 自带的语音合成）。

仓库里不放中文录音，所以用系统 TTS 合成一段有停顿、有标点、
有不同句式的句子，再重采样到 whisper 需要的 16kHz。

用法: python tools/make_zh_sample.py
"""

from __future__ import annotations

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
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "tests" / "zh16k.wav"
RAW = ROOT / "tests" / "_zh_raw.wav"

TEXT = (
    "今天下午三点，我们开个短会，讨论一下语音转文字工具的优化方向。"
    "主要有三个问题：第一，识别速度要再快一点；"
    "第二，断句要更自然；最后，导出格式要支持字幕。"
)


def synthesize() -> bool:
    """调系统 TTS 念一段中文。返回是否成功。"""
    ps = (
        "Add-Type -AssemblyName System.Speech; "
        "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
        "$v = $s.GetInstalledVoices() | "
        "Where-Object { $_.VoiceInfo.Culture -like 'zh*' } | "
        "Select-Object -First 1; "
        "if (-not $v) { exit 1 }; "
        f"$s.SelectVoice($v.VoiceInfo.Name); "
        f"$s.SetOutputToWaveFile('{RAW}'); "
        f"$s.Speak('{TEXT}'); "
        "$s.SetOutputToNull()"
    )
    r = subprocess.run(["powershell", "-NoProfile", "-Command", ps],
                       capture_output=True, text=True, timeout=180)
    if r.returncode != 0 or not RAW.is_file():
        print("系统里没有中文 TTS 语音，跳过。")
        print("Windows 可以在 设置 > 时间和语言 > 语言 里添加中文语音包。")
        return False
    return True


def resample_to_16k(src: Path, dst: Path) -> None:
    with wave.open(str(src), "rb") as w:
        rate, width, ch = w.getframerate(), w.getsampwidth(), w.getnchannels()
        if width != 2 or ch != 1:
            raise SystemExit(f"需要 16bit 单声道，实际 {width}/{ch}")
        x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32)

    if rate == 16000:
        y = x.astype(np.int16)
    else:
        # 线性插值。测试音频不追求音质，能量级对就行
        t = np.arange(len(x) * 16000 / rate, dtype=np.float32)
        y = np.interp(t, np.arange(len(x), dtype=np.float32), x).astype(np.int16)

    with wave.open(str(dst), "wb") as o:
        o.setnchannels(1)
        o.setsampwidth(2)
        o.setframerate(16000)
        o.writeframes(y.tobytes())


def main() -> int:
    if not synthesize():
        return 0
    resample_to_16k(RAW, OUT)
    RAW.unlink(missing_ok=True)
    with wave.open(str(OUT), "rb") as w:
        dur = w.getnframes() / w.getframerate()
    print(f"已生成 {OUT.name}（{dur:.1f}s, 16kHz 单声道）")
    print(f"文本: {TEXT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
