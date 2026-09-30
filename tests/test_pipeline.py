"""识别链路的单元测试：VAD 分段、文本后处理、音频缓冲。

这些是程序的核心，但之前只靠端到端跑一遍看输出对不对——出问题很难定位到
具体是哪一层。这里用合成的确定性输入覆盖边界条件：静音、噪声、超长段、
损坏输入等。
"""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
import pytest

from audio import BUFFER_SECONDS, RingBuffer
from text_norm import TextNormalizer, is_hallucination, is_suspicious
from vad import FRAME_SAMPLES, SAMPLE_RATE, Segmenter, VADConfig

ROOT = Path(__file__).resolve().parent.parent


def _silence(sec: float) -> np.ndarray:
    return np.zeros(int(sec * SAMPLE_RATE), dtype=np.int16)


def _tone(sec: float, freq: float = 220.0, amp: float = 0.3) -> np.ndarray:
    t = np.arange(int(sec * SAMPLE_RATE)) / SAMPLE_RATE
    return (np.sin(2 * np.pi * freq * t) * amp * 32767).astype(np.int16)


def _feed(seg: Segmenter, audio: np.ndarray, chunk: int = 1024) -> list:
    out = []
    for i in range(0, audio.size, chunk):
        out.extend(seg.feed(audio[i : i + chunk]))
    return out


# -- 文本后处理 -------------------------------------------------------------


class TestTextNormalizer:
    @pytest.fixture
    def norm(self):
        return TextNormalizer()

    def test_traditional_to_simplified(self, norm):
        assert norm("今天下午三點") == "今天下午三点"
        assert norm("我們開個短會") == "我们开个短会"

    def test_halfwidth_punct_in_chinese(self, norm):
        assert norm("第一,识别速度") == "第一，识别速度"
        assert norm("他说:今天开会") == "他说：今天开会"

    def test_english_punctuation_untouched(self, norm):
        """英文句子里的逗号不能被转成全角。"""
        s = "Ask not what your country can do for you, then go."
        assert norm(s) == s

    def test_empty_and_whitespace(self, norm):
        assert norm("") == ""
        assert norm("   ") == ""

    def test_mixed_script(self, norm):
        assert norm("GPU 加速 95%") == "GPU 加速 95%"


class TestHallucination:
    @pytest.mark.parametrize("text", [
        "(字幕:貝爾)", "（字幕：贝尔）", "（音乐）", "(音樂)",
        "Thank you for watching", "Music", "APPLAUSE",
        "谢谢观看", "订阅，转发，打赏", "...", "（）",
    ])
    def test_detects(self, text):
        n = TextNormalizer()
        assert is_hallucination(n(text)), f"应判为幻觉: {text!r}"

    @pytest.mark.parametrize("text", [
        "今天下午三点", "我们开个短会，讨论一下优化方向。",
        "第一，识别速度要再快一点",
        "Ask not what your country can do for you",
        "括号在句中(测试)不算幻觉",
    ])
    def test_real_speech_passes(self, text):
        n = TextNormalizer()
        assert not is_hallucination(n(text)), f"不应判为幻觉: {text!r}"


class TestSuspicious:
    def test_short_audio_short_text_is_fine(self):
        """用户说单个"好"，音频也短，不该被当成碎片。"""
        assert not is_suspicious("好", duration_ms=300)

    def test_long_audio_garbage_is_flagged(self):
        """录了 1.2 秒只出 "TR"，是典型的截断残段。"""
        assert is_suspicious("TR", duration_ms=1200)

    def test_empty_is_flagged(self):
        assert is_suspicious("", duration_ms=1000)
        assert is_suspicious("   ", duration_ms=1000)

    def test_normal_text_passes(self):
        assert not is_suspicious("这是一句正常的话", duration_ms=3000)


# -- VAD 分段 ---------------------------------------------------------------


def _speech() -> np.ndarray:
    """真实语音片段。

    之前用纯正弦波当测试信号，结果 5 条分段测试全挂——webrtcvad 根本不把
    220Hz 正弦波当语音，检不出语音自然切不出段。真实语音才有意义。
    """
    p = ROOT / "tests" / "fixtures" / "speech.wav"
    if not p.is_file():
        pytest.skip("缺少 tests/fixtures/speech.wav")
    with wave.open(str(p)) as w:
        return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)


def _gap(sec: float) -> np.ndarray:
    return _silence(sec)


class TestSegmenter:
    def test_silence_yields_nothing(self):
        seg = Segmenter(VADConfig())
        out = _feed(seg, _silence(5.0))
        out.extend(seg.flush())
        assert out == [], "5 秒静音不应产生任何语音段"

    def test_speech_detected(self):
        seg = Segmenter(VADConfig())
        out = _feed(seg, _speech())
        out.extend(seg.flush())
        assert out, "真实语音应该被检出"

    def test_long_gap_splits_into_two(self):
        seg = Segmenter(VADConfig(end_frames=30))
        out = _feed(seg, np.concatenate([_speech(), _gap(1.5), _speech()]))
        out.extend(seg.flush())
        assert len(out) >= 2, "两段语音中间的长静音应切成两段"

    def test_short_pause_merges(self):
        """短暂停顿小于阈值时不该把一句话切开。"""
        seg = Segmenter(VADConfig(end_frames=60))
        out = _feed(seg, np.concatenate([_speech(), _gap(0.3), _speech()]))
        out.extend(seg.flush())
        assert len(out) == 1, f"0.3 秒停顿不该切段，实际切了 {len(out)} 段"

    def test_max_segment_forces_split(self):
        """超长段要强制切开，否则 whisper 超出 30 秒窗口会出问题。"""
        seg = Segmenter(VADConfig(max_segment_s=1.0))
        out = _feed(seg, _speech())
        out.extend(seg.flush())
        assert len(out) >= 2, "4 秒语音在 1 秒上限下应切多段"
        for s in out:
            assert s.duration_ms <= 1800, f"单段 {s.duration_ms}ms 超上限太多"

    def test_flush_emits_unfinished(self):
        """录音停止时未说完的段必须定稿，否则最后半句会丢。"""
        seg = Segmenter(VADConfig())
        _feed(seg, _speech()[: 2 * SAMPLE_RATE])   # 故意不给尾部静音
        out = seg.flush()
        assert out, "未说完的段在 flush 时必须定稿"

    def test_empty_input_is_safe(self):
        seg = Segmenter(VADConfig())
        assert seg.feed(np.zeros(0, dtype=np.int16)) == []
        assert seg.feed(None) == []

    def test_irregular_chunk_sizes(self):
        """block 大小不是 20ms 整数倍时也要正确处理。"""
        seg = Segmenter(VADConfig())
        audio = np.concatenate([_speech(), _gap(1.2)])
        out = []
        for size in (333, 777, 1000, 555):
            for i in range(0, audio.size, size):
                out.extend(seg.feed(audio[i : i + size]))
        out.extend(seg.flush())
        assert out, "不规则分块也应该正常切段"

    def test_peek_returns_independent_copy(self):
        """peek 必须返回独立数组，否则推理线程会读到正在被追加的缓冲。"""
        seg = Segmenter(VADConfig(min_preview_ms=100))
        _feed(seg, _speech())
        a = seg.peek_active()
        assert a is not None, "1.2 秒语音后应该有可预览的段"
        before = a.samples.copy()
        _feed(seg, _speech())
        assert np.array_equal(a.samples, before), "peek 的结果被后续帧改动了"

    def test_very_short_noise_dropped(self):
        seg = Segmenter(VADConfig(min_segment_ms=2000))
        out = _feed(seg, _speech()[: SAMPLE_RATE])
        out.extend(seg.flush())
        assert out == [], "1 秒语音低于 2 秒下限应被丢弃"


# -- 环形缓冲 ---------------------------------------------------------------


class TestRingBuffer:
    def test_write_then_read(self):
        rb = RingBuffer(seconds=2)
        data = _tone(0.5)
        rb.write(data)
        got = rb.read(data.size)
        assert np.array_equal(got, data)

    def test_read_since_returns_only_new(self):
        rb = RingBuffer(seconds=5)
        a, pos = rb.read_since(0)
        assert a.size == 0
        data = _tone(0.3)
        rb.write(data)
        a, pos2 = rb.read_since(pos)
        assert np.array_equal(a, data)
        assert pos2 == data.size

    def test_overflow_keeps_newest(self):
        """写超容量时应保留最新的，丢最旧的。"""
        rb = RingBuffer(seconds=1)
        old = _tone(0.8, freq=100.0)
        new = _tone(0.8, freq=800.0)
        rb.write(old)
        rb.write(new)
        got = rb.read(SAMPLE_RATE)
        assert got.size == SAMPLE_RATE
        # 尾部应该全是 new 的内容
        tail = got[-SAMPLE_RATE // 2 :]
        ref = new[-SAMPLE_RATE // 2 :]
        assert np.array_equal(tail, ref)

    def test_write_larger_than_capacity(self):
        rb = RingBuffer(seconds=1)
        big = _tone(3.0)
        rb.write(big)
        got = rb.read(SAMPLE_RATE)
        assert got.size == SAMPLE_RATE
        assert np.array_equal(got, big[-SAMPLE_RATE:])

    def test_read_more_than_available(self):
        rb = RingBuffer(seconds=2)
        data = _tone(0.5)
        rb.write(data)
        got = rb.read(data.size * 3)
        assert got.size == data.size, "请求超过存量时应返回全部而不是报错"

    def test_empty_read(self):
        rb = RingBuffer(seconds=1)
        assert rb.read(100).size == 0

    def test_reset_clears(self):
        rb = RingBuffer(seconds=2)
        rb.write(_tone(0.5))
        rb.reset()
        assert rb.read(1000).size == 0
        assert rb.total_written == 0

    def test_default_capacity(self):
        assert BUFFER_SECONDS >= 30, "缓冲至少要 30 秒，够覆盖一段长语音"


# -- 真实样本回归 -----------------------------------------------------------


class TestRealAudio:
    """用仓库里的真实录音做回归，防止改动破坏既有行为。"""

    def test_silence_file(self):
        p = ROOT / "tests" / "room.wav"
        if not p.is_file():
            pytest.skip("需要先录一段房间噪声")
        with wave.open(str(p)) as w:
            x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
        seg = Segmenter(VADConfig())
        out = _feed(seg, x)
        out.extend(seg.flush())
        # 不强制要求 0 段（环境噪声会误判），但单段不能离谱地长
        for s in out:
            assert s.duration_ms <= 20000, f"段长 {s.duration_ms}ms 异常"

    def test_english_sample_transcribes(self):
        p = ROOT / "tests" / "jfk.wav"
        if not p.is_file():
            pytest.skip("需要 tests/jfk.wav")
        from transcribe import Transcriber
        from whisper_bridge import WhisperEngine

        eng = WhisperEngine(ROOT / "models" / "ggml-small.bin", use_gpu=False, n_threads=4)
        if not eng.model_path.is_file():
            pytest.skip("没有模型文件")
        eng.load()
        eng.silence_logs()
        with wave.open(str(p)) as w:
            x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
        r = eng.transcribe(x, language="en")
        eng.free()
        assert "country" in r.text.lower(), f"识别结果异常: {r.text!r}"
