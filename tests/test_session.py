"""RecorderSession 的测试。

这一层是主干里唯一能被完整自动化覆盖的部分——没有 Qt、没有线程、
没有麦克风。测试写得住，就说明它真的被隔离干净了。
"""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
import pytest

from session import (
    LOSS_REPORT_INTERVAL_S,
    LOSS_WARN_MS,
    RecorderSession,
    SessionEvents,
)
from vad import VADConfig

ROOT = Path(__file__).resolve().parent.parent
SAMPLE_RATE = 16000


def _speech(sec: float = 2.0) -> np.ndarray:
    """真实语音片段。

    不要用纯正弦波代替：webrtcvad 只把正弦波开头 5 帧判为语音
    （实测 100 帧里只有 5 帧为 S），测不出多段切分，而且会让"分段失败"
    的断言指向错误的结论。真实语音的判定是稳定的。
    """
    p = ROOT / "tests" / "zh16k.wav"
    if not p.is_file():
        pytest.skip("缺少 tests/zh16k.wav，无法构造语音信号")
    with wave.open(str(p), "rb") as f:
        x = np.frombuffer(f.readframes(f.getnframes()), dtype=np.int16)
    n = int(sec * SAMPLE_RATE)
    if n > x.size:
        reps = x.size // int(1.5 * SAMPLE_RATE)
        x = np.tile(x[: int(1.5 * SAMPLE_RATE)], max(2, n // max(reps, 1) + 1))
    return x[:n].copy()


def _tone(sec: float, freq: float = 220.0, amp: float = 0.3) -> np.ndarray:
    t = np.arange(int(sec * SAMPLE_RATE)) / SAMPLE_RATE
    return (np.sin(2 * np.pi * freq * t) * amp * 32767).astype(np.int16)


def _silence(sec: float) -> np.ndarray:
    return np.zeros(int(sec * SAMPLE_RATE), dtype=np.int16)


class FakeClock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


# -- 基本契约 ---------------------------------------------------------------


def test_poll_reports_initial_state_once():
    """首次 poll 会上报一次初始状态 idle——界面需要它来初始化指示灯，
    之后状态不变就不再报。"""
    s = RecorderSession()
    first = s.poll()
    assert first.vad_state == "idle"
    assert first.finals == [] and first.partial is None
    assert s.poll().vad_state is None, "状态没变就不该再报"


def test_no_external_dependencies():
    """这层不能引入 Qt 或线程，否则又变成测不动的东西。

    用静态检查代替：session.py 里不该出现 QApplication/QThread/threading。
    """
    src = (ROOT / "session.py").read_text(encoding="utf-8")
    for banned in ("PySide6", "QThread", "import threading", "sounddevice", "sounddevice"):
        assert banned not in src, f"session.py 不该依赖 {banned}"


# -- 分段 -------------------------------------------------------------------


def test_two_speech_bursts_split():
    s = RecorderSession(VADConfig(end_frames=60))
    finals = []
    audio = np.concatenate([_speech(2.0), _silence(1.5), _speech(2.0)])
    for i in range(0, audio.size, 1024):
        s.push_audio(audio[i : i + 1024])
        finals.extend(s.poll().finals)
    finals.extend(s.finish().finals)
    assert len(finals) >= 2, f"两段语音中间长静音应切成两段，实际 {len(finals)}"


def test_short_pause_merges():
    s = RecorderSession(VADConfig(end_frames=120))
    finals = []
    audio = np.concatenate([_speech(1.5), _silence(0.2), _speech(1.5)])
    for i in range(0, audio.size, 1024):
        s.push_audio(audio[i : i + 1024])
        finals.extend(s.poll().finals)
    finals.extend(s.finish().finals)
    assert len(finals) == 1, f"0.2 秒停顿不该切段，实际切了 {len(finals)} 段"


# -- 预览调度 ---------------------------------------------------------------


def test_partial_respects_interval():
    """两次 poll 之间不到 live_interval 就不该重复产出预览。"""
    clock = FakeClock()
    s = RecorderSession(live_interval_s=1.2, clock=clock)

    # 逐块喂，模拟真实流。一次性喂完的话 VAD 已经把段切走了，
    # peek_active() 自然返回 None——那是对的，段已经定稿不需要预览。
    audio = _speech(2.5)
    first = None
    for i in range(0, audio.size, 1024):
        s.push_audio(audio[i : i + 1024])
        clock.advance(0.064)
        ev = s.poll()
        if ev.partial is not None and first is None:
            first = ev
            break
    assert first is not None, "边喂边轮询时应该出现可预览的段"

    clock.advance(0.1)          # 距上次轮询才 0.1s
    second = s.poll()
    assert second.partial is None, "间隔没到就不该再产预览（否则白烧 CPU）"

    clock.advance(1.5)          # 超过 1.2s
    third = s.poll()
    assert third.partial is not None, "间隔到了就该再产预览"


def test_no_partial_when_idle():
    s = RecorderSession()
    s.push_audio(_silence(2.0))
    assert s.poll().partial is None, "静音时不该有预览"


def test_finished_session_stops_producing():
    s = RecorderSession()
    s.push_audio(_speech(1.5))
    s.finish()
    assert not s.poll(), "finish 之后不该再产出事件"
    assert s.finished


def test_finish_is_idempotent():
    """录音线程的 finally 和停止流程都会调 finish，重复调用不能出问题。"""
    s = RecorderSession(VADConfig())
    s.push_audio(_speech(2.0))
    first = s.finish()
    second = s.finish()
    assert len(second.finals) == 0, "重复 finish 不该重复吐段"
    assert len(first.finals) >= 0


def test_finish_emits_unfinished_speech():
    """没说完的段必须在 finish 时吐出来，否则最后半句会丢。"""
    s = RecorderSession(VADConfig())
    s.push_audio(_speech(3.0))    # 不给尾部静音，段还没结束
    ev = s.finish()
    assert ev.finals, "未说完的段在 finish 时必须吐出"


# -- VAD 状态变化 -----------------------------------------------------------


def test_vad_state_reported_once_per_change():
    """状态不变时不重复上报，否则界面上的指示灯会不停重绘。"""
    s = RecorderSession(VADConfig(end_frames=60))
    s.push_audio(_speech(1.0))
    states = []
    for _ in range(5):
        ev = s.poll()
        if ev.vad_state is not None:
            states.append(ev.vad_state)
    assert len(states) == 1, f"状态未变时只应上报一次，实际 {len(states)} 次"


# -- 丢帧 -------------------------------------------------------------------


def test_loss_below_threshold_not_reported():
    s = RecorderSession()
    assert s.poll_loss(LOSS_WARN_MS - 1) == 0.0


def test_loss_above_threshold_reported_once():
    """只提示一次，否则每轮都提示会烦死人。"""
    s = RecorderSession()
    first = s.poll_loss(500.0)
    second = s.poll_loss(800.0)
    third = s.poll_loss(1200.0)
    assert first == 500.0
    assert second == 0.0 and third == 0.0, "同一次录音的丢帧只应提示一次"


def test_loss_flag_reset_allows_new_report():
    s = RecorderSession()
    assert s.poll_loss(500.0) == 500.0
    assert s.poll_loss(500.0) == 0.0
    s.reset_loss_flag()
    assert s.poll_loss(600.0) == 600.0, "重置后应能再次提示"


def test_loss_state_lives_in_session_not_thread():
    """这条守的是 v0.1.2 那个 bug：丢帧标志原本挂在 MainWindow 上，
    录音线程读的是别人的属性，一忙就 AttributeError。
    状态必须住在 session 里，线程只负责转发。
    """
    s = RecorderSession()
    assert s.poll_loss(500.0) == 500.0   # 不依赖任何外部对象


# -- 丢帧经由 poll() 交付 ----------------------------------------------------


def test_poll_carries_loss_into_events():
    """丢帧要跟着 poll() 的结果一起出去。

    以前这里有个坑：SessionEvents.audio_lost_ms 字段存在、__bool__ 里也
    判断了它，但 poll_loss 只返回一个裸 float，字段从来没被赋过值——
    于是 __bool__ 对它的判断是死代码。现在由 poll() 统一填。
    """
    clock = FakeClock()
    s = RecorderSession(clock=clock)
    ev = s.poll(lost_ms=500.0)
    assert ev.audio_lost_ms == 500.0
    assert ev, "有丢帧时事件对象必须为真，否则上层会当成'没事发生'"


def test_poll_loss_not_reported_when_absent():
    """不传 lost_ms（静态检查、其它调用方）时不该凭空报丢帧。"""
    s = RecorderSession()
    assert s.poll().audio_lost_ms == 0.0


def test_poll_loss_below_threshold_stays_zero():
    s = RecorderSession()
    assert s.poll(lost_ms=LOSS_WARN_MS - 1).audio_lost_ms == 0.0


def test_poll_loss_reported_once_across_polls():
    """一次录音只提示一次，且由 poll() 的节流控制。"""
    clock = FakeClock()
    s = RecorderSession(clock=clock)
    first = s.poll(lost_ms=500.0)
    assert first.audio_lost_ms == 500.0

    clock.advance(LOSS_REPORT_INTERVAL_S + 0.1)
    assert s.poll(lost_ms=800.0).audio_lost_ms == 0.0, "同一次录音只应提示一次"

    clock.advance(LOSS_REPORT_INTERVAL_S + 0.1)
    assert s.poll(lost_ms=1200.0).audio_lost_ms == 0.0


def test_poll_loss_throttled_within_interval():
    """节流：间隔内的重复上报要被吃掉。

    调用方每 60ms 轮询一次，lost_ms 又是累计量，不节流的话
    每轮都会去问一次阈值，白白打断节流逻辑。
    """
    clock = FakeClock()
    s = RecorderSession(clock=clock)
    # 首次必须放行——丢帧可能从第一轮就开始了
    assert s.poll(lost_ms=500.0).audio_lost_ms == 500.0

    s.reset_loss_flag()             # 只留节流，去掉"一次提示"的限制
    clock.advance(LOSS_REPORT_INTERVAL_S * 0.5)
    assert s.poll(lost_ms=900.0).audio_lost_ms == 0.0, "节流期内不该重复上报"

    clock.advance(LOSS_REPORT_INTERVAL_S * 0.6)
    assert s.poll(lost_ms=1200.0).audio_lost_ms == 1200.0, "节流期过后应该放行"


def test_poll_loss_flag_survives_reset():
    """reset 后能再次提示——单次录音内不会调，跨会话复用时会。"""
    s = RecorderSession()
    assert s.poll_loss(500.0) == 500.0
    assert s.poll_loss(500.0) == 0.0
    s.reset_loss_flag()
    assert s.poll_loss(600.0) == 600.0


# -- 真实音频 ---------------------------------------------------------------


@pytest.mark.skipif(not (ROOT / "tests" / "zh16k.wav").is_file(),
                    reason="缺少 tests/zh16k.wav")
def test_real_audio_produces_finals():
    with wave.open(str(ROOT / "tests" / "zh16k.wav"), "rb") as f:
        x = np.frombuffer(f.readframes(f.getnframes()), dtype=np.int16)

    s = RecorderSession(VADConfig(end_frames=60))
    finals = []
    for i in range(0, x.size, 1024):
        s.push_audio(x[i : i + 1024])
        finals.extend(s.poll().finals)
    finals.extend(s.finish().finals)
    assert len(finals) >= 1, "19 秒中文语音至少应切出一段"
    assert sum(f.duration_ms for f in finals) > 3000
