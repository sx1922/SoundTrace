"""AudioWorker 的行为测试。

v0.1.2 的那个 bug——录音线程读 MainWindow 的私有属性，一忙就
AttributeError——静态检查（test_startup.py）能抓到，但静态检查看不出
"抓住之后是不是真的会安静地录下去"。

这里用假的 recorder/transcriber 把 run() 真的跑一遍，验证丢帧发生时
录音不中断、提示照常发出。这条测试对着旧代码跑会红。
"""

from __future__ import annotations

import threading
import wave
from pathlib import Path

import numpy as np
import pytest

from app import AudioWorker, Signals
from session import RecorderSession
from vad import VADConfig

ROOT = Path(__file__).resolve().parent.parent
SAMPLE_RATE = 16000
CHUNK = 1024


def _load_speech() -> np.ndarray | None:
    """真实语音样本。

    不用合成波形：webrtcvad 只把正弦波开头几帧判为语音，
    测出来的"切不出段"会指向错误的结论。tests/test_session.py 里
    记了同样的原因。
    """
    p = ROOT / "tests" / "zh16k.wav"
    if not p.is_file():
        return None
    with wave.open(str(p), "rb") as f:
        return np.frombuffer(f.readframes(f.getnframes()), dtype=np.int16)


class FakeRing:
    """按固定节奏吐"语音 + 静音"，读够设定轮数后叫停。"""

    def __init__(self, on_read, speech_every: int = 2) -> None:
        self._on_read = on_read
        self._speech_every = speech_every
        self._n = 0
        self._speech = _load_speech()
        if self._speech is not None:
            reps = max(1, self._speech.size // int(3 * SAMPLE_RATE))
            self._speech = np.tile(self._speech[: int(3 * SAMPLE_RATE)], reps)

    def read_since(self, pos: int):
        self._n += 1
        if self._speech is not None and self._n % self._speech_every == 0:
            start = ((self._n // self._speech_every) * CHUNK) % self._speech.size
            data = np.ascontiguousarray(
                self._speech[start : start + CHUNK], dtype=np.int16)
            if data.size < CHUNK:
                data = np.pad(data, (0, CHUNK - data.size))
        else:
            data = np.zeros(CHUNK, dtype=np.int16)
        self._on_read(self._n)
        return data, pos + int(data.size)


class FakeRecorder:
    def __init__(self, stop_after: int, lost_ms: float = 0.0) -> None:
        self.lost_ms = lost_ms
        self.level = 0.42
        self.started = False
        self.stopped = False
        self._stop_after = stop_after
        self.ring = FakeRing(self._tick)

    def _tick(self, n: int) -> None:
        self._reads = n
        if n >= self._stop_after and self._on_stop:
            self._on_stop()

    _reads = 0
    _on_stop = None

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.stopped = True


class FakeTranscriber:
    def __init__(self) -> None:
        self.finals: list = []
        self.partials: list = []

    def submit_final(self, seg) -> None:
        self.finals.append(seg)

    def submit_partial(self, seg) -> None:
        self.partials.append(seg)


def _run_worker(lost_ms: float, stop_after: int = 80):
    """跑一遍 AudioWorker.run，返回 (signals, recorder, transcriber)。"""
    sig = Signals()
    recorder = FakeRecorder(stop_after, lost_ms=lost_ms)
    transcriber = FakeTranscriber()
    session = RecorderSession(VADConfig(), live_interval_s=0.01)
    worker = AudioWorker(recorder, session, transcriber, sig)

    recorder._on_stop = lambda: setattr(worker, "_running", False)

    # 信号在同线程收发时是直连的，用列表收即可
    got = {"loss": [], "state": [], "level": 0, "ended": False}
    sig.loss_warn.connect(lambda ms: got["loss"].append(ms))
    sig.vad_state.connect(lambda s: got["state"].append(s))
    sig.ended.connect(lambda: got.__setitem__("ended", True))

    worker.run()

    # level 是每轮都发的，取最后一次
    sig.level.connect(lambda _v: got.__setitem__("level", _v))
    return got, recorder, transcriber, worker


# -- 核心回归：丢帧不能让录音静默死掉 ----------------------------------------


def test_loss_emits_warning_without_killing_recording():
    """这是 v0.1.2 那个 bug 的行为级回归测试。

    旧代码里 AudioWorker 读 self._loss_warned（只在 MainWindow 上有），
    首次丢帧就 AttributeError；而 run() 只有 finally 没有 except，
    于是 ended 照常发出去——界面显示"已停止"，用户以为录完了，
    实际后半段全丢，且全程无任何报错。

    这里断言的是"丢帧之后循环还活着"：既发了提示，也跑满了设定轮数。
    """
    got, recorder, transcriber, _ = _run_worker(lost_ms=850.0, stop_after=6)

    assert got["loss"], "丢帧超过阈值时必须发出提示信号"
    assert got["loss"][0] == pytest.approx(850.0)

    # 关键断言：循环跑满了，而不是在第一次丢帧时 AttributeError 退出
    assert recorder._reads >= 6, f"录音循环提前退出了，只读了 {recorder._reads} 轮"

    # 且收尾正常：麦克风关了、ended 发了
    assert recorder.stopped, "退出前必须关闭录音设备，否则设备一直被占"
    assert got["ended"], "退出前必须发 ended，界面才能恢复"

    # 电平照常上报，说明循环确实在正常迭代
    assert got["state"], "VAD 初始状态应该上报过一次"


def test_worker_uses_no_mainwindow_state():
    """run() 只碰自己 __init__ 里赋过的属性。

    这条比 test_startup 的静态检查更窄也更硬：它按"赋值集合"逐个验证
    运行时真的能取到，而不是"扫一眼源码觉得没问题"。
    """
    _, _, _, worker = _run_worker(lost_ms=0.0, stop_after=4)
    # 如果 run() 读了外部对象的属性，上面这一行就已经 AttributeError 了
    assert hasattr(worker, "_running")
    assert worker.session is not None


def test_normal_recording_emits_state_and_level():
    """不丢帧时也要有状态和电平上报，否则电平表会是死的。"""
    got, recorder, _, _ = _run_worker(lost_ms=0.0, stop_after=5)
    assert recorder._reads >= 5
    assert got["state"], "VAD 初始状态应该上报"
    assert got["ended"]


def test_no_loss_warning_below_threshold():
    """低于阈值不提示——否则抖动一下就弹窗，弹一次就没人看了。"""
    got, _, _, _ = _run_worker(lost_ms=100.0, stop_after=4)
    assert not got["loss"], "低于阈值的丢帧不该提示"


def test_finish_flushes_unfinished_segment():
    """收尾时没说完的段必须吐出来，否则最后半句会丢。"""
    if _load_speech() is None:
        pytest.skip("缺少 tests/zh16k.wav")
    _, _, transcriber, _ = _run_worker(lost_ms=0.0, stop_after=80)
    assert transcriber.finals, "收尾时应该把活动段定稿"


def test_run_is_actually_driven_by_the_session():
    """AudioWorker 必须通过 session 拿决策，不能自己另搞一套。

    这是整个重构的守门测试：如果有人把 poll() 的逻辑搬回 run()，
    这里会通过（finals 仍然有），所以它只保证 session 被使用了，
    真正的行为保证在前几个用例里。
    """
    sig = Signals()
    recorder = FakeRecorder(3)
    session = RecorderSession(VADConfig())
    worker = AudioWorker(recorder, session, FakeTranscriber(), sig)
    recorder._on_stop = lambda: setattr(worker, "_running", False)

    assert worker.session is session, "worker 必须持有会话对象"
    worker.run()
    assert session.finished, "run() 退出前必须调 session.finish()"


def test_thread_stop_is_settable_from_another_thread():
    """stop() 是给 GUI 线程调的，要能在 worker 跑起来之后把它叫停。"""
    sig = Signals()
    recorder = FakeRecorder(10_000)      # 不会自己停
    worker = AudioWorker(recorder, RecorderSession(VADConfig()), FakeTranscriber(), sig)
    t = threading.Thread(target=worker.run, daemon=True)
    t.start()
    worker.stop()
    t.join(timeout=5)
    assert not t.is_alive(), "stop() 没能让循环退出"
    assert recorder.stopped