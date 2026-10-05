"""Transcriber 的线程与队列测试。

v0.1.3 之前 transcribe.py 零测试覆盖，而 P0-2 就住在里面：
start() 没有重入保护、录音结束也不停线程，第二次录音起两条线程
并发操作同一个 whisper context。这几条测试就是防它回来的。

引擎是假的——真 whisper 需要模型文件，而且并发崩溃本来也不该
在单测里真的发生一次。
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import numpy as np
import pytest

from transcribe import Output, Transcriber
from vad import SpeechSegment

SAMPLE_RATE = 16000


def _seg(sec: float = 1.0) -> SpeechSegment:
    n = int(sec * SAMPLE_RATE)
    return SpeechSegment(
        samples=np.zeros(n, dtype=np.int16),
        start_sample=0,
        end_sample=n,
    )


class FakeResult:
    def __init__(self, text: str = "测试文本") -> None:
        self.text = text
        self.elapsed_s = 0.01
        self.rtf = 0.1


class FakeEngine:
    """记录自己被哪些线程调用过。

    并发调用会被记为并发——这正是我们要断言不发生的事。
    """

    def __init__(self, delay: float = 0.0, text: str = "测试文本") -> None:
        self.delay = delay
        self.text = text
        self.calls = 0
        self.lock = threading.Lock()
        self.concurrent = 0
        self.max_concurrent = 0
        self.thread_names: list[str] = []
        self.freed = False

    def transcribe(self, samples, language=None, no_speech_thold=None):
        with self.lock:
            self.calls += 1
            self.concurrent += 1
            self.max_concurrent = max(self.max_concurrent, self.concurrent)
            self.thread_names.append(threading.current_thread().name)
        try:
            if self.delay:
                time.sleep(self.delay)
            return FakeResult(self.text)
        finally:
            with self.lock:
                self.concurrent -= 1

    def free(self) -> None:
        self.freed = True


def _make(engine: FakeEngine, **kw) -> Transcriber:
    t = Transcriber(Path("fake-model.bin"), **kw)
    t._engine = engine          # 绕过耗时的 load_blocking()
    return t


# -- 重入保护：P0-2 的正面测试 ----------------------------------------------


def test_second_start_does_not_spawn_second_thread():
    """连续两次 start() 只能有一条推理线程。

    旧代码无条件建新线程并覆盖 self._thread，上一条线程引用一丢就再也
    停不掉，而 _stop 被 clear() 过，它会一直活到下次 stop()——
    两条线程同时对同一个 whisper context 调 whisper_full，直接段错误。
    """
    engine = FakeEngine()
    t = _make(engine)
    got: list[Output] = []

    t.start(got.append, lambda _e: None)
    first = t._thread
    t.start(got.append, lambda _e: None)     # 模拟第二次录音

    assert t._thread is first, "第二次 start() 不该另起一条线程"
    assert t._thread is not None and t._thread.is_alive()

    t.stop_worker()
    engine.free()


def test_engine_never_called_concurrently():
    """真正要守的不变量：引擎不能被两个线程同时调用。

    上一条只查线程对象，这条查实际行为——即使将来有人换种方式又搞出
    并发，这里也会红。
    """
    engine = FakeEngine(delay=0.02)
    t = _make(engine)
    t.start(lambda _o: None, lambda _e: None)
    first = t._thread
    t.start(lambda _o: None, lambda _e: None)

    for i in range(6):
        t.submit_final(_seg())

    t.wait_idle(timeout=10)
    t.stop_worker(timeout=5)

    assert engine.max_concurrent <= 1, (
        f"引擎被并发调用了（峰值 {engine.max_concurrent}），"
        "whisper context 不是线程安全的"
    )
    assert first is not None


def test_stop_worker_keeps_engine_loaded():
    """stop_worker 只停线程，模型必须留着。

    录音结束走的是这条路径——如果它把引擎也卸载了，每录一次就要
    重新加载一次模型（几秒），等于把识别体验做废了。
    """
    engine = FakeEngine()
    t = _make(engine)
    t.start(lambda _o: None, lambda _e: None)
    assert t.stop_worker(timeout=5) is True
    assert t._engine is engine, "stop_worker 不该卸载引擎"
    assert not engine.freed


def test_start_after_stop_worker_spawns_fresh_thread():
    """上一条线程真退出后，下一次 start() 应当能正常起新线程。

    重入保护不能写成一刀切的"永不新建"。
    """
    engine = FakeEngine()
    t = _make(engine)
    t.start(lambda _o: None, lambda _e: None)
    first = t._thread
    t.stop_worker(timeout=5)

    t.start(lambda _o: None, lambda _e: None)
    assert t._thread is not first
    t.stop_worker(timeout=5)


def test_start_without_engine_raises():
    t = Transcriber(Path("fake-model.bin"))
    with pytest.raises(RuntimeError):
        t.start(lambda _o: None, lambda _e: None)


# -- stop() 的释放时机 ------------------------------------------------------


def test_stop_frees_engine_when_thread_exits():
    engine = FakeEngine()
    t = _make(engine)
    t.start(lambda _o: None, lambda _e: None)
    t.stop(timeout=5)
    assert engine.freed, "线程退出后 stop() 应该释放引擎"
    assert t._engine is None


def test_stop_does_not_free_engine_while_thread_busy():
    """线程还在 whisper 里跑时，不能释放引擎。

    这就是 use-after-free：join 超时被忽略的话，whisper_free() 会去
    free 别人正在读的 context，表现为随机段错误，很难查。
    """
    engine = FakeEngine(delay=1.0)
    t = _make(engine)
    t.start(lambda _o: None, lambda _e: None)
    t.submit_final(_seg())

    # 等到线程确实进了引擎
    deadline = time.monotonic() + 2
    while engine.calls == 0 and time.monotonic() < deadline:
        time.sleep(0.01)

    t.stop(timeout=0.05)          # 故意超时

    assert not engine.freed, "线程还在推理时释放引擎 = use-after-free"
    assert t._engine is not None

    t.stop(timeout=10)            # 这次让它正常退出并释放
    assert engine.freed


# -- wait_idle 的竞态 -------------------------------------------------------


def test_wait_idle_waits_for_inflight_task():
    """最后一个任务出队后、_inflight 置位前有个窗口。

    旧代码在这窗口里 busy() 返回 False，wait_idle 立刻返回，
    最后一句会在界面说"已停止"之后才吐出来。
    """
    engine = FakeEngine(delay=0.3)
    t = _make(engine)
    t.start(lambda _o: None, lambda _e: None)
    t.submit_final(_seg())

    t.wait_idle(timeout=10)
    # wait_idle 返回时，引擎必须已经真的跑完了
    assert engine.calls >= 1
    assert engine.concurrent == 0, "wait_idle 返回时仍有任务在飞"
    t.stop_worker(timeout=5)


def test_inflight_cleared_after_filtered_output():
    """被幻觉/碎片过滤掉的段也必须清掉 _inflight。

    _process 里有好几条提前 return；漏掉任何一条，_inflight 会永久卡在
    True，之后 wait_idle() 再也等不满、界面永远停在"正在处理…��。
    """
    engine = FakeEngine(text="谢谢观看")     # 典型幻觉短语
    t = _make(engine)
    t.start(lambda _o: None, lambda _e: None)
    for _ in range(3):
        t.submit_final(_seg())

    t.wait_idle(timeout=10)
    assert not t.busy(), f"过滤掉的段没清 _inflight（busy={t.busy()}）"
    t.stop_worker(timeout=5)


def test_busy_is_false_after_drain():
    engine = FakeEngine(delay=0.05)
    t = _make(engine)
    t.start(lambda _o: None, lambda _e: None)
    for _ in range(4):
        t.submit_final(_seg())
    assert t.busy()
    t.wait_idle(timeout=10)
    assert not t.busy()
    t.stop_worker(timeout=5)


def test_outputs_are_delivered():
    engine = FakeEngine(text="你好世界")
    t = _make(engine)
    got: list[Output] = []
    t.start(got.append, lambda _e: None)
    t.submit_final(_seg())

    deadline = time.monotonic() + 5
    while not got and time.monotonic() < deadline:
        time.sleep(0.01)
    t.stop_worker(timeout=5)

    assert got, "定稿结果没有送达回调"
    assert got[0].kind == "final"
    assert got[0].result.text == "你好世界"
    assert engine.thread_names and engine.thread_names[0] == "transcribe"