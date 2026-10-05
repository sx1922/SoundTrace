"""一次录音会话的纯逻辑。

这一层刻意不依赖 Qt、不起线程、不碰硬件，只吃 int16 音频、吐决策：

    session = RecorderSession(vad_config, live_interval_s)
    session.push_audio(chunk)          # 收音频
    events = session.poll()            # 该识别什么、该显示什么
    events = session.finish()          # 收尾

这样做的唯一理由是**可测**。之前的业务逻辑长在 AudioWorker.run()
那个 while 循环里，开麦、VAD、预览调度、丢帧检查、电平上报混在一起，
测试只能绕开它——于是"点开始录音就崩"这类问题连续三个版本漏到用户手上
（v0.1.0 丢失 MicRecorder.start 方法、v0.1.2 录音线程读 MainWindow 的私有属性）。

线程、硬件、Qt 各自待在各自那一层，这一层只管状态和判断。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from vad import Segmenter, SpeechSegment, VADConfig

# 丢帧累计超过这个值就提醒用户。低于 0.3% 属于时钟抖动，见 MicRecorder.lost_ms。
LOSS_WARN_MS = 200.0
# 两次丢帧检查的最小间隔。lost_ms 是累计量，节流放在这一层做，
# 调用方只管每轮把最新的累计值传进来。
LOSS_REPORT_INTERVAL_S = 1.0


@dataclass
class SessionEvents:
    """一次 poll() 产出的全部决策。线程层只负责把它们发出去。"""

    finals: list[SpeechSegment] = field(default_factory=list)
    partial: SpeechSegment | None = None
    vad_state: str | None = None          # None 表示没变化
    level: float | None = None            # 电平，None 表示不更新
    audio_lost_ms: float = 0.0            # >0 表示这次检测到丢帧

    def __bool__(self) -> bool:
        return bool(self.finals or self.partial is not None
                    or self.vad_state or self.level is not None
                    or self.audio_lost_ms)


class RecorderSession:
    """把音频流转成"该识别什么"。

    单线程使用，不需要加锁——调用方（录音线程）本来就是独占的。
    """

    def __init__(
        self,
        vad_config: VADConfig | None = None,
        live_interval_s: float = 1.2,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.segmenter = Segmenter(vad_config or VADConfig())
        self.live_interval_s = live_interval_s
        self._clock = clock

        self._pending: list[SpeechSegment] = []
        self._last_partial = 0.0
        self._last_state = ""
        self._last_loss_check: float | None = None
        self._loss_reported = False
        self._finished = False

    # -- 输入 --------------------------------------------------------------

    def push_audio(self, samples: np.ndarray) -> None:
        """喂一段音频。

        Segmenter.feed() 返回本轮切出的段，而 take_finalized() 会清空缓冲。
        所以这里必须把返回值存下来——丢掉就等于段凭空消失（第一版正是
        这么写的，导致所有测试都切不出段）。
        """
        if samples is not None and samples.size:
            self._pending.extend(self.segmenter.feed(samples))

    # -- 决策 --------------------------------------------------------------

    def poll(self, level: float | None = None,
             lost_ms: float | None = None) -> SessionEvents:
        """问"现在该干什么"。

        level / lost_ms 都由调用方给出（它才持有 MicRecorder），本层不碰硬件。
        丢帧的节流也在这里做：调用方每 60ms 轮询一次，不节流的话阈值判断
        会被高频触发，而 lost_ms 是累计量、只会越滚越大。
        """
        if self._finished:
            return SessionEvents(level=level)

        ev = SessionEvents(level=level)
        # 段在 push_audio 里攒着，这里一次性交付。不用 take_finalized()
        # ——它会清空 Segmenter 的内部缓冲，和 feed() 的返回值重复。
        ev.finals, self._pending = self._pending, []

        state = self.segmenter.state.value
        if state != self._last_state:
            ev.vad_state = state
            self._last_state = state

        now = self._clock()
        if now - self._last_partial >= self.live_interval_s:
            active = self.segmenter.peek_active()
            if active is not None:
                ev.partial = active
            self._last_partial = now

        # 用 None 而不是 0.0 当初始值：0.0 会让"首次检查"也过节流，
        # 而首次恰恰是最该检查的一次（丢帧可能从第一轮就开始了）。
        if lost_ms is not None and (
            self._last_loss_check is None
            or now - self._last_loss_check >= LOSS_REPORT_INTERVAL_S
        ):
            self._last_loss_check = now
            ev.audio_lost_ms = self.poll_loss(lost_ms)

        return ev

    def poll_loss(self, lost_ms: float) -> float:
        """报告累计丢帧；返回本次应当提示的毫秒数，0 表示不提示。

        丢帧是累计量，只在首次越过阈值时提示一次，否则每轮都弹会烦人。
        """
        if lost_ms < LOSS_WARN_MS or self._loss_reported:
            return 0.0
        self._loss_reported = True
        return lost_ms

    def reset_loss_flag(self) -> None:
        """新的一次丢帧才重新提示。"""
        self._loss_reported = False

    def finish(self) -> SessionEvents:
        """收尾：把没说完的段吐出来，之后不再产出。

        重复调用是安全的（录音线程的 finally 和停止流程都会走到）。
        """
        if self._finished:
            return SessionEvents()
        self._finished = True
        out, self._pending = self._pending, []
        return SessionEvents(finals=out + list(self.segmenter.flush()))

    # -- 状态 --------------------------------------------------------------

    @property
    def finished(self) -> bool:
        return self._finished

    @property
    def state(self) -> str:
        return self.segmenter.state.value

