"""语音端点检测与分段。

为什么必须有 VAD：whisper 对纯静音会"脑补"出内容（经典输出是
"Thank you for watching!"）。麦克风场景下只要有一丝环境噪音，
不过滤就会在正文里不断冒出来。

状态机：
    IDLE ──连续 N 帧语音──> SPEECH ──连续 M 帧静音──> TAIL ──> 定稿
     ^                      |                          |
     └──────────────────────┴────超时强制切分──────────┘

段首不会切掉第一个字：开段时把之前攒的 preroll_ms 预缓冲一起带上。
段尾不会切掉最后一个字：要连续静音 end_frames 帧才收段，尾字落在静音之前。
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
import webrtcvad

SAMPLE_RATE = 16000
FRAME_MS = 20  # webrtcvad 只接受 10/20/30ms
FRAME_SAMPLES = SAMPLE_RATE * FRAME_MS // 1000  # 320


class State(Enum):
    IDLE = "idle"
    SPEECH = "speech"
    TAIL = "tail"


@dataclass
class VADConfig:
    # 0 最宽松（几乎所有声音都算语音），3 最严格。麦克风建议 2。
    aggressiveness: int = 2
    # 连续多少帧判定为语音才触发开段，3 帧 = 60ms
    start_frames: int = 3
    # 连续多少帧静音才判定说话结束。60 帧 = 1200ms。
    # 这个值对准确率的影响最大（段切碎了就丢上下文），实测 600ms 的
    # 字符错误率 36.5%，1200ms 降到 32.0%。默认值和 config.silence_ms
    # 保持一致，改动请走 from_config()，别在这里另设一份。
    end_frames: int = 60
    # 判定开段时向前回溯多少毫秒，避免吃掉第一个字
    preroll_ms: int = 300
    # 单段最长多少秒，超时强制切分
    max_segment_s: float = 15.0
    # 短于这个长度的段直接丢弃，多半是咳嗽或键盘声
    min_segment_ms: int = 500
    # 语音段积累到这个长度之前不做实时预览（预览对上下文不足的片段容易猜错）
    min_preview_ms: int = 1200
    # 额外的能量下限，叠在 webrtcvad 之上做双保险。
    # 只当"绝对静音"用，设得远低于正常说话的电平——笔记本内置麦克风
    # 在安静房间里峰值也常常只有 0.01~0.03，门限高了会把轻声整段挡掉。
    energy_floor: float = 0.0015

    @classmethod
    def from_config(cls, cfg) -> "VADConfig":
        """从 Config 构造，避免断句阈值散落在多个地方各设一份。"""
        return cls(
            aggressiveness=cfg.vad_aggressiveness,
            end_frames=max(5, int(cfg.silence_ms) // FRAME_MS),
            min_segment_ms=cfg.min_segment_ms,
        )


@dataclass
class SpeechSegment:
    samples: np.ndarray
    start_sample: int  # 在整条流中的绝对位置
    end_sample: int
    finalized: bool = False
    is_speech: bool = False

    @property
    def duration_ms(self) -> int:
        return int(self.samples.size * 1000 / SAMPLE_RATE)


@dataclass
class VadStats:
    frames: int = 0
    speech_frames: int = 0

    @property
    def speech_ratio(self) -> float:
        return self.speech_frames / self.frames if self.frames else 0.0


class _Accum:
    """追加写入的 int16 缓冲，均摊 O(1)。

    之前这里用 np.concatenate 逐帧拼接：每 20ms 帧都要复制一遍整个累积
    数组，是 O(n^2)。一个 15 秒的段落要复制约 9000 万个元素，而且全程
    跑在录音线程上，会跟 PortAudio 回调抢 CPU，音频容易出破音。
    """

    __slots__ = ("_buf", "_n")

    def __init__(self, capacity: int = 0):
        self._buf = np.zeros(max(capacity, 1024), dtype=np.int16)
        self._n = 0

    def append(self, frame: np.ndarray) -> None:
        need = self._n + frame.size
        if need > self._buf.size:
            cap = max(self._buf.size * 2, need)
            grown = np.zeros(cap, dtype=np.int16)
            grown[: self._n] = self._buf[: self._n]
            self._buf = grown
        self._buf[self._n : need] = frame
        self._n = need

    def keep_last(self, k: int) -> None:
        """只保留最后 k 个样本。k <= 0 时清空。"""
        if k <= 0:
            self._n = 0
            return
        if self._n <= k:
            return
        # 挪到开头，避免尾部空间被浪费、频繁触发扩容
        self._buf[:k] = self._buf[self._n - k : self._n]
        self._n = k

    def to_array(self) -> np.ndarray:
        return self._buf[: self._n].copy()

    def clear(self) -> None:
        self._n = 0

    def __len__(self) -> int:
        return self._n


class Segmenter:
    """把连续音频流切成语音段。

    feed() 每来一块数据调一次；有定稿的段就放进 out_segments 供上层取走。
    线程安全：采集线程和消费线程都只通过 feed/取属性交互。
    """

    def __init__(self, config: VADConfig | None = None):
        self.config = config or VADConfig()
        self._vad = webrtcvad.Vad(self.config.aggressiveness)
        self._reset()

    def _reset(self) -> None:
        self.state = State.IDLE
        self._speech_run = 0
        self._silence_run = 0
        self._carry = _Accum()  # 开段前的预缓冲
        self._active = _Accum()  # 当前语音段
        self._segment_start = 0  # 当前段的绝对起始样本位置
        self._cursor = 0  # 已消费到的绝对样本位置
        self.stats = VadStats()
        self._residue = np.zeros(0, dtype=np.int16)  # 半帧残留
        self._out: list[SpeechSegment] = []

    def reset(self) -> None:
        self._reset()

    def feed(self, chunk: np.ndarray) -> list[SpeechSegment]:
        """喂入新音频，返回本次新定稿的语音段。"""
        if chunk is None or chunk.size == 0:
            return []

        chunk = np.ascontiguousarray(chunk, dtype=np.int16)
        self._out = []

        # 上次剩下的半帧接在前面，保证 VAD 看到的帧长始终是 320
        if self._residue.size:
            chunk = np.concatenate((self._residue, chunk))
            self._residue = np.zeros(0, dtype=np.int16)

        offset = 0
        while offset + FRAME_SAMPLES <= chunk.size:
            frame = chunk[offset : offset + FRAME_SAMPLES]
            offset += FRAME_SAMPLES

            is_speech = self._classify(frame)
            self.stats.frames += 1
            if is_speech:
                self.stats.speech_frames += 1

            self._step(frame, is_speech)
            self._cursor += FRAME_SAMPLES

        self._residue = np.ascontiguousarray(chunk[offset:], dtype=np.int16)
        produced, self._out = self._out, []
        return produced

    # -- 判定 --------------------------------------------------------------

    def _classify(self, frame: np.ndarray) -> bool:
        cfg = self.config
        # 能量下限：挡住持续的底噪，让 webrtcvad 只做"是不是人声"的判断
        if float(np.abs(frame).mean()) / 32768.0 < cfg.energy_floor:
            return False
        try:
            return self._vad.is_speech(frame.tobytes(), SAMPLE_RATE)
        except Exception:
            # 帧长不对时 webrtcvad 会抛异常，保守当成语音，交给后面的能量判断兜底
            return True

    def _step(self, frame: np.ndarray, is_speech: bool) -> None:
        cfg = self.config

        if self.state is State.IDLE:
            # 始终攒一小段预缓冲，开段时一起带上
            self._carry.append(frame)
            self._carry.keep_last(cfg.preroll_ms * SAMPLE_RATE // 1000)

            if is_speech:
                self._speech_run += 1
                if self._speech_run >= cfg.start_frames:
                    self._begin()
            else:
                self._speech_run = 0
            return

        # SPEECH / TAIL 都在累积音频
        self._active.append(frame)
        n_active = len(self._active)

        max_samples = int(cfg.max_segment_s * SAMPLE_RATE)
        if is_speech:
            self._silence_run = 0
            if self.state is State.TAIL:
                self.state = State.SPEECH  # 又说上话了
            self._speech_run += 1
        else:
            self._silence_run += 1
            if self.state is State.SPEECH and self._silence_run >= cfg.end_frames:
                self.state = State.TAIL

        if self.state is State.TAIL or n_active >= max_samples:
            self._end(carry_over=n_active >= max_samples)

    def _begin(self) -> None:
        self._active.clear()
        self._active.append(self._carry.to_array())
        self._segment_start = self._cursor + FRAME_SAMPLES - len(self._active)
        self.state = State.SPEECH
        self._silence_run = 0
        self._speech_run = 0
        self._carry.clear()

    def _end(self, carry_over: bool = False) -> None:
        cfg = self.config
        n_active = len(self._active)

        if n_active < cfg.min_segment_ms * SAMPLE_RATE // 1000:
            # 太短，多半是杂音。丢掉，但保留预缓冲继续等。
            self.state = State.IDLE
            self._active.clear()
            self._speech_run = 0
            self._silence_run = 0
            return

        audio = self._active.to_array()
        start_sample = self._segment_start
        end_sample = start_sample + n_active

        seg = SpeechSegment(
            samples=audio,
            start_sample=start_sample,
            end_sample=end_sample,
            finalized=True,
            is_speech=True,
        )
        self._out.append(seg)

        self.state = State.IDLE
        self._active.clear()
        self._speech_run = 0
        self._silence_run = 0

        if carry_over:
            # 因为超长而强制切分时，说话人还在继续。留一小段尾巴当下一段的
            # 预缓冲，否则紧跟其后的内容会被切掉开头。
            self._carry.append(audio[-cfg.preroll_ms * SAMPLE_RATE // 1000 :])
            # 尾巴已经是明显的语音，让下一段直接接上
            self._speech_run = cfg.start_frames
        else:
            # 正常断句：后面是静音，丢掉即可
            self._carry.clear()

    # -- 增量预览 ----------------------------------------------------------

    def peek_active(self) -> SpeechSegment | None:
        """当前正在进行的语音段（未定稿），用于实时预览。

        太短的段不给预览：只喂一秒语音给 whisper，它听不全上下文，
        会猜出乱七八糟的词（实测出现过把句首听成 "Watch your cunt"）。
        定稿不受影响，预览本来就是临时结果。
        """
        n = len(self._active)
        if self.state is State.IDLE or n == 0:
            return None
        if n < self.config.min_preview_ms * SAMPLE_RATE // 1000:
            return None
        return SpeechSegment(
            # 这里必须物化成独立数组：推理线程要拿它喂给 whisper，
            # 不能让下一帧 append 改到同一块内存
            samples=self._active.to_array(),
            start_sample=self._segment_start,
            end_sample=self._segment_start + n,
            finalized=False,
        )

    def take_finalized(self) -> list[SpeechSegment]:
        out, self._out = self._out, []
        return out

    def flush(self) -> list[SpeechSegment]:
        """录音停止时把没说完的段强行定稿，否则最后半句会丢。"""
        out = self.take_finalized()
        n = len(self._active)
        if n >= self.config.min_segment_ms * SAMPLE_RATE // 1000:
            seg = SpeechSegment(
                samples=self._active.to_array(),
                start_sample=self._segment_start,
                end_sample=self._segment_start + n,
                finalized=True,
                is_speech=True,
            )
            out.append(seg)
        self._reset()
        return out
