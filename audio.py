"""麦克风采集：设备枚举、16kHz 单声道 int16 采集、线程安全环形缓冲区。

PortAudio 的回调运行在它自己的音频线程上，不能在里面做耗时操作，否则会
爆音。这里回调只把数据 copy 进环形缓冲，VAD 那一侧按需读取。
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Callable

import numpy as np
import sounddevice as sd

SAMPLE_RATE = 16000  # whisper 只吃 16kHz，重采样反而引入误差
BLOCK_SIZE = 1024  # 64ms @16kHz
BUFFER_SECONDS = 30


@dataclass
class InputDevice:
    index: int
    name: str
    channels: int
    default_samplerate: float

    @property
    def label(self) -> str:
        return f"[{self.index}] {self.name}"


class AudioError(RuntimeError):
    pass


def _clean_name(raw) -> str:
    """修正 PortAudio 返回的设备名乱码。

    Windows 上 PortAudio 把 UTF-8 字节按 Latin-1 解了出来，中文设备名会
    显示成"话筒"这样的乱码。这里反向转一次；只有当转完之后
    Latin-1 补充区字符变少时（说明真的解出了正常文字）才采用，
    否则保留原名，避免把本来正常的名字转坏。
    """
    name = raw if isinstance(raw, str) else str(raw)
    if name.isascii():
        return name
    try:
        fixed = name.encode("latin-1").decode("utf-8")
    except (UnicodeDecodeError, UnicodeEncodeError):
        return name
    mojibake = sum(1 for c in name if 0x80 <= ord(c) <= 0xFF)
    if mojibake and sum(1 for c in fixed if 0x80 <= ord(c) <= 0xFF) < mojibake:
        return fixed
    return name


def list_input_devices() -> list[InputDevice]:
    devices = []
    for i, d in enumerate(sd.query_devices()):
        if d.get("max_input_channels", 0) < 1:
            continue
        devices.append(
            InputDevice(
                index=i,
                name=_clean_name(d.get("name", "unknown")),
                channels=d["max_input_channels"],
                default_samplerate=float(d.get("default_samplerate", SAMPLE_RATE)),
            )
        )
    return devices


def default_input_index() -> int:
    """尽量找出一个合理的默认输入设备。

    不能直接信 PortAudio 报的默认值：这台机器上它把默认输入指向了 4 号，
    而 4 号其实是扬声器（0 个输入通道），照搬会开不出流。所以候选设备
    必须真的带输入通道才认。
    """
    candidates = []
    try:
        if hasattr(sd, "query_default_input"):
            candidates.append(sd.query_default_input())
    except Exception:
        pass
    try:
        out, inp = sd.default.device
        candidates.append(inp)
    except Exception:
        pass

    devs = list_input_devices()
    valid = {d.index for d in devs}
    for c in candidates:
        if c is None:
            continue
        try:
            ci = int(c)
        except (TypeError, ValueError):
            continue
        if ci in valid:
            return ci

    return devs[0].index if devs else -1


class RingBuffer:
    """固定容量的 int16 环形缓冲区，满了就覆盖最旧的样本。"""

    def __init__(self, seconds: int = BUFFER_SECONDS, rate: int = SAMPLE_RATE):
        self._cap = seconds * rate
        self._buf = np.zeros(self._cap, dtype=np.int16)
        self._write = 0
        self._total = 0
        self._lock = threading.Lock()

    def write(self, data: np.ndarray) -> None:
        with self._lock:
            n = data.size
            if n >= self._cap:
                self._buf[:] = data[-self._cap :]
                self._write = 0
            else:
                end = self._write + n
                if end <= self._cap:
                    self._buf[self._write : end] = data
                else:
                    first = self._cap - self._write
                    self._buf[self._write :] = data[:first]
                    self._buf[: end - self._cap] = data[first:]
                self._write = end % self._cap
            self._total += n

    def read(self, n: int) -> np.ndarray:
        """取最新的 n 个样本。不足 n 时返回现有全部。"""
        with self._lock:
            avail = min(self._total, self._cap)
            n = min(n, avail)
            if n == 0:
                return np.zeros(0, dtype=np.int16)
            start = (self._write - n) % self._cap
            if start + n <= self._cap:
                return self._buf[start : start + n].copy()
            first = self._cap - start
            return np.concatenate((self._buf[start:], self._buf[: n - first]))

    def read_since(self, position: int) -> tuple[np.ndarray, int]:
        """读取自 position 以来新到的样本，同时返回更新后的 position。"""
        with self._lock:
            end = self._total
            if position > end:
                position = 0  # 环形缓冲绕回或被重置，从头开始
            start = max(0, end - self._cap)
            if position < start:
                position = start
            count = end - position
            n = min(count, self._cap)
            read_start = end - n
            s = (read_start % self._cap)
            if s + n <= self._cap:
                data = self._buf[s : s + n].copy()
            else:
                first = self._cap - s
                data = np.concatenate((self._buf[s:], self._buf[: n - first]))
            return data, end

    @property
    def total_written(self) -> int:
        return self._total

    def reset(self) -> None:
        with self._lock:
            self._buf[:] = 0
            self._write = 0
            self._total = 0


class MicRecorder:
    """持续把麦克风数据写进环形缓冲。

    on_level 回调用于驱动界面上的音量条，务必保持廉价。
    """

    def __init__(
        self,
        device_index: int = -1,
        on_level: Callable[[float], None] | None = None,
        seconds: int = BUFFER_SECONDS,
    ):
        self.device_index = device_index
        self.on_level = on_level
        self.ring = RingBuffer(seconds=seconds)
        self._stream: sd.InputStream | None = None
        self._level = 0.0
        self._lock = threading.Lock()
        self._overflow = 0

    @property
    def level(self) -> float:
        """归一化到 0~1 的音量，驱动界面音量条。"""
        return self._level

    @property
    def overflow_count(self) -> int:
        return self._overflow

    def start(self) -> None:
        if self._stream is not None:
            return
        kwargs = dict(
            samplerate=SAMPLE_RATE,
            blocksize=BLOCK_SIZE,
            dtype="int16",
            channels=1,
            callback=self._callback,
        )
        if self.device_index is not None and self.device_index >= 0:
            kwargs["device"] = self.device_index
        try:
            self._stream = sd.InputStream(**kwargs)
            self._stream.start()
        except Exception as e:
            self._stream = None
            raise AudioError(
                f"无法打开麦克风 (device={self.device_index}): {e}\n"
                "请在界面里换一个输入设备，或检查系统隐私设置是否允许录音。"
            ) from e

    def stop(self) -> None:
        with self._lock:
            if self._stream is not None:
                try:
                    self._stream.stop()
                    self._stream.close()
                finally:
                    self._stream = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.stop()

    def _callback(self, indata, frames, time_info, status):  # noqa: ARG002
        if status:
            # 输入溢出意味着回调没赶上，丢弃这一块但继续跑
            self._overflow += 1
        data = np.frombuffer(indata, dtype=np.int16)
        if data.size:
            self.ring.write(data)
            peak = float(np.abs(data).max()) / 32768.0
            # 上升快、下降慢，读起来更像真的电平表
            self._level = peak if peak > self._level else self._level * 0.75 + peak * 0.25
        if self.on_level:
            try:
                self.on_level(self._level)
            except Exception:
                pass

    def drain(self) -> np.ndarray:
        """停止录音后把残余音频全部取走。"""
        with self._lock:
            if self._stream is not None:
                try:
                    self._stream.stop()
                except Exception:
                    pass
            return self.ring.read(self.ring.total_written)

    def snapshot(self, seconds: float) -> np.ndarray:
        return self.ring.read(int(seconds * SAMPLE_RATE))
