"""无界面端到端联调：麦克风 -> VAD 分段 -> 识别 -> 终端逐行打印。

用来在调 GUI 之前把实时链路调通。按 Ctrl+C 结束。

用法:
    python tools/realtime_test.py
    python tools/realtime_test.py --model base --lang zh --sensitivity 2
    python tools/realtime_test.py --file tests/jfk.wav --loop   # 不开麦，跑文件
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import device  # noqa: E402
from audio import MicRecorder, default_input_index, list_input_devices  # noqa: E402
from transcribe import Output, Transcriber  # noqa: E402
from vad import VADConfig, Segmenter  # noqa: E402

SAMPLE_RATE = 16000
REALTIME_SPEED = 1.0  # 放慢一倍跑文件，方便肉眼核对分段点


class Printer:
    def __init__(self):
        self.lock = threading.Lock()
        self.start = time.monotonic()
        self.partial = ""
        self._last_drawn = None

    def _elapsed(self) -> str:
        return f"[{time.monotonic() - self.start:6.1f}s]"

    def on_output(self, out: Output) -> None:
        if out.kind == "partial":
            with self.lock:
                self.partial = out.result.text
            return
        with self.lock:
            self.partial = ""
        seg = out.segment
        print(f"\n{self._elapsed()} ━━ 定稿 {seg.duration_ms / 1000:.1f}s "
              f"RTF={out.result.rtf:.2f}")
        print(f"{self._elapsed()} ━━ {out.result.text}")

    def tick(self) -> None:
        with self.lock:
            p = self.partial
        # 只在内容变化时重画，否则终端会被刷屏
        if p == self._last_drawn:
            return
        self._last_drawn = p
        sys.stdout.write(f"\r{self._elapsed()} [预览] {p[:60]:<60}")
        sys.stdout.flush()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="small")
    ap.add_argument("--lang", default="zh")
    ap.add_argument("--sensitivity", type=int, default=2, choices=[0, 1, 2, 3])
    ap.add_argument("--silence-ms", type=int, default=None,
                    help="停顿阈值(ms)，不指定则用 config.json 里的值")
    ap.add_argument("--live-interval-ms", type=int, default=1200)
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--verbose", action="store_true", help="打印 whisper.cpp 原生日志")
    ap.add_argument("--file", type=Path, help="用 wav 文件代替麦克风")
    ap.add_argument("--loop", action="store_true", help="把文件循环播放，测长时间表现")
    ap.add_argument("--seconds", type=float, help="录满这么多秒后自动停止（用于自动化测试）")
    args = ap.parse_args()
    from config import Config
    if args.silence_ms is None:
        args.silence_ms = Config.load().silence_ms

    model = ROOT / "models" / f"ggml-{args.model}.bin"
    if not model.is_file():
        print(f"模型不存在: {model}\n先运行 python tools/fetch_assets.py --model {args.model}")
        return 1

    info = device.detect("gpu" if args.gpu else "auto")
    print(f"设备: {info.label}  ({info.detail})")
    print(f"CPU 线程数: {device.default_threads()}")
    print(f"模型: {model.name}")

    print("加载模型 …")
    t0 = time.monotonic()
    tr = Transcriber(
        model_path=model, use_gpu=info.use_gpu,
        n_threads=device.default_threads(), language=args.lang,
        no_speech_thold=0.6, vendor_dir=info.vendor_dir,
        live_interval_s=args.live_interval_ms / 1000.0,
        verbose=args.verbose,
    )
    version = tr.load_blocking()
    print(f"就绪: whisper.cpp {version}，加载耗时 {time.monotonic() - t0:.1f}s\n")

    printer = Printer()
    tr.start(printer.on_output, lambda e: print(f"\n[错误] {e}"))

    if args.file:
        return _run_file(args, tr, printer)

    return _run_mic(args, tr, printer)


def _run_mic(args, tr: Transcriber, printer: Printer) -> int:
    devs = list_input_devices()
    print("可用输入设备:")
    for d in devs:
        print(f"  [{d.index}] {d.name}")
    idx = default_input_index()
    print(f"\n使用设备 [{idx}]，按 Ctrl+C 结束\n")

    from config import Config
    vcfg = VADConfig.from_config(Config.load())
    vcfg.end_frames = max(5, args.silence_ms // 20)
    vcfg.aggressiveness = args.sensitivity
    seg = Segmenter(vcfg)
    rec = MicRecorder(device_index=idx)
    try:
        rec.start()
    except Exception as e:
        print(f"打开麦克风失败: {e}")
        tr.shutdown()
        return 1

    print("● 录音中 …\n")
    pos = 0
    last_partial = 0.0
    last_tick = 0.0
    deadline = time.monotonic() + args.seconds if args.seconds else None
    try:
        while True:
            if deadline and time.monotonic() >= deadline:
                print("\n■ 到时自动停止 …")
                break
            data, pos = rec.ring.read_since(pos)
            if data.size:
                for s in seg.feed(data):
                    tr.submit_final(s)
                now = time.monotonic()
                if now - last_partial >= args.live_interval_ms / 1000.0:
                    a = seg.peek_active()
                    if a is not None:
                        tr.submit_partial(a)
                    last_partial = now
            if time.monotonic() - last_tick > 0.1:
                printer.tick()
                last_tick = time.monotonic()
            time.sleep(0.03)
    except KeyboardInterrupt:
        print("\n\n■ 停止 …")
    finally:
        rec.stop()
        for s in seg.flush():
            tr.submit_final(s)
        tr.wait_idle(timeout=60)
        time.sleep(0.3)
        tr.shutdown()

    if tr.avg_rtf:
        print(f"\n定稿 {tr.total_segments} 段，预览 {tr.total_previews} 次，"
              f"过滤幻觉 {tr.dropped_hallucinations} 次，碎片 {tr.dropped_fragments} 段，"
              f"跳过预览 {tr.previews_suppressed} 次，平均速度 {1 / tr.avg_rtf:.1f}x")
    else:
        print("\n没有识别到语音")
    return 0


def _run_file(args, tr: Transcriber, printer: Printer) -> int:
    import wave

    with wave.open(str(args.file), "rb") as w:
        rate, width, ch = w.getframerate(), w.getsampwidth(), w.getnchannels()
        if (rate, width, ch) != (SAMPLE_RATE, 2, 1):
            print(f"需要 16kHz/单声道/16bit，实际 {rate}/{ch}/{width}")
            tr.shutdown()
            return 1
        samples = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)

    from config import Config
    vcfg = VADConfig.from_config(Config.load())
    vcfg.end_frames = max(5, args.silence_ms // 20)
    vcfg.aggressiveness = args.sensitivity
    seg = Segmenter(vcfg)
    print(f"文件 {args.file.name}: {samples.size / SAMPLE_RATE:.1f}s，"
          f"按 {REALTIME_SPEED}x 速度喂入\n")

    step = int(SAMPLE_RATE * 0.064 / REALTIME_SPEED)
    pos = 0
    last_partial = 0.0
    rounds = 0
    while pos < samples.size or args.loop:
        chunk = samples[pos : pos + step]
        pos += step
        for s in seg.feed(chunk):
            tr.submit_final(s)
        now = time.monotonic()
        if now - last_partial >= args.live_interval_ms / 1000.0:
            a = seg.peek_active()
            if a is not None:
                tr.submit_partial(a)
            last_partial = now
        printer.tick()
        time.sleep(step / SAMPLE_RATE / REALTIME_SPEED)
        if pos >= samples.size:
            if not args.loop:
                break
            pos = 0
            rounds += 1
            print(f"\n（循环第 {rounds + 1} 轮）")

    for s in seg.flush():
        tr.submit_final(s)
    tr.wait_idle(timeout=120)
    time.sleep(0.3)
    tr.shutdown()
    print(f"\n定稿 {tr.total_segments} 段，预览 {tr.total_previews} 次，"
          f"过滤幻觉 {tr.dropped_hallucinations} 次，碎片 {tr.dropped_fragments} 段，"
          f"跳过预览 {tr.previews_suppressed} 次")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
