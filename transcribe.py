"""推理调度：模型加载、定稿/增量两条任务路径、幻觉过滤。

只有一条推理线程持有 whisper context——whisper 的 context 不是线程安全的，
并发调用会直接崩。定稿任务走 FIFO 队列，增量任务只保留"最新一条"，
因为用户还在说话时，排队的旧预览没有价值。
"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from text_norm import TextNormalizer, is_hallucination, is_suspicious
from vad import SpeechSegment
from whisper_bridge import Result, WhisperEngine, WhisperError


@dataclass
class Task:
    segment: SpeechSegment
    kind: str  # "final" | "partial"


@dataclass
class Output:
    kind: str
    segment: SpeechSegment
    result: Result


class Transcriber:
    """引擎的拥有者。start() 之后模型已在后台加载完毕。"""

    def __init__(
        self,
        model_path: Path,
        use_gpu: bool = False,
        n_threads: int | None = None,
        language: str = "zh",
        no_speech_thold: float = 0.6,
        vendor_dir: Path | None = None,
        live_interval_s: float = 1.2,
        verbose: bool = False,
        simplify: bool = True,
        fix_punctuation: bool = True,
    ):
        self.model_path = model_path
        self.use_gpu = use_gpu
        self.n_threads = n_threads
        self.language = language
        self.no_speech_thold = no_speech_thold
        self.vendor_dir = vendor_dir
        self.live_interval_s = live_interval_s
        self.verbose = verbose
        self.normalizer = TextNormalizer(simplify=simplify,
                                         fix_punctuation=fix_punctuation)

        self._engine: WhisperEngine | None = None
        self._final_q: queue.Queue[Task] = queue.Queue()
        self._partial: Task | None = None
        self._partial_lock = threading.Lock()
        self._inflight = False   # 正在 whisper 里跑的任务
        self._inflight_lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._on_output = None
        self._on_error = None
        self._finalized_upto = 0

        # 统计
        self.avg_rtf = 0.0
        self.last_rtf = 0.0
        self._last_elapsed_s = 0.0
        self.dropped_hallucinations = 0
        self.dropped_fragments = 0
        self.total_segments = 0      # 定稿段数
        self.total_previews = 0
        self.previews_suppressed = 0

    # -- 生命周期 ----------------------------------------------------------

    def load_blocking(self) -> str:
        """同步加载模型，返回 whisper.cpp 版本号。"""
        engine = WhisperEngine(self.model_path, use_gpu=self.use_gpu,
                               n_threads=self.n_threads, vendor_dir=self.vendor_dir,
                               verbose=self.verbose)
        engine.load()
        self._engine = engine
        return engine.version

    def start(self, on_output: callable, on_error: callable) -> None:
        if self._engine is None:
            raise RuntimeError("模型尚未加载，请先调用 load_blocking()")
        # 回调存在实例上而不是当参数传给线程：重入时新线程可能根本不会起，
        # 这时回调必须已经是新的，否则旧线程还在往过期的窗口上发信号。
        self._on_output = on_output
        self._on_error = on_error
        # 重入保护：再起一条线程的话，两条会同时对同一个 whisper context
        # 调 whisper_full，而它不是线程安全的——直接段错误。
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, name="transcribe", daemon=True
        )
        self._thread.start()

    def stop_worker(self, timeout: float = 5.0) -> bool:
        """只停推理线程，保留已加载的引擎。

        一次录音结束就该调它：线程留着不动，下次 start() 会和它抢同一个
        context。返回值 False 表示线程在 timeout 内没退出——此时**不要**
        释放引擎，那会造成 use-after-free。

        stop() 才是"连模型一起卸载"，只在切换模型和退出程序时用。
        """
        self._stop.set()
        if self._thread is None:
            return True
        self._thread.join(timeout=timeout)
        alive = self._thread.is_alive()
        if not alive:
            self._thread = None
        return not alive

    def stop(self, timeout: float = 5.0) -> None:
        """停线程并卸载模型。仅供切换模型 / 退出程序使用。"""
        self._stop.set()
        joined = True
        if self._thread:
            self._thread.join(timeout=timeout)
            # join 超时说明线程还在 whisper_full 里，此时 free() 就是
            # use-after-free。宁可漏释放，也不能崩在别人正读着的内存上。
            joined = not self._thread.is_alive()
            if joined:
                self._thread = None
        if self._engine and joined:
            self._engine.free()
            self._engine = None

    # -- 任务提交 ----------------------------------------------------------

    def submit_final(self, segment: SpeechSegment) -> None:
        self._finalized_upto = max(self._finalized_upto, segment.end_sample)
        # 这个段定稿了，之前为它排的增量预览没意义了
        with self._partial_lock:
            if self._partial and self._partial.segment.start_sample == segment.start_sample:
                self._partial = None
        self._final_q.put(Task(segment, "final"))

    def submit_partial(self, segment: SpeechSegment) -> None:
        if segment.end_sample <= self._finalized_upto:
            return
        if not self.preview_affordable():
            self.previews_suppressed += 1
            return
        with self._partial_lock:
            self._partial = Task(segment, "partial")

    def preview_affordable(self) -> bool:
        """算力够不够跑增量预览。

        whisper 的编码器恒定按 30 秒 mel 窗口计算，所以每次调用的开销和
        音频长短几乎无关——2 秒的片段和 30 秒的片段成本差不多。这意味着
        真正的瓶颈是"调用次数"而不是"音频总长"。

        如果上一次识别耗时已经接近预览间隔，再叠增量预览只会把定稿饿死，
        这时直接放弃预览，优先保证正文出字。
        """
        if self._last_elapsed_s <= 0:
            return True
        return self._last_elapsed_s < self.live_interval_s * 0.8

    def shutdown(self, timeout: float = 5.0) -> None:
        self.stop(timeout=timeout)

    def busy(self) -> bool:
        with self._inflight_lock:
            inflight = self._inflight
        return not self._final_q.empty() or inflight

    def wait_idle(self, timeout: float = 60.0) -> None:
        """等定稿任务真正跑完，用于停止录音时收尾。

        只看队列是否为空是不够的：最后一个任务可能已经从队列里取出来、
        正在 whisper 里跑。这时候队列是空的但活没干完，界面却已经提示
        "已停止"，用户紧接着关窗口就会丢掉最后一句话。
        """
        deadline = time.monotonic() + timeout
        while self.busy() and time.monotonic() < deadline:
            time.sleep(0.02)

    # -- 工作循环 ----------------------------------------------------------

    def _run(self) -> None:
        while not self._stop.is_set():
            task = self._take_task()
            if task is None:
                time.sleep(0.03)
                continue
            if task.kind == "stop":
                break
            # finally 而不是逐条出口清理：_process 里有一串提前 return
            # （空段、幻觉、碎片），漏掉任何一条 _inflight 都会永久卡在
            # True，wait_idle() 从此再也不会返回。
            try:
                self._process(task, self._on_output, self._on_error)
            finally:
                self._finish_task()

    def _take_task(self) -> Task | None:
        """定稿优先；没有定稿才取最新的增量。

        置 _inflight 必须和出队原子地发生：任务一旦离开队列，而 _inflight
        还没置位，busy() 就会误判成"空闲"，wait_idle() 提前返回，
        最后一句会在界面上刚说"已停止"之后才吐出来。
        """
        task = None
        try:
            task = self._final_q.get_nowait()
        except queue.Empty:
            with self._partial_lock:
                task, self._partial = self._partial, None
        if task is None:
            return None
        with self._inflight_lock:
            self._inflight = True
        return task

    def _finish_task(self) -> None:
        with self._inflight_lock:
            self._inflight = False

    def _process(self, task: Task, on_output, on_error) -> None:
        seg = task.segment
        if seg is None or seg.samples.size == 0:
            return
        try:
            result = self._engine.transcribe(
                seg.samples,
                language=self.language,
                no_speech_thold=self.no_speech_thold,
            )
        except WhisperError as e:
            on_error(str(e))
            return
        except Exception as e:  # 推理线程不能因为一次异常就整体死掉
            on_error(f"识别失败: {e}")
            return

        if not result.text:
            self.dropped_hallucinations += 1
            return

        # 繁简统一 + 标点规整。必须排在幻觉/碎片判定之前：
        # whisper 的幻觉输出常带繁体（实测 "(字幕:貝爾)"），
        # 不先转简体就匹配不上。
        result.text = self.normalizer(result.text)

        # 幻觉过滤放在这里，而不是只依赖 whisper 的 no_speech_prob。
        # 实测在安静房间录到的噪声段上，whisper 输出的 no_speech_prob
        # 是 0.000——它非常确信那是人话，把阈值从 0.6 调到 0.3 也毫无
        # 变化。只有短语识别这一层拦得住。
        if is_hallucination(result.text):
            self.dropped_hallucinations += 1
            return

        if is_suspicious(result.text, seg.duration_ms):
            self.dropped_fragments += 1
            return

        if task.kind == "final":
            self.total_segments += 1
        else:
            self.total_previews += 1
        self._last_elapsed_s = result.elapsed_s
        if result.rtf > 0:
            self.last_rtf = result.rtf
            # 指数滑动平均，界面上的速度指示不会被单次抖动带偏
            self.avg_rtf = result.rtf if self.avg_rtf == 0 else self.avg_rtf * 0.7 + result.rtf * 0.3

        on_output(Output(task.kind, seg, result))
