"""ctypes 封装 whisper.cpp 的 whisper.dll。

只暴露实时转写需要的最小接口：加载模型、喂 float32 采样、读回分段文本。
结构体布局严格对照 vendor/whisper.h（b5130），改模型或升级 whisper.cpp 时
必须重新核对头文件，否则 ctypes 会静默读到错位的字段。
"""

from __future__ import annotations

import ctypes
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

import platforms
ROOT = platforms.app_root()
VENDOR_DIR = ROOT / "vendor" / "whisper" / "Release"

WHISPER_SAMPLING_GREEDY = 0

# 静音幻觉常见的固定短语，识别结果整段命中这些内容时直接丢弃。
HALLUCINATION_MARKERS = (
    "thank you for watching",
    "thanks for watching",
    "thank you very much",
    "please subscribe",
    "subtitles by",
    "amara.org",
    "字幕由",
    "谢谢观看",
    "感谢观看",
    "请不吝点赞",
    "订阅",
    "转发",
    "打赏",
)


class WhisperError(RuntimeError):
    pass


# --------------------------------------------------------------------------
# 结构体定义
# --------------------------------------------------------------------------


class _WhisperAhead(ctypes.Structure):
    _fields_ = [("n_text_layer", ctypes.c_int), ("n_head", ctypes.c_int)]


class _WhisperAheads(ctypes.Structure):
    _fields_ = [
        ("n_heads", ctypes.c_size_t),
        ("heads", ctypes.POINTER(_WhisperAhead)),
    ]


class WhisperContextParams(ctypes.Structure):
    _fields_ = [
        ("use_gpu", ctypes.c_bool),
        ("flash_attn", ctypes.c_bool),
        ("gpu_device", ctypes.c_int),
        ("dtw_token_timestamps", ctypes.c_bool),
        ("dtw_aheads_preset", ctypes.c_int),
        ("dtw_n_top", ctypes.c_int),
        ("dtw_aheads", _WhisperAheads),
        ("dtw_mem_size", ctypes.c_size_t),
    ]


class WhisperVadParams(ctypes.Structure):
    _fields_ = [
        ("threshold", ctypes.c_float),
        ("min_speech_duration_ms", ctypes.c_int),
        ("min_silence_duration_ms", ctypes.c_int),
        ("max_speech_duration_s", ctypes.c_float),
        ("speech_pad_ms", ctypes.c_int),
        ("samples_overlap", ctypes.c_float),
    ]


class _Greedy(ctypes.Structure):
    _fields_ = [("best_of", ctypes.c_int)]


class _BeamSearch(ctypes.Structure):
    _fields_ = [("beam_size", ctypes.c_int), ("patience", ctypes.c_float)]


_LOG_CALLBACK = ctypes.CFUNCTYPE(None, ctypes.c_int, ctypes.c_char_p, ctypes.c_void_p)


class WhisperFullParams(ctypes.Structure):
    _fields_ = [
        ("strategy", ctypes.c_int),
        ("n_threads", ctypes.c_int),
        ("n_max_text_ctx", ctypes.c_int),
        ("offset_ms", ctypes.c_int),
        ("duration_ms", ctypes.c_int),
        ("translate", ctypes.c_bool),
        ("no_context", ctypes.c_bool),
        ("no_timestamps", ctypes.c_bool),
        ("single_segment", ctypes.c_bool),
        ("print_special", ctypes.c_bool),
        ("print_progress", ctypes.c_bool),
        ("print_realtime", ctypes.c_bool),
        ("print_timestamps", ctypes.c_bool),
        ("token_timestamps", ctypes.c_bool),
        ("thold_pt", ctypes.c_float),
        ("thold_ptsum", ctypes.c_float),
        ("max_len", ctypes.c_int),
        ("split_on_word", ctypes.c_bool),
        ("max_tokens", ctypes.c_int),
        ("debug_mode", ctypes.c_bool),
        ("audio_ctx", ctypes.c_int),
        ("tdrz_enable", ctypes.c_bool),
        ("suppress_regex", ctypes.c_char_p),
        ("initial_prompt", ctypes.c_char_p),
        ("carry_initial_prompt", ctypes.c_bool),
        ("prompt_tokens", ctypes.c_void_p),
        ("prompt_n_tokens", ctypes.c_int),
        ("language", ctypes.c_char_p),
        ("detect_language", ctypes.c_bool),
        ("suppress_blank", ctypes.c_bool),
        ("suppress_nst", ctypes.c_bool),
        ("temperature", ctypes.c_float),
        ("max_initial_ts", ctypes.c_float),
        ("length_penalty", ctypes.c_float),
        ("temperature_inc", ctypes.c_float),
        ("entropy_thold", ctypes.c_float),
        ("logprob_thold", ctypes.c_float),
        ("no_speech_thold", ctypes.c_float),
        ("greedy", _Greedy),
        ("beam_search", _BeamSearch),
        ("new_segment_callback", ctypes.c_void_p),
        ("new_segment_callback_user_data", ctypes.c_void_p),
        ("progress_callback", ctypes.c_void_p),
        ("progress_callback_user_data", ctypes.c_void_p),
        ("encoder_begin_callback", ctypes.c_void_p),
        ("encoder_begin_callback_user_data", ctypes.c_void_p),
        ("abort_callback", ctypes.c_void_p),
        ("abort_callback_user_data", ctypes.c_void_p),
        ("logits_filter_callback", ctypes.c_void_p),
        ("logits_filter_callback_user_data", ctypes.c_void_p),
        ("grammar_rules", ctypes.c_void_p),
        ("n_grammar_rules", ctypes.c_size_t),
        ("i_start_rule", ctypes.c_size_t),
        ("grammar_penalty", ctypes.c_float),
        ("vad", ctypes.c_bool),
        ("vad_model_path", ctypes.c_char_p),
        ("vad_params", WhisperVadParams),
    ]


# --------------------------------------------------------------------------
# 结果结构
# --------------------------------------------------------------------------


@dataclass
class Segment:
    t0_ms: int
    t1_ms: int
    text: str
    no_speech_prob: float


@dataclass
class Result:
    text: str = ""
    segments: list[Segment] = field(default_factory=list)
    elapsed_s: float = 0.0
    audio_s: float = 0.0
    lang_id: int = -1

    @property
    def rtf(self) -> float:
        """real-time factor：识别耗时 / 音频时长。小于 1 才跟得上实时。"""
        return self.elapsed_s / self.audio_s if self.audio_s > 0 else 0.0


# --------------------------------------------------------------------------
# 引擎
# --------------------------------------------------------------------------


class WhisperEngine:
    """封装一个已加载的 whisper context。同一时刻只允许一个线程调用。"""

    def __init__(
        self,
        model_path: Path,
        use_gpu: bool = False,
        n_threads: int | None = None,
        vendor_dir: Path | None = None,
        verbose: bool = False,
    ):
        self.model_path = Path(model_path)
        self.use_gpu = use_gpu
        self.n_threads = n_threads or max(1, (os.cpu_count() or 4) - 1)
        self.vendor_dir = Path(vendor_dir) if vendor_dir else VENDOR_DIR
        self.verbose = verbose
        self._dll = None
        self._ctx = None
        self._log_cb = None  # 必须保持引用，否则被 GC 后回调指向野指针

    # -- 加载 / 释放 -------------------------------------------------------

    def load(self) -> None:
        if not self.model_path.is_file():
            raise WhisperError(f"模型文件不存在: {self.model_path}")

        plat = platforms.current()
        dll_path = self.vendor_dir / plat.whisper_path_name
        if not dll_path.is_file():
            raise WhisperError(
                f"找不到 whisper.dll: {dll_path}\n"
                "请先运行 python tools/fetch_assets.py 下载 whisper.cpp 运行时。"
            )

        self._dll_dir_handle = platforms.add_vendor_to_search_path(self.vendor_dir)

        # 顺序有讲究：必须先把 ggml 后端注册进注册表，再加载 whisper.dll。
        # 反过来做（先 WinDLL 再注册后端）会在建 context 时撞 ggml 的
        # layout 断言直接崩掉，实测过。
        #
        # ggml 后端 DLL（ggml-cpu-haswell.dll 等）在运行期按名字动态加载。
        # 它默认只在"宿主进程 exe 目录"里找，而我们的宿主是 python.exe，
        # 于是扫描不到任何后端，建 context 时 GGML_ASSERT(device) 崩掉。
        # 这里显式把 vendor 目录喂给注册表，GGML_BACKEND_PATH 那个环境变量
        # 走的是另一条 out-of-tree 加载路径，对这个场景不生效。
        self._load_backends()

        self._dll = platforms.load_library(dll_path)
        self._bind()

        # 尽早静音。模型加载会刷几十行日志，双击 run.bat 时这些会直接
        # 冲掉控制台；GUI 状态栏已经显示了实际设备，不需要在这里重复。
        if not self.verbose:
            self.silence_logs()

        version = self._dll.whisper_version()
        if not version:
            raise WhisperError(
                f"{plat.whisper_path_name} 加载异常：whisper_version() 返回空")

        cparams = self._dll.whisper_context_default_params()
        cparams.use_gpu = self.use_gpu
        self._ctx = self._dll.whisper_init_from_file_with_params(
            str(self.model_path).encode("utf-8"), cparams
        )
        if not self._ctx:
            raise WhisperError(
                f"模型加载失败: {self.model_path.name}\n"
                "请确认模型完整下载，且与 whisper.dll 版本匹配。"
            )
        self.version = version.decode("utf-8", "replace")

    def free(self) -> None:
        if self._dll is not None and self._ctx:
            self._dll.whisper_free(self._ctx)
            self._ctx = None
        if getattr(self, "_dll_dir_handle", None):
            self._dll_dir_handle.close()
            self._dll_dir_handle = None

    def __del__(self):
        try:
            self.free()
        except Exception:
            pass

    # -- ggml 后端注册 -----------------------------------------------------

    def _load_backends(self) -> None:
        """把 vendor 目录里的计算后端注册进 ggml。

        必须发生在 whisper_init_from_file_with_params 之前，否则 context
        初始化时找不到任何 device 直接断言失败。

        CUDA 后端要额外预加载一次：ggml 内部用 LoadLibrary 加载
        ggml-cuda.dll 时解析不到同目录的 cublas/cudart 依赖，会静默失败
        （release 版把日志静音了，外部只表现为 whisper 打印 "no GPU found"）。
        先用 ctypes 按绝对路径把它拉进进程，依赖就能通过 AddDllDirectory 找到，
        之后 load_all_from_path 就能把它注册进来。
        """
        plat = platforms.current()
        for name in (plat.cuda_backend, plat.cpu_backend):
            backend = self.vendor_dir / f"{name}.{plat.lib_ext}"
            if backend.is_file():
                try:
                    platforms.load_library(backend)
                except OSError:
                    pass

        ggml = platforms.load_library(
            self.vendor_dir / f"{plat.ggml_lib}.{plat.lib_ext}")
        ggml.ggml_backend_load_all_from_path.argtypes = [ctypes.c_char_p]
        ggml.ggml_backend_dev_count.restype = ctypes.c_size_t
        ggml.ggml_backend_load_all_from_path(str(self.vendor_dir).encode("utf-8"))
        n = ggml.ggml_backend_dev_count()
        if n == 0:
            raise WhisperError(
                f"ggml 没能在 {self.vendor_dir} 找到任何计算后端。\n"
                "请检查 ggml-cpu-*.dll 是否完整解压。"
            )
        self.backend_count = n
        # 装了 CUDA 构建时注册数是 2（CUDA + CPU），纯 CPU 构建是 1
        self.cuda_available = n > 1

    # -- 函数签名 ----------------------------------------------------------

    def _bind(self) -> None:
        d = self._dll
        d.whisper_version.restype = ctypes.c_char_p
        d.whisper_version.argtypes = []

        d.whisper_context_default_params.restype = WhisperContextParams
        d.whisper_context_default_params.argtypes = []

        d.whisper_init_from_file_with_params.restype = ctypes.c_void_p
        d.whisper_init_from_file_with_params.argtypes = [
            ctypes.c_char_p,
            WhisperContextParams,
        ]

        d.whisper_free.restype = None
        d.whisper_free.argtypes = [ctypes.c_void_p]

        d.whisper_full_default_params.restype = WhisperFullParams
        d.whisper_full_default_params.argtypes = [ctypes.c_int]

        d.whisper_full.restype = ctypes.c_int
        d.whisper_full.argtypes = [
            ctypes.c_void_p,
            WhisperFullParams,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
        ]

        d.whisper_full_n_segments.restype = ctypes.c_int
        d.whisper_full_n_segments.argtypes = [ctypes.c_void_p]

        for fn in ("whisper_full_get_segment_t0", "whisper_full_get_segment_t1"):
            getattr(d, fn).restype = ctypes.c_int64
            getattr(d, fn).argtypes = [ctypes.c_void_p, ctypes.c_int]

        d.whisper_full_get_segment_text.restype = ctypes.c_char_p
        d.whisper_full_get_segment_text.argtypes = [ctypes.c_void_p, ctypes.c_int]

        d.whisper_full_get_segment_no_speech_prob.restype = ctypes.c_float
        d.whisper_full_get_segment_no_speech_prob.argtypes = [ctypes.c_void_p, ctypes.c_int]

        d.whisper_full_lang_id.restype = ctypes.c_int
        d.whisper_full_lang_id.argtypes = [ctypes.c_void_p]

        d.whisper_log_set.restype = None
        d.whisper_log_set.argtypes = [_LOG_CALLBACK, ctypes.c_void_p]

    # -- 静音日志 ----------------------------------------------------------

    def silence_logs(self) -> None:
        """关掉 whisper.cpp / ggml 打到 stderr 的日志。

        whisper.cpp 没有暴露开关，只能把回调换成空的。GUI 模式下那些
        每秒几十行的 CPU 后端信息会直接冲垮窗口。
        """

        def _sink(level, text, user_data):  # noqa: ARG001
            return

        self._log_cb = _LOG_CALLBACK(_sink)
        self._dll.whisper_log_set(self._log_cb, None)

    # -- 识别 --------------------------------------------------------------

    def transcribe(
        self,
        samples_i16: np.ndarray,
        language: str = "zh",
        no_speech_thold: float = 0.6,
        temperature: float = 0.0,
        initial_prompt: str | None = None,
    ) -> Result:
        """识别一段 int16 单声道 16kHz 音频。

        Args:
            samples_i16: int16 numpy 数组。
            language: 语言代码，"auto" 表示自动检测。
            no_speech_thold: 判定为"没说话"的概率阈值。麦克风场景下必须
                调高（默认 0.2 太松），否则环境噪音会不断触发幻觉输出。
        """
        if self._ctx is None:
            raise WhisperError("模型尚未加载，请先调用 load()")

        samples_i16 = np.ascontiguousarray(samples_i16, dtype=np.int16)
        if samples_i16.size == 0:
            return Result(audio_s=0.0)

        pcm = samples_i16.astype(np.float32) / 32768.0

        params = self._dll.whisper_full_default_params(WHISPER_SAMPLING_GREEDY)
        params.n_threads = self.n_threads
        params.language = language.encode("utf-8")
        params.translate = False
        params.no_timestamps = True
        # 逐段送入时强制单段输出，否则模型会自己再切一刀，时间戳和
        # no_speech_prob 的归属就乱了。
        params.single_segment = True
        params.print_progress = False
        params.print_realtime = False
        params.print_special = False
        params.print_timestamps = False
        params.suppress_blank = True
        params.no_speech_thold = no_speech_thold
        params.temperature = temperature
        params.temperature_inc = 0.0  # 关掉温度回退，避免幻觉时反复重采样
        params.greedy.best_of = 1
        if initial_prompt:
            # 上一段的文本作为解码上下文。分段独立识别时模型每段都从零开始，
            # 容易把跨段的一句话听断。只保留尾部：prompt 的可用长度是
            # whisper_n_text_ctx()/2，取多了会被截断在无意义的位置。
            params.initial_prompt = initial_prompt[-120:].encode("utf-8")
            params.carry_initial_prompt = True

        started = time.perf_counter()
        rc = self._dll.whisper_full(
            self._ctx, params, pcm.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), pcm.size
        )
        elapsed = time.perf_counter() - started

        if rc != 0:
            raise WhisperError(f"whisper_full 返回错误码 {rc}")

        segments: list[Segment] = []
        parts: list[str] = []
        for i in range(self._dll.whisper_full_n_segments(self._ctx)):
            raw = self._dll.whisper_full_get_segment_text(self._ctx, i)
            text = (raw or b"").decode("utf-8", "replace").strip()
            if not text:
                continue
            seg = Segment(
                t0_ms=self._dll.whisper_full_get_segment_t0(self._ctx, i),
                t1_ms=self._dll.whisper_full_get_segment_t1(self._ctx, i),
                text=text,
                no_speech_prob=self._dll.whisper_full_get_segment_no_speech_prob(self._ctx, i),
            )
            segments.append(seg)
            parts.append(text)

        joined = "".join(parts).strip()
        if _is_hallucination(joined):
            joined = ""

        return Result(
            text=joined,
            segments=segments,
            elapsed_s=elapsed,
            audio_s=pcm.size / 16000.0,
            lang_id=self._dll.whisper_full_lang_id(self._ctx),
        )


def _is_hallucination(text: str) -> bool:
    """过滤 whisper 对静音/噪声的经典幻觉输出。"""
    if not text:
        return True
    stripped = text.strip().strip("。．.！!？?、,")
    if not stripped:
        return True
    lowered = stripped.lower()
    return any(m in lowered for m in HALLUCINATION_MARKERS)

