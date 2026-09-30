"""SoundTrace（声迹）— 本地实时语音转文字，桌面界面。

线程划分：
    录音/VAD 线程  读环形缓冲 -> 分段 -> 投递给推理器
    推理线程       在 Transcriber 内部，只碰 whisper context
    GUI 主线程     只负责显示，所有更新都靠 signal 回来

界面刻意把"已确认正文"和"实时预览"分开显示：预览是每 1.2 秒重算一次
的临时结果，会改；正文是断句定稿后的结果，不会改。
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

from PySide6.QtCore import QObject, Qt, QThread, QTimer, Signal, Slot
from PySide6.QtGui import QAction, QFont, QKeySequence
from PySide6.QtWidgets import (
    QFrame,
    QApplication,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,   # 模型加载时的忙碌指示
    QPushButton,
    QSlider,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

import branding
import platforms
import recommend
import device
from audio import AudioError, MicRecorder, list_input_devices, default_input_index
from config import Config
import theme
from download_dialog import DownloadDialog
from live_cursor import EmptyStateOverlay, LiveInsertionMark
from settings import SettingsDialog
from widgets import LevelMeter
from transcribe import Output, Transcriber
from vad import Segmenter, VADConfig

SAMPLE_RATE = 16000
POLL_INTERVAL_IDLE = 0.06   # 秒


class Signals(QObject):
    """把各线程的回调转成 Qt signal。"""

    level = Signal(float)
    vad_state = Signal(str)
    final_text = Signal(str, int, int)      # 文本, 起始毫秒, 结束毫秒
    partial_text = Signal(str)
    error = Signal(str)
    info = Signal(str)
    status = Signal(str)
    rtf = Signal(float)
    ended = Signal()


class AudioWorker(QObject):
    """录音 + VAD 循环，跑在自己的 QThread 里。"""

    def __init__(self, recorder: MicRecorder, segmenter: Segmenter,
                 transcriber: Transcriber, live_interval_ms: int, sig: Signals):
        super().__init__()
        self.recorder = recorder
        self.segmenter = segmenter
        self.transcriber = transcriber
        self.live_interval = live_interval_ms / 1000.0
        self.sig = sig
        self._running = False

    def stop(self) -> None:
        self._running = False

    @Slot()
    def run(self) -> None:
        self._running = True
        try:
            self.recorder.start()
        except AudioError as e:
            self.sig.error.emit(str(e))
            self.sig.ended.emit()
            return

        pos = 0
        last_partial = 0.0
        last_state = ""
        last_loss_check = 0.0
        try:
            while self._running:
                data, pos = self.recorder.ring.read_since(pos)
                if data.size:
                    for seg in self.segmenter.feed(data):
                        self.transcriber.submit_final(seg)

                    state = self.segmenter.state.value
                    if state != last_state:
                        self.sig.vad_state.emit(state)
                        last_state = state

                    now = time.monotonic()
                    if now - last_partial >= self.live_interval:
                        active = self.segmenter.peek_active()
                        if active is not None:
                            self.transcriber.submit_partial(active)
                        last_partial = now

                self.sig.level.emit(self.recorder.level)
                tick = time.monotonic()
                if tick - last_loss_check >= 1.0:
                    last_loss_check = tick
                    lost = self.recorder.lost_ms
                    if lost > 200 and not self._loss_warned:
                        # 机器忙，录音有缺口。用户不看到就以为'全录上了'
                        self._loss_warned = True
                        self._loss_total += lost
                        self._on_loss_warn(lost)
                # 采集是 PortAudio 自己的线程在跑，这里只是轮询环形缓冲。
                # 醒得再勤也不会让音频更准，只是白烧 CPU——老电脑上这一点
                # 在意明显。60ms 相对 1.2 秒的预览间隔完全够。
                time.sleep(POLL_INTERVAL_IDLE)
        finally:
            # 停录音时把没说完的段定稿，否则最后半句会丢
            for seg in self.segmenter.flush():
                self.transcriber.submit_final(seg)
            self.recorder.stop()
            self.sig.ended.emit()


class ModelLoader(QObject):
    """后台加载模型。加载要好几秒，不能卡住界面。

    整个程序生命周期只用这一个线程，旧模型也在这上面释放。
    原因：whisper/ggml 的 CUDA 上下文被并发操作会直接段错误
    （实测 "access violation reading 0x15"）——切换模型时如果靠
    Python 的 GC 在 GUI 线程上释放旧模型，就会和这里的加载撞上。
    """

    loaded = Signal(str)   # whisper.cpp 版本
    failed = Signal(str)
    trigger = Signal()     # 跨线程请求加载
    release = Signal()     # 跨线程请求释放当前模型

    def __init__(self) -> None:
        super().__init__()
        self._pending: Transcriber | None = None
        self._current: Transcriber | None = None
        self.trigger.connect(self.run)
        self.release.connect(self._release)

    def submit(self, transcriber: Transcriber) -> None:
        self._pending = transcriber

    @Slot()
    def _release(self) -> None:
        if self._current is not None:
            self._current.stop()
            self._current = None

    @Slot()
    def run(self) -> None:
        pending = self._pending
        self._pending = None
        if pending is None:
            return
        try:
            # 顺序必须是"先加载新的、再释放旧的"，不能反。
            # whisper_free() 之后 ggml 的全局后端注册表就废了，
            # 同进程内再加载任何模型都会以 0xc000001d（非法指令）崩掉。
            # 先建后放则两个 context 可以共存，实测来回切没问题。
            # 代价是切换瞬间两个模型同时占着显存/内存，8G 卡上完全够。
            version = pending.load_blocking()
            old = self._current
            self._current = pending
            self.loaded.emit(version)
            if old is not None:
                old.stop()   # 新模型确认可用后才丢旧的
        except Exception as e:
            # 加载失败：旧模型原封不动还能继续用
            try:
                pending.stop()
            except Exception:
                pass
            self.failed.emit(str(e))


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.cfg = Config.load()
        self.transcriber: Transcriber | None = None
        self.audio_thread: QThread | None = None
        self.audio_worker: AudioWorker | None = None
        self.sig = Signals()
        self.rows: list[tuple[int, int, str]] = []  # 导出 srt 用
        self.recording = False
        self._stopping = False   # 已请求停止，等录音线程收尾
        self._error = ""         # 本次会话的错误，收尾时别覆盖掉
        self._rec_start = 0.0
        self._tick_timer: QTimer | None = None
        self._toast_timer: QTimer | None = None
        self._loss_warned = False
        self._loss_total = 0.0
        self.loader_thread: QThread | None = None
        self.loader: ModelLoader | None = None
        self._loading_model = ""
        self._device_pref = self.cfg.device
        self._vad_state = "idle"

        # 尽早套主题，否则控件会先以系统默认样式建好再被重绘
        self.theme = theme.apply(self.cfg.theme)
        self._base_title = branding.WINDOW_TITLE
        self.setWindowTitle(self._base_title)
        self.setMinimumSize(720, 460)
        self.resize(self.cfg.window_w, self.cfg.window_h)
        self._build_ui()
        self._connect()
        self._check_assets()

    # -- 界面 --------------------------------------------------------------

    def _build_ui(self) -> None:
        central = QWidget()
        root = QVBoxLayout(central)
        root.setSpacing(10)

        # 标题行：品牌 + 一句话说明
        head = QHBoxLayout()
        brand = QLabel(branding.APP_NAME)
        bf = QFont(self.font())
        bf.setPointSize(15)
        bf.setBold(True)
        brand.setFont(bf)
        head.addWidget(brand)
        tag = QLabel(branding.APP_NAME_CN + " · " + branding.APP_TAGLINE)
        tag.setStyleSheet(f"color:{self.theme.text_muted};")
        head.addWidget(tag)
        head.addStretch(1)
        # 录音指示 + 计时放在抬头右侧：口述时眼睛主要在文字区，
        # 计时埋在底部状态行会被忽略
        self.rec_pill = QLabel("")
        self.rec_pill.setVisible(False)
        head.addWidget(self.rec_pill)

        ver = QLabel("v" + branding.APP_VERSION)
        ver.setStyleSheet(f"color:{self.theme.text_muted};")
        head.addWidget(ver)
        root.addLayout(head)

        # 第一行：设备与参数
        row1 = QHBoxLayout()
        row1.addWidget(QLabel("麦克风"))
        self.device_combo = QComboBox()
        self.device_combo.setMinimumWidth(260)
        row1.addWidget(self.device_combo)
        row1.addSpacing(12)

        row1.addWidget(QLabel("语言"))
        self.lang_combo = QComboBox()
        self.lang_combo.addItems(["中文", "English", "自动检测"])
        self.lang_combo.setCurrentIndex(0)
        row1.addWidget(self.lang_combo)
        row1.addSpacing(12)

        row1.addWidget(QLabel("灵敏度"))
        self.sens_slider = QSlider(Qt.Horizontal)
        self.sens_slider.setRange(0, 3)
        self.sens_slider.setValue(self.cfg.vad_aggressiveness)
        self.sens_slider.setMaximumWidth(110)
        self.sens_slider.setToolTip(
            "越往右越严格，噪声越不容易被当成说话\n"
            "（四档对语音检出率没有影响，只影响噪声误判）"
        )
        self.sens_label = QLabel(str(self.cfg.vad_aggressiveness))
        row1.addWidget(self.sens_slider)
        row1.addWidget(self.sens_label)
        row1.addSpacing(12)

        row1.addWidget(QLabel("模型"))
        self.model_combo = QComboBox()
        self.model_combo.setMinimumWidth(170)
        self.model_combo.setToolTip("切换模型需要重新加载（几秒）")
        row1.addWidget(self.model_combo)
        self._refresh_models()
        row1.addStretch(1)
        root.addLayout(row1)

        # 第二行：主按钮 + 工具按钮
        row2 = QHBoxLayout()
        self.start_btn = QPushButton("开始录音")
        self.start_btn.setMinimumHeight(42)
        self.start_btn.setMinimumWidth(150)
        btn_font = QFont(self.font())
        btn_font.setPointSize(12)
        btn_font.setBold(True)
        self.start_btn.setFont(btn_font)
        row2.addWidget(self.start_btn)

        self.clear_btn = QPushButton("清空")
        self.copy_btn = QPushButton("复制全文")
        self.export_btn = QPushButton("导出…")
        self.settings_btn = QPushButton("设置…")
        self.settings_btn.setToolTip("断句、预览间隔、繁简与标点等")
        row2.addSpacing(16)
        row2.addWidget(self.clear_btn)
        row2.addWidget(self.copy_btn)
        row2.addWidget(self.export_btn)
        row2.addStretch(1)
        row2.addWidget(self.settings_btn)
        row2.addStretch(1)
        root.addLayout(row2)

        # 模型加载指示。加载要几秒，只给一行状态文字用户容易以为卡死了
        self.busy_bar = QProgressBar()
        self.busy_bar.setRange(0, 0)  # 0,0 == 不确定进度（滚动条）
        self.busy_bar.setMaximumHeight(4)
        self.busy_bar.setTextVisible(False)
        self.busy_bar.setVisible(False)
        root.addWidget(self.busy_bar)

        # 正文
        self.text_edit = QTextEdit()
        self.text_edit.setPlaceholderText("识别结果会显示在这里…")
        self.text_edit.setLineWrapMode(QTextEdit.WidgetWidth)
        root.addWidget(self.text_edit, 1)

        # 覆盖在正文上的两层：录音时的活插入点、空状态引导
        self.live_mark = LiveInsertionMark(self.text_edit)
        self.empty_state = EmptyStateOverlay(self.text_edit, self.theme)

        # 实时预览
        # 预览做成正文的视觉延续：左侧一条竖线 + 缩进 + 灰斜体，
        # 一眼能看出"这句还没定稿"，而不是另一块独立内容。
        preview_box = QWidget()
        pv = QHBoxLayout(preview_box)
        pv.setContentsMargins(0, 0, 0, 0)
        pv.setSpacing(8)
        pv_bar = QFrame()
        pv_bar.setFrameShape(QFrame.VLine)
        pv_bar.setStyleSheet(f"color:{self.theme.border};")
        pv.addWidget(pv_bar)
        self._pv_bar = pv_bar
        self.preview = QLabel("")
        self.preview.setWordWrap(True)
        self.preview.setStyleSheet(
            f"color:{self.theme.text_muted}; font-style:italic; padding:4px 0;"
        )
        pv.addWidget(self.preview, 1)
        self.preview_box = preview_box
        self.preview_box.setVisible(False)
        self.preview_box.setMaximumHeight(76)
        root.addWidget(self.preview_box)

        # 底部状态条
        row3 = QHBoxLayout()
        self.level_bar = LevelMeter()
        self.level_bar.set_theme(self.theme)
        self.level_bar.setToolTip(
            "输入电平（dBFS，刻度 -60 ~ 0）。\n"
            "长时间贴着表底说明麦克风没在收音：换设备，或调高系统输入音量。\n"
            "顶端红线是峰值保持，会缓慢回落。"
        )
        row3.addWidget(QLabel("音量"))
        row3.addWidget(self.level_bar)

        self.vad_dot = QLabel("● 静音")
        self.vad_dot.setStyleSheet("color:#999999;")
        row3.addSpacing(16)
        row3.addWidget(self.vad_dot)

        self.rtf_label = QLabel("")
        row3.addSpacing(16)
        row3.addWidget(self.rtf_label)

        self.elapsed_label = QLabel("")
        self.elapsed_label.setStyleSheet("font-family:Consolas,monospace;")
        row3.addSpacing(16)
        row3.addWidget(self.elapsed_label)
        row3.addStretch(1)

        self.status_label = QLabel("准备中…")
        row3.addWidget(self.status_label, 1)

        # 独立的反馈位：复制/导出这类"操作完成"不能挤占状态栏，
        # 状态栏随时会被设备信息、过滤统计覆盖，用户来不及看
        self.toast_label = QLabel("")
        self.toast_label.setVisible(False)
        row3.addWidget(self.toast_label)
        root.addLayout(row3)

        self.setCentralWidget(central)
        self._add_shortcuts()

    def _add_shortcuts(self) -> None:
        act = QAction(self)
        act.setShortcut(QKeySequence("Ctrl+R"))
        act.setToolTip("开始 / 停止录音")
        act.triggered.connect(self.toggle_recording)
        self.addAction(act)

        # 复制全文用 Ctrl+Shift+C，不要占用 Ctrl+C。
        # 正文框是可编辑的（用来改错字），绑走标准复制键会让用户
        # 没法复制选中的片段——那是编辑时的基本操作。
        act = QAction(self)
        act.setShortcut(QKeySequence("Ctrl+Shift+C"))
        act.setToolTip("复制全文")
        act.triggered.connect(self.copy_all)
        self.addAction(act)

        act = QAction(self)
        act.setShortcut(QKeySequence("Ctrl+S"))
        act.setToolTip("导出")
        act.triggered.connect(self.export)
        self.addAction(act)

    def _connect(self) -> None:
        self.start_btn.clicked.connect(self.toggle_recording)
        self.clear_btn.clicked.connect(self.clear_all)
        self.copy_btn.clicked.connect(self.copy_all)
        self.export_btn.clicked.connect(self.export)
        self.settings_btn.clicked.connect(self.open_settings)
        self.sens_slider.valueChanged.connect(
            lambda v: (self.sens_label.setText(str(v)), self._on_param_changed())
        )
        # currentIndexChanged 传的是行号，槽里要的是文件名，这里做一次转换。
        # 直接连过去会把 0 当成 falsy 的文件名，切换静默失效。
        self.model_combo.currentIndexChanged.connect(
            lambda i: self._on_model_selected(self.model_combo.itemData(i))
        )

        self.sig.level.connect(self._on_level)
        self.sig.vad_state.connect(self._on_vad_state)
        self.sig.final_text.connect(self._on_final_text)
        self.sig.partial_text.connect(self._on_partial_text)
        self.sig.error.connect(self._on_error)
        self.sig.status.connect(self._on_status)
        self.sig.rtf.connect(self._on_rtf)
        self.sig.ended.connect(self._on_stream_ended)

    # -- 启动检查 ----------------------------------------------------------

    def _check_assets(self) -> None:
        plat = platforms.current()
        missing = []
        if not self.cfg.model_path().is_file():
            missing.append(f"模型 {self.cfg.model}")
        if not (platforms.app_root() / "vendor" / "whisper" / "Release"
                / plat.whisper_path_name).is_file():
            missing.append("whisper.cpp 运行时")
        if not missing:
            self._refresh_devices()
            self._start_model_load()
            return

        # 打包后的 exe 里没有 tools/fetch_assets.py，得能在程序内下载
        detail = "将下载约 500MB（运行时 40MB + 模型 466MB），只需一次，之后完全离线运行。"
        if platforms.is_frozen():
            QMessageBox.information(
                self, "首次运行",
                "还需要下载：" + "、".join(missing) + "。" + detail)
            self.start_btn.setEnabled(False)
            self._offer_download()
            return

        QMessageBox.warning(
            "以下内容还没准备好："
            + chr(10).join(f"  · {m}" for m in missing)
            + "请在项目目录运行： python tools/fetch_assets.py",
        )
        self.start_btn.setEnabled(False)
        self.status_label.setText("缺少必要文件")

    def _offer_download(self) -> None:
        # 弹一个带进度的下载窗口
        root = platforms.app_root()
        want_cuda = device.nvidia_present()
        dlg = DownloadDialog(self, self.cfg.model, want_cuda)
        dlg.start_download(root)
        dlg.exec()
        if dlg.ok:
            self._refresh_devices()
            self._start_model_load()

    def _refresh_devices(self) -> None:
        self.device_combo.blockSignals(True)
        self.device_combo.clear()
        try:
            devs = list_input_devices()
        except Exception as e:
            QMessageBox.warning(self, "枚举设备失败", str(e))
            devs = []
        if not devs:
            self.device_combo.addItem("没有找到可用的输入设备", -1)
        for d in devs:
            self.device_combo.addItem(d.label, d.index)
        default_idx = default_input_index()
        idx = self.device_combo.findData(default_idx)
        self.device_combo.setCurrentIndex(idx if idx >= 0 else 0)
        self.device_combo.blockSignals(False)

    def _auto_pick_model(self) -> None:
        """首次运行按可用内存挑模型。

        老电脑直接加载默认模型会被系统换页，表现为界面卡、录音丢字。
        这里按实测占用表选一个装得下的。
        """
        have = [p.name for p in self.cfg.available_models()]
        if not have:
            return
        rec = recommend.recommend(preferred=None, have=have)
        idx = self.model_combo.findData(rec.model)
        if idx >= 0:
            self.model_combo.setCurrentIndex(idx)
        self.cfg.model = rec.model
        self.cfg.save()
        if rec.tight:
            self._on_status(f"内存偏紧（可用 {rec.available_mb}MB），已选 {rec.model}")

    def _refresh_models(self) -> None:
        """列出 models/ 下已有的权重。"""
        self.model_combo.blockSignals(True)
        self.model_combo.clear()
        for p in self.cfg.available_models():
            self.model_combo.addItem(f"{p.name}  —  {recommend.describe(p.name)}",
                                    p.name)
        if self.model_combo.count() == 0:
            self.model_combo.addItem("(models/ 目录为空)", "")
        idx = self.model_combo.findData(self.cfg.model)
        self.model_combo.setCurrentIndex(idx if idx >= 0 else 0)
        self.model_combo.blockSignals(False)

    def _on_model_selected(self, name: str) -> None:
        """切换模型。加载要几秒，必须后台做，且不能正在录音。"""
        if not name or name == self.cfg.model:
            return
        if self.recording or self._stopping:
            QMessageBox.information(self, "正在录音", "先停止录音再切换模型。")
            self._refresh_models()
            return
        self.cfg.model = name
        self.cfg.save()
        self._on_status(f"正在加载 {name}…")
        self._start_model_load()

    def _on_param_changed(self) -> None:
        """参数只在下次录音生效，给个明确提示，别让人以为没反应。"""
        if self.recording or self._stopping:
            return
        self._on_status("参数已改，下次录音生效")

    def _start_model_load(self) -> None:
        self.busy_bar.setVisible(True)
        self.status_label.setText("正在加载模型…")
        self.start_btn.setEnabled(False)
        info = device.detect(self.cfg.device)
        self.device_info = info
        tr = Transcriber(
            model_path=self.cfg.model_path(),
            use_gpu=info.use_gpu,
            n_threads=device.default_threads(),
            language=self._language_code(),
            no_speech_thold=self.cfg.no_speech_thold,
            vendor_dir=info.vendor_dir,
            live_interval_s=self.cfg.live_interval_ms / 1000.0,
            simplify=self.cfg.simplify_chinese,
            fix_punctuation=self.cfg.fix_punctuation,
        )
        # self.transcriber 只在新模型加载成功后才指向它，
        # 加载期间 start_recording 会因为它还是 None 而拒绝开始
        self._loading_model = tr.model_path.name

        if self.loader_thread is None:
            self.loader_thread = QThread(self)
            self.loader = ModelLoader()
            self.loader.moveToThread(self.loader_thread)
            self.loader.loaded.connect(self._on_model_loaded)
            self.loader.failed.connect(self._on_model_failed)
            self.loader_thread.start()

        self.loader.submit(tr)
        self.loader.trigger.emit()

    # -- 录音控制 ----------------------------------------------------------

    def toggle_recording(self) -> None:
        if self.recording:
            self.stop_recording()
        else:
            self.start_recording()

    def start_recording(self) -> None:
        # 按钮和快捷键可能几乎同时触发，没有这层保护会起两个录音线程，
        # 旧的被引用覆盖掉、再也停不下来
        if self.recording or self._stopping:
            return
        if self.transcriber is None or self.transcriber._engine is None:
            what = f"（正在加载 {self._loading_model}）" if self._loading_model else ""
            self._on_status("模型还没就绪" + what + "，请稍候")
            return
        dev_idx = self.device_combo.currentData()
        if dev_idx is None:
            dev_idx = -1

        vcfg = VADConfig.from_config(self.cfg)
        vcfg.aggressiveness = self.sens_slider.value()  # 滑块优先于配置文件
        segmenter = Segmenter(vcfg)
        recorder = MicRecorder(device_index=dev_idx)

        self.transcriber.language = self._language_code()
        self.transcriber.start(self._on_transcribe_output, self._on_transcribe_error)

        self.audio_thread = QThread(self)
        self.audio_worker = AudioWorker(
            recorder, segmenter, self.transcriber, self.cfg.live_interval_ms, self.sig
        )
        self.audio_worker.moveToThread(self.audio_thread)
        self.audio_thread.started.connect(self.audio_worker.run)
        self.audio_thread.start()

        self.recording = True
        self._stopping = False
        self._error = ""
        self._loss_warned = False
        self._loss_total = 0.0
        self._rec_start = time.monotonic()
        self.setWindowTitle("[● 录音中] " + self._base_title)
        self.start_btn.setText("停止录音")
        self.start_btn.setStyleSheet(
            f"background:{self.theme.danger}; color:#ffffff;"
            f"border:1px solid {self.theme.danger}; border-radius:8px;"
        )
        self._set_controls_enabled(False)
        self.live_mark.set_active(True)
        self.rec_pill.setVisible(True)
        self._update_rec_pill("00:00")
        self._on_status(f"录音中 · {self.device_info.label} · {self.device_info.detail}")
        self._on_vad_state("idle")
        self._tick_timer = QTimer(self)
        self._tick_timer.timeout.connect(self._on_tick)
        self._tick_timer.start(500)
        # 峰值保持线的回落，独立于计时器
        self.peak_timer = QTimer(self)
        self.peak_timer.timeout.connect(self.level_bar.decay)
        self.peak_timer.start(90)

    def _on_tick(self) -> None:
        """录音时长。定时器在停止时销毁。"""
        el = int(time.monotonic() - self._rec_start)
        stamp = f"{el // 60:02d}:{el % 60:02d}"
        self.elapsed_label.setText(stamp)
        self._update_rec_pill(stamp)

    def stop_recording(self) -> None:
        """请求停止。

        这里绝不能阻塞 GUI 线程。录音线程收尾要走 VAD flush + 录音关闭，
        之前用 wait(4000) 同步等，界面会整块卡住，"正在处理最后一段…"
        连绘制都来不及。改成只发停止信号，界面复位交给
        _on_stream_ended 收尾。
        """
        if not self.recording or self._stopping:
            return
        self._stopping = True
        self.start_btn.setEnabled(False)
        self.start_btn.setText("正在停止…")
        self._on_status("正在处理最后一段…")
        if self.audio_worker:
            self.audio_worker.stop()
        if self.audio_thread:
            # quit() 只让事件循环退出，run() 槽仍会跑完并在 finally 里
            # emit ended，由那个信号完成剩下的收尾
            self.audio_thread.quit()

    def _on_stream_ended(self) -> None:
        """录音线程收尾，等推理队列跑空后恢复界面。"""
        if not self.recording and not self._stopping:
            # 麦克风打开失败时也会走这里，但那时不该覆盖错误信息
            return

        if self.transcriber:
            self.transcriber.wait_idle(timeout=60)
        if self._tick_timer:
            self._tick_timer.stop()
            self._tick_timer.deleteLater()
            self._tick_timer = None
        if getattr(self, "peak_timer", None):
            self.peak_timer.stop()
            self.peak_timer.deleteLater()
            self.peak_timer = None
        self.level_bar.reset()
        if self.audio_thread:
            self.audio_thread.wait(5000)
            self.audio_thread = None
        self.audio_worker = None

        self.recording = False
        self._stopping = False
        self.setWindowTitle(self._base_title)
        self.start_btn.setText("开始录音")
        self.start_btn.setStyleSheet("")
        self.start_btn.setEnabled(True)
        self._set_controls_enabled(True)
        self._on_vad_state("idle")
        self.level_bar.reset()
        self.elapsed_label.setText("")
        self.live_mark.set_active(False)
        self.rec_pill.setVisible(False)
        # 预览只是临时结果，停止时直接丢掉。把它拼进状态栏是没意义的：
        # 停止录音时 Segmenter.flush() 已经把没说完的段走正常识别定稿了，
        # 剩下的预览要么已被定稿覆盖，要么是没通过过滤的碎片
        # （实测安静环境下残留过 "（字幕：贝尔）" 这种幻觉）。
        self.preview.clear()
        self.preview_box.setVisible(False)

        if self._error:
            # 出过错就保留错误信息，别用"已停止"盖掉
            self._on_status("已停止 · " + self._error)
            return

        # 状态栏只报有意义的数字：处理速度和被过滤掉的量
        bits = [f"已停止 · {len(self.rows)} 段"]
        if self._loss_total > 500:
            bits.append(f"⚠ 有 {self._loss_total / 1000:.1f}s 音频因机器繁忙没录上")
        if self.transcriber:
            if self.transcriber.dropped_hallucinations:
                bits.append(f"过滤幻觉 {self.transcriber.dropped_hallucinations}")
            if self.transcriber.dropped_fragments:
                bits.append(f"过滤碎片 {self.transcriber.dropped_fragments}")
            if self.transcriber.previews_suppressed:
                bits.append(f"跳过预览 {self.transcriber.previews_suppressed}")
        self._on_status(" · ".join(bits))

    def _set_controls_enabled(self, enabled: bool) -> None:
        # 录音中这些都不能改：已经开了线程，中途改不会生效，
        # 还会让人以为改了就有用
        for w in (self.device_combo, self.sens_slider, self.lang_combo,
                  self.model_combo, self.settings_btn):
            w.setEnabled(enabled)
        # 清空和导出在录音中也锁掉，避免混进未识别的内容
        self.clear_btn.setEnabled(enabled)
        self.export_btn.setEnabled(enabled)

    # -- 推理回调（推理线程 -> GUI 线程）-----------------------------------

    def _on_transcribe_output(self, out: Output) -> None:
        if out.kind == "final":
            self.sig.final_text.emit(
                out.result.text,
                int(out.segment.start_sample * 1000 / SAMPLE_RATE),
                int(out.segment.end_sample * 1000 / SAMPLE_RATE),
            )
        else:
            self.sig.partial_text.emit(out.result.text)
        if out.result.rtf > 0:
            self.sig.rtf.emit(out.result.rtf)

    def _on_transcribe_error(self, msg: str) -> None:
        self.sig.error.emit(msg)

    # -- GUI 槽 ------------------------------------------------------------

    @Slot(float)
    def _on_level(self, level: float) -> None:
        self.level_bar.set_level(level)

    def _near_bottom(self, tolerance_px: int = 40) -> bool:
        # 滚动条是否已经贴着底部，用来决定要不要跟随新内容。
        # 正文框是可编辑的，文本一长用户就会往回翻着改字，
        # 此时每来一段新识别就强制滚到底部很烦人。
        bar = self.text_edit.verticalScrollBar()
        return bar.value() >= bar.maximum() - tolerance_px

    @Slot(str)
    def _on_loss_warn(self, lost_ms: float) -> None:
        """录音出现缺口时提示一次。

        机器一忙，PortAudio 的回调就赶不上截止时间，丢掉的音频等于丢掉的
        字。不告诉用户的话，他只会觉得"这软件识别得不准"。
        """
        self.toast_label.setText(f"⚠ 机器繁忙，约 {lost_ms / 1000:.1f} 秒音频没录上")
        self.toast_label.setStyleSheet(
            f"color:{self.theme.warning}; font-weight:600;")
        self.toast_label.setVisible(True)
        if self._toast_timer is None:
            self._toast_timer = QTimer(self)
            self._toast_timer.setSingleShot(True)
            self._toast_timer.timeout.connect(self._hide_toast)
        self._toast_timer.start(6000)

    def _toast(self, text: str, msec: int = 4000) -> None:
        """短暂反馈，不覆盖常驻状态。

        复制、导出、清空这些操作的结果放这里，几秒后自动消失；
        状态栏留给"当前处于什么状态"这类需要持续可见的信息。
        """
        self.toast_label.setText(text)
        self.toast_label.setStyleSheet(
            f"color:{self.theme.success}; padding-right:4px;")
        self.toast_label.setVisible(True)
        if self._toast_timer is None:
            self._toast_timer = QTimer(self)
            self._toast_timer.setSingleShot(True)
            self._toast_timer.timeout.connect(self._hide_toast)
        self._toast_timer.start(msec)

    def _hide_toast(self) -> None:
        self.toast_label.setVisible(False)
        self.toast_label.clear()

    def _update_rec_pill(self, stamp: str) -> None:
        """抬头上的录音指示牌。"""
        t = self.theme
        self.rec_pill.setText(f"  ●  录音中  {stamp}  ")
        self.rec_pill.setStyleSheet(
            f"color:{t.accent_text}; background:{t.danger};"
            f"border-radius:11px; padding:3px 10px; font-weight:600;")

    def _on_vad_state(self, state: str) -> None:
        self._vad_state = state
        if state == "speech":
            self.vad_dot.setText("● 说话中")
            self.vad_dot.setStyleSheet(f"color:{self.theme.success}; font-weight:bold;")
        elif state == "tail":
            self.vad_dot.setText("● 停顿中")
            self.vad_dot.setStyleSheet(f"color:{self.theme.warning};")
        else:
            self.vad_dot.setText("● 静音")
            self.vad_dot.setStyleSheet(f"color:{self.theme.text_muted};")

    @Slot(str, int, int)
    def _on_final_text(self, text: str, start_ms: int, end_ms: int) -> None:
        cursor = self.text_edit.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        if self.text_edit.toPlainText():
            cursor.insertText("\n")
        cursor.insertText(text)
        self.text_edit.setTextCursor(cursor)
        # 只在用户本来就贴着底部时跟随。正文框是可编辑的，长文本里
        # 往回翻着改字时，不该被新识别的段落一次次拽回底部。
        if self._near_bottom():
            self.text_edit.ensureCursorVisible()
        self.rows.append((start_ms, end_ms, text))
        # 正文有更新，说明上一句已经定稿，预览可以收了。
        # 文本一并清掉：留着陈旧内容的话，将来任何让预览重新显示的路径
        # 都会把上一句的残留亮出来。停止录音时也是这么处理的，保持一致。
        self.preview.clear()
        self.preview_box.setVisible(False)

    @Slot(str)
    def _on_partial_text(self, text: str) -> None:
        self.preview.setText(text)
        self.preview_box.setVisible(True)

    @Slot(str)
    def _on_error(self, msg: str) -> None:
        # 记下来：录音线程收尾时会更新状态栏，不能把刚报的错误冲掉
        self._error = msg.splitlines()[0]
        self._on_status("出错：" + self._error)
        QMessageBox.warning(self, "出错", msg)

    @Slot(str)
    def _on_status(self, text: str) -> None:
        self.status_label.setText(text)

    @Slot(float)
    def _on_rtf(self, rtf: float) -> None:
        if rtf > 0:
            speed = "快于实时" if rtf < 1 else f"慢于实时 {rtf:.1f}倍"
            self.rtf_label.setText(f"速度 {1 / rtf:.1f}x {speed}")

    @Slot(str)
    def _on_model_loaded(self, version: str) -> None:
        # 加载成功才把 self.transcriber 指过去
        self.busy_bar.setVisible(False)
        self.transcriber = self.loader._current
        self._loading_model = ""
        self.start_btn.setEnabled(True)
        self._set_controls_enabled(True)
        name = self.cfg.model
        self._on_status(f"就绪 · {name} · whisper.cpp {version} · {self.device_info.label}")

    @Slot(str)
    def _on_model_failed(self, msg: str) -> None:
        self._loading_model = ""
        self.busy_bar.setVisible(False)
        self._on_status("模型加载失败")
        # 旧模型可能还在用，start 是否可用取决于有没有可用的 engine
        if self.transcriber is not None and self.transcriber._engine is not None:
            self.start_btn.setEnabled(True)
            self._set_controls_enabled(True)
        QMessageBox.critical(self, "模型加载失败", msg)

    # -- 工具 --------------------------------------------------------------

    def _language_code(self) -> str:
        idx = self.lang_combo.currentIndex()
        return {0: "zh", 1: "en", 2: "auto"}.get(idx, "zh")

    def set_theme(self, name: str) -> None:
        """切换浅色 / 深色。

        QSS 是挂在 QApplication 上的，所以重设一次全局生效；
        自绘控件（电平表、预览竖线、VAD 灯）得单独通知。
        """
        if name == self.cfg.theme:
            return
        self.cfg.theme = name
        self.theme = theme.apply(name)
        self.cfg.save()
        self.level_bar.set_theme(self.theme)
        self.empty_state.set_theme(self.theme)
        self.preview.setStyleSheet(
            f"color:{self.theme.text_muted}; font-style:italic; padding:4px 0;"
        )
        if hasattr(self, "_pv_bar"):
            self._pv_bar.setStyleSheet(f"color:{self.theme.border};")
        self._on_vad_state(self._vad_state)

    def open_settings(self) -> None:
        dlg = SettingsDialog(self.cfg, self)
        if dlg.exec() != SettingsDialog.Accepted:
            return
        if getattr(dlg, "theme_changed", False):
            self.set_theme(dlg.theme_combo.currentData())
        # 设备偏好变了要重新探测；其余参数下次录音生效
        if self.cfg.device != self._device_pref:
            self._device_pref = self.cfg.device
            self._on_status("设备设置已改，正在重新加载…")
            self._start_model_load()
        else:
            self._on_status("设置已保存，下次录音生效")
        self.cfg.save()

    def clear_all(self) -> None:
        if self.recording:
            QMessageBox.information(
                self, "正在录音", "先停止录音再清空，否则会混进新的内容。"
            )
            return
        if not self.rows and not self.text_edit.toPlainText().strip():
            return
        # 清空不可撤销，而且时间轴一起没了，误点代价不小
        ret = QMessageBox.question(
            self,
            "确认清空",
            f"清空全部 {len(self.rows)} 段识别结果？此操作无法撤销。",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if ret != QMessageBox.Yes:
            return
        self.text_edit.clear()
        self.rows.clear()
        self.preview.clear()
        self.preview_box.setVisible(False)
        self._toast("已清空")

    def copy_all(self) -> None:
        QApplication.clipboard().setText(self.text_edit.toPlainText())
        self._toast("已复制全文到剪贴板")

    def export(self) -> None:
        text = self.text_edit.toPlainText().strip()
        if not text:
            QMessageBox.information(self, "没有内容", "还没有可导出的文本。")
            return
        if self.recording or self._stopping:
            QMessageBox.information(
                self, "正在录音",
                "录音过程中导出会漏掉还没识别的内容。\n建议先点「停止录音」再导出。",
            )
            return
        default = time.strftime("transcript-%Y%m%d-%H%M%S")
        path, _ = QFileDialog.getSaveFileName(
            self, "导出", default, "字幕文件 (*.srt);;文本文件 (*.txt)"
        )
        if not path:
            return
        try:
            if path.lower().endswith(".txt"):
                Path(path).write_text(text, encoding="utf-8")
                note = ""
            else:
                srt, from_screen = self._to_srt()
                Path(path).write_text(srt, encoding="utf-8")
                note = "（含你的修改）" if from_screen else "（原始识别结果，编辑未被导出）"
        except Exception as e:
            QMessageBox.warning(self, "导出失败", str(e))
            return
        extra = note.strip("（）") if note else ""
        self._toast("已导出" + (f"（{extra}）" if extra else ""))

    def _to_srt(self) -> tuple[str, bool]:
        """生成 srt 内容。

        正文框是可编辑的，用户会改错字。但时间轴只和最初切出来的段绑定，
        没法跟着编辑走，所以按行号对齐：行数一致就用屏幕上的文字，
        这样用户的修改能进字幕；对不上就退回原始文本并说明。

        Returns:
            (srt 内容, 是否采用了屏幕上的文字)
        """
        lines = self.text_edit.toPlainText().splitlines()
        from_screen = bool(lines) and len(lines) == len(self.rows)
        texts = lines if from_screen else [r[2] for r in self.rows]

        cues = []
        for i, (start, end, _orig) in enumerate(self.rows, 1):
            text = texts[i - 1] if i - 1 < len(texts) else ""
            if not text.strip():
                continue
            cues.append(f"{len(cues) + 1}\n{_srt_ts(start)} --> {_srt_ts(end)}\n{text}\n")
        return "\n".join(cues), from_screen

    def closeEvent(self, event) -> None:
        self._persist_config()
        self._shutdown()
        super().closeEvent(event)

    def _shutdown(self) -> None:
        """按顺序收尾。关窗时阻塞等待是可以接受的。

        顺序很重要：
          1. 先让录音线程停下并等它真正结束（它还会往推理队列里投最后几段）
          2. 再等推理队列跑空
          3. 然后才释放模型
        早先 stop_recording 是同步的，顺序天然正确；改成非阻塞之后
        必须在这里显式等，否则会在录音线程还在投递任务时把引擎释放掉。
        """
        if self.recording or self._stopping:
            self.stop_recording()
            if self.audio_thread:
                self.audio_thread.wait(8000)
        if self._tick_timer:
            self._tick_timer.stop()
            self._tick_timer = None
        if self.transcriber:
            self.transcriber.wait_idle(timeout=30)

        # 模型必须在加载线程上释放，那里才有对应的 CUDA 上下文归属。
        # 用信号让它在自己线程里做，直接调用会退化成跨线程释放。
        # 电平表峰值保持的定时器
        if getattr(self, "peak_timer", None):
            self.peak_timer.stop()
            self.peak_timer.deleteLater()
            self.peak_timer = None

        if self.loader_thread is not None:
            # 不管线程是否还在跑都要等它退出，否则进程收尾时 Qt 会报
            # "QThread: Destroyed while thread is still running"
            if self.loader_thread.isRunning():
                if self.loader is not None:
                    self.loader.release.emit()
                self.loader_thread.quit()
            self.loader_thread.wait(30000)
            self.loader_thread = None
        elif self.transcriber:
            self.transcriber.shutdown()

    def _persist_config(self) -> None:
        try:
            self.cfg.vad_aggressiveness = self.sens_slider.value()
            self.cfg.silence_ms = max(200, self.cfg.silence_ms)
            idx = self.device_combo.currentData()
            if idx is not None:
                self.cfg.input_device = idx
            size = self.size()
            self.cfg.window_w, self.cfg.window_h = size.width(), size.height()
            self.cfg.save()
        except Exception:
            pass


def _srt_ts(ms: int) -> str:
    h, rem = divmod(max(0, ms), 3600000)
    m, rem = divmod(rem, 60000)
    s, msec = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{msec:03d}"


def main() -> int:
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )
    app = QApplication(sys.argv)
    app.setFont(QFont("Microsoft YaHei UI", 10))
    win = MainWindow()
    win.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
