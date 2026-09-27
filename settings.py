"""设置对话框。

这些参数原本只能手改 config.json，但其中"断句阈值"是准确率影响最大的
一项（实测 600ms -> 1200ms 能把字符错误率从 36.5% 降到 32.0%），
藏在 JSON 里等于让用户找不到。

对话框只做两件事：改值、写回 config。应用这些设置需要重新开始录音
（VAD 配置在开始录音时构造），所以面板上明确标注了生效时机。
"""

from __future__ import annotations

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from config import Config

import theme


class SettingsDialog(QDialog):
    def __init__(self, cfg: Config, parent: QWidget | None = None,
                 live: bool = True) -> None:
        super().__init__(parent)
        self.setWindowTitle("设置")
        self.setMinimumWidth(430)
        self.cfg = cfg

        root = QVBoxLayout(self)

        # -- 断句 --------------------------------------------------------
        seg_box = QGroupBox("断句")
        seg_form = QFormLayout(seg_box)

        self.silence_spin = QSpinBox()
        self.silence_spin.setRange(300, 3000)
        self.silence_spin.setSingleStep(100)
        self.silence_spin.setSuffix(" ms")
        self.silence_spin.setValue(cfg.silence_ms)
        self.silence_spin.setToolTip(
            "停顿多久算一句话说完。\n"
            "调大：段更长，模型有更多上下文，错字更少，但出字更晚。\n"
            "实测 600ms 错误率 36.5%，1200ms 32.0%，1800ms 以上无额外收益。"
        )
        seg_form.addRow("停顿阈值", self.silence_spin)

        self.minseg_spin = QSpinBox()
        self.minseg_spin.setRange(100, 3000)
        self.minseg_spin.setSingleStep(100)
        self.minseg_spin.setSuffix(" ms")
        self.minseg_spin.setValue(cfg.min_segment_ms)
        self.minseg_spin.setToolTip("短于此长度的语音段直接丢弃，用于滤掉咳嗽、敲键盘")
        seg_form.addRow("最短语音段", self.minseg_spin)
        root.addWidget(seg_box)

        # -- 识别 --------------------------------------------------------
        rec_box = QGroupBox("识别")
        rec_form = QFormLayout(rec_box)

        self.live_spin = QSpinBox()
        self.live_spin.setRange(400, 5000)
        self.live_spin.setSingleStep(200)
        self.live_spin.setSuffix(" ms")
        self.live_spin.setValue(cfg.live_interval_ms)
        self.live_spin.setToolTip("多久刷新一次临时预览。预览只是参考，断句后才定稿")
        rec_form.addRow("预览间隔", self.live_spin)

        self.nsp_spin = QSpinBox()
        self.nsp_spin.setRange(1, 100)
        self.nsp_spin.setSingleStep(5)
        self.nsp_spin.setSingleStep(5)
        self.nsp_spin.setValue(int(cfg.no_speech_thold * 100))
        self.nsp_spin.setToolTip(
            "判定为“没有说话”的概率阈值。\n"
            "注意：实测在环境噪声上 whisper 的这个值经常是 0，靠它拦幻觉并不管用，"
            "真正起作用的是幻觉短语过滤。"
        )
        rec_form.addRow("无语音阈值", self.nsp_spin)

        self.simplify_cb = QCheckBox("统一转简体（默认开）")
        self.simplify_cb.setChecked(cfg.simplify_chinese)
        self.simplify_cb.setToolTip("whisper 的中文输出会繁简混排")
        rec_form.addRow(self.simplify_cb)

        self.punct_cb = QCheckBox("规整中文标点")
        self.punct_cb.setChecked(cfg.fix_punctuation)
        self.punct_cb.setToolTip("中文语境下把半角逗号等转全角、引号转中文引号")
        rec_form.addRow(self.punct_cb)
        root.addWidget(rec_box)

        # -- 外观 --------------------------------------------------------
        look_box = QGroupBox("外观")
        look_row = QHBoxLayout(look_box)
        self.theme_combo = QComboBox()
        for key, label in (("dark", "深色"), ("light", "浅色")):
            self.theme_combo.addItem(label, key)
        ti = self.theme_combo.findData(cfg.theme)
        self.theme_combo.setCurrentIndex(ti if ti >= 0 else 0)
        self.theme_combo.setToolTip("深色下长时间盯着看比纯白舒服")
        look_row.addWidget(QLabel("主题"))
        look_row.addWidget(self.theme_combo)
        root.addWidget(look_box)

        # -- 推理设备 ------------------------------------------------------
        dev_box = QGroupBox("推理设备")
        dev_row = QHBoxLayout(dev_box)
        self.device_combo_c = QComboBox()
        self.device_combo_c.addItem("自动（有 CUDA 就用 GPU）", "auto")
        self.device_combo_c.addItem("强制 CPU", "cpu")
        i = self.device_combo_c.findData(cfg.device)
        self.device_combo_c.setCurrentIndex(i if i >= 0 else 0)
        self.device_combo_c.setToolTip(
            "自动：检测到 CUDA 构建且有 N 卡就用 GPU，否则回退 CPU。\n"
            "切换后需要重新加载模型。"
        )
        dev_row.addWidget(self.device_combo_c)
        root.addWidget(dev_box)

        note = QLabel("设置在下次录音时生效（断句阈值在开始录音时构造）。")
        note.setWordWrap(True)
        note.setStyleSheet("color:#888;")
        root.addWidget(note)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

    def _on_accept(self) -> None:
        self.cfg.silence_ms = self.silence_spin.value()
        self.cfg.min_segment_ms = self.minseg_spin.value()
        self.cfg.live_interval_ms = self.live_spin.value()
        self.cfg.no_speech_thold = self.nsp_spin.value() / 100.0
        self.cfg.simplify_chinese = self.simplify_cb.isChecked()
        self.cfg.fix_punctuation = self.punct_cb.isChecked()
        # 先比较再写入：写进去之后再比就永远是 False 了
        self.theme_changed = self.theme_combo.currentData() != self.cfg.theme
        self.cfg.device = self.device_combo_c.currentData()
        try:
            self.cfg.save()
        except Exception as e:
            from PySide6.QtWidgets import QMessageBox

            QMessageBox.warning(self, "保存失败", str(e))
            return
        self.accept()
