"""界面交互逻辑测试。

这些路径之前只靠手动点过，0.1.0 的事故说明"手动点过一次"不够——
测试必须在每次构建前自动跑。

用真实的 MainWindow，但把模型加载和录音换掉，只测逻辑不碰硬件。
"""

from __future__ import annotations

from pathlib import Path

import pytest
from PySide6.QtWidgets import QMessageBox

import app as appmod

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def win(qapp, monkeypatch):
    """一个不加载模型、不碰麦克风的 MainWindow。"""
    monkeypatch.setattr(QMessageBox, "warning", staticmethod(lambda *a, **k: QMessageBox.Ok))
    monkeypatch.setattr(QMessageBox, "question", staticmethod(lambda *a, **k: QMessageBox.Yes))
    monkeypatch.setattr(appmod.MainWindow, "_start_model_load", lambda self: None)
    w = appmod.MainWindow()
    yield w
    w.close()


# -- 文本累积 ---------------------------------------------------------------


def test_final_text_accumulates(win):
    win._on_final_text("第一句", 0, 1000)
    win._on_final_text("第二句", 1000, 2000)
    assert win.text_edit.toPlainText() == "第一句\n第二句"
    assert len(win.rows) == 2
    assert win.rows[0] == (0, 1000, "第一句")


def test_partial_does_not_touch_body(win):
    win._on_final_text("已定稿", 0, 1000)
    win._on_partial_text("临时预览")
    assert "临时预览" not in win.text_edit.toPlainText()
    assert not win.preview_box.isHidden()


def test_final_hides_preview(win):
    win._on_partial_text("临时")
    assert not win.preview_box.isHidden()
    win._on_final_text("定稿了", 0, 500)
    assert win.preview_box.isHidden()
    assert win.preview.text() == ""


def test_srt_uses_edited_text_when_lines_match(win):
    """用户改了字，导出的字幕要反映修改。"""
    win._on_final_text("今天下午三点", 0, 2000)
    win._on_final_text("我们开会", 2000, 4000)
    win.text_edit.setPlainText("今天下午三点钟\n我们开会")
    srt, from_screen = win._to_srt()
    assert from_screen is True
    assert "今天下午三点钟" in srt


def test_srt_falls_back_when_line_count_differs(win):
    win._on_final_text("一", 0, 1000)
    win._on_final_text("二", 1000, 2000)
    win.text_edit.setPlainText("一")          # 少了一行
    srt, from_screen = win._to_srt()
    assert from_screen is False
    assert "二" in srt                        # 退回原始内容


def test_srt_skips_blank_lines(win):
    win._on_final_text("甲", 0, 1000)
    win._on_final_text("乙", 1000, 2000)
    win._on_final_text("丙", 2000, 3000)
    win.text_edit.setPlainText("甲\n\n丙")
    srt, _ = win._to_srt()
    assert srt.count("-->") == 2


def test_srt_timestamp_format(win):
    win._on_final_text("测试", 3723456, 3726000)
    srt, _ = win._to_srt()
    assert "01:02:03,456 --> 01:02:06,000" in srt


# -- 清空 -------------------------------------------------------------------


def test_clear_resets_everything(win):
    win._on_final_text("内容", 0, 1000)
    win._on_partial_text("预览")
    win.clear_all()
    assert win.text_edit.toPlainText() == ""
    assert win.rows == []
    assert win.preview_box.isHidden()


def test_clear_blocked_while_recording(win):
    win._on_final_text("内容", 0, 1000)
    win.recording = True
    win.clear_all()
    assert win.text_edit.toPlainText() == "内容"   # 没被清掉
    win.recording = False


# -- 主题 -------------------------------------------------------------------


def test_theme_switch_updates_widgets(win):
    win.set_theme("light")
    assert win.theme.name == "light"
    assert win.level_bar._theme.name == "light"
    win.set_theme("dark")
    assert win.theme.name == "dark"
    assert win.level_bar._theme.name == "dark"


def test_vad_indicator_reacts(win):
    win._on_vad_state("speech")
    assert "说话中" in win.vad_dot.text()
    win._on_vad_state("idle")
    assert "静音" in win.vad_dot.text()
    assert win._vad_state == "idle"


# -- 状态机 -----------------------------------------------------------------


def test_start_recording_rejected_before_model_loaded(win):
    win.transcriber = None
    win.start_recording()
    assert win.recording is False
    assert "模型" in win.status_label.text()


def test_stop_is_guarded(win):
    """没在录音时调 stop 不能炸。"""
    win.stop_recording()
    assert win.recording is False


def test_tick_updates_elapsed(win, monkeypatch):
    win._rec_start = 0.0
    monkeypatch.setattr(appmod.time, "monotonic", lambda: 65.0)
    win._on_tick()
    assert win.elapsed_label.text() == "01:05"


def test_near_bottom_detects_position(win):
    bar = win.text_edit.verticalScrollBar()
    bar.setValue(bar.maximum())
    assert win._near_bottom() is True


def test_controls_locked_while_recording(win):
    win._set_controls_enabled(False)
    assert not win.model_combo.isEnabled()
    assert not win.export_btn.isEnabled()
    assert not win.settings_btn.isEnabled()
    win._set_controls_enabled(True)
    assert win.model_combo.isEnabled()
    assert win.export_btn.isEnabled()


# -- 配置 -------------------------------------------------------------------


def test_config_roundtrip(tmp_path):
    from config import Config

    p = tmp_path / "cfg.json"
    c = Config()
    c.silence_ms = 1500
    c.model = "ggml-base.bin"
    c.theme = "light"
    c.save(p)

    c2 = Config.load(p)
    assert c2.silence_ms == 1500
    assert c2.model == "ggml-base.bin"
    assert c2.theme == "light"


def test_config_ignores_unknown_keys(tmp_path):
    """老版本 config 里有新版本不认识的键，不能因此崩。"""
    import json

    from config import Config

    p = tmp_path / "cfg.json"
    p.write_text(json.dumps({"silence_ms": 900, "future_option": True}),
                 encoding="utf-8")
    c = Config.load(p)
    assert c.silence_ms == 900


def test_vad_config_from_config():
    """VAD 配置只能有一个来源，否则改了 config.json 也不生效。"""
    from config import Config
    from vad import VADConfig

    c = Config()
    c.silence_ms = 1500
    c.vad_aggressiveness = 1
    c.min_segment_ms = 700
    v = VADConfig.from_config(c)
    assert v.end_frames == 75        # 1500ms / 20ms
    assert v.aggressiveness == 1
    assert v.min_segment_ms == 700


# -- 口述体验相关的覆盖层 ---------------------------------------------------


def test_insertion_mark_hidden_when_idle(win):
    """不录音时不该有插入点指示，免得误以为还能往那儿输入。"""
    assert win.live_mark.isHidden()


def test_insertion_mark_cursor_rect_in_bounds(win, qapp):
    """插入点必须落在正文可视区内，否则会画到窗口外面。"""
    win._on_final_text("一段文字", 0, 1000)
    win.live_mark.set_active(True)
    win.text_edit.show()
    qapp.processEvents()
    rect = win.live_mark.cursor_rect()
    assert rect is not None, "拿不到插入点位置"
    x, y, _w, _h = rect
    inner = win.text_edit.contentsRect()
    assert inner.left() <= x <= inner.right(), f"插入点 x={x} 越界"
    assert inner.top() <= y <= inner.bottom(), f"插入点 y={y} 越界"
    win.live_mark.set_active(False)


def test_insertion_mark_visible_while_recording(win):
    win.live_mark.set_active(True)
    assert not win.live_mark.isHidden()
    win.live_mark.set_active(False)
    assert win.live_mark.isHidden()


def test_empty_state_tracks_content(win, qapp):
    """空文本显示引导，有内容后必须消失，否则会盖住识别结果。"""
    win.text_edit.setPlainText("")
    qapp.processEvents()
    win.empty_state._sync(win.text_edit)
    assert not win.empty_state.isHidden()
    win._on_final_text("有内容了", 0, 1000)
    assert win.empty_state.isHidden()


def test_rec_pill_toggle(win):
    win.rec_pill.setVisible(True)
    assert not win.rec_pill.isHidden()
    win.rec_pill.setVisible(False)
    assert win.rec_pill.isHidden()


def test_rec_pill_shows_timer(win):
    win._update_rec_pill("01:23")
    assert "01:23" in win.rec_pill.text()
    assert "录音中" in win.rec_pill.text()


def test_toast_does_not_touch_status(win):
    """操作反馈走独立位，不能把常驻状态冲掉。"""
    win._on_status("录音中 · GPU")
    win._toast("已复制全文到剪贴板")
    assert win.status_label.text() == "录音中 · GPU", "toast 覆盖了状态栏"
    assert "已复制" in win.toast_label.text()


def test_toast_texts_exist(win):
    win.copy_all()
    assert "已复制" in win.toast_label.text()


# -- 停止录音的收尾不能阻塞界面 ---------------------------------------------


class _BusyTranscriber:
    """一个永远 busy 的假推理器。"""

    def __init__(self, busy=True):
        self.busy_value = busy
        self.stopped = False
        self.waited = False
        self.dropped_hallucinations = 0
        self.dropped_fragments = 0
        self.previews_suppressed = 0

    def busy(self):
        return self.busy_value

    def stop_worker(self, timeout=5.0):
        self.stopped = True
        return True

    def shutdown(self, timeout=5.0):
        pass

    def wait_idle(self, timeout=60.0):
        # 关闭程序时 _shutdown 会等队列排空，那里有等待是合理的。
        # 本组用例验证的是停止录音这条路径不等，靠的是下面两条断言：
        # _draining 为真、且推理线程尚未被停掉。
        self.waited = True


def test_stream_end_does_not_block_on_busy_queue(win):
    """队列没排空时，_on_stream_ended 必须立刻返回。

    旧实现在这里直接 wait_idle(60)，慢机器上界面整块冻住 60 秒——
    而它自己的注释写着"这里绝不能阻塞 GUI 线程"。现在改成轮询。
    """
    win.recording = True
    win._stopping = True
    win.transcriber = _BusyTranscriber()

    win._on_stream_ended()      # 返回即证明没阻塞

    assert win._draining, "应该进入等待推理排空的状态"
    assert win.recording is False, "界面应已复位"
    assert win.start_btn.isEnabled(), "按钮应已恢复可用"
    assert win.transcriber.stopped is False, "队列没排空时不该停推理线程"


def test_finish_runs_once_queue_drains(win):
    """队列排空后，_try_finish 完成收尾并停掉推理线程。"""
    win.recording = True
    win._stopping = True
    tr = _BusyTranscriber()
    win.transcriber = tr

    win._on_stream_ended()
    assert win._draining

    tr.busy_value = False
    win._try_finish()

    assert win._draining is False
    assert tr.stopped, "排空后必须停掉推理线程，否则下次录音会和它抢 context"


def test_idle_queue_finishes_immediately(win):
    """队列本来就是空的（麦克风打不开等场景），不该挂定时器。"""
    win.recording = True
    win._stopping = True
    tr = _BusyTranscriber(busy=False)
    win.transcriber = tr

    win._on_stream_ended()

    assert win._draining is False
    assert win._drain_timer is None, "空闲时不该起轮询定时器"
    assert tr.stopped
