"""真实端到端冒烟：点开始 -> 麦克风 -> VAD -> 识别 -> 文字入库。

之前所有 GUI 测试都绕开了真实录音（用假的 transcriber、直接调槽函数），
结果"点开始录音立刻崩"这种问题一路漏到用户手上——MicRecorder.start()
整个方法被误删过，录音线程还有个 UnboundLocalError，都没被发现。

这个脚本走完整链路：开真实麦克风、真实 VAD、真实 whisper。
无声环境下也应该正常跑完不崩；有人说话时文字应该进正文框。

用法:
    python tools/live_check.py              # 静默环境，检查不崩
    python tools/live_check.py --speak 15   # 15 秒内对着麦克风说话
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import app as appmod  # noqa: E402
from PySide6.QtWidgets import QApplication, QMessageBox  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--speak", type=int, default=0, help="录音秒数（>0 时请对着麦克风说话）")
    ap.add_argument("--model", default=None)
    args = ap.parse_args()

    errors: list[str] = []
    QMessageBox.critical = staticmethod(
        lambda _s, _t, m: errors.append(str(m)[:200]) or QMessageBox.Ok)
    QMessageBox.warning = staticmethod(lambda *a, **k: QMessageBox.Ok)
    QMessageBox.question = staticmethod(lambda *a, **k: QMessageBox.Yes)

    qapp = QApplication([])
    if args.model:
        import json
        cfg = Path(ROOT) / "config.json"
        d = json.loads(cfg.read_text(encoding="utf-8"))
        d["model"] = args.model
        cfg.write_text(json.dumps(d, indent=2, ensure_ascii=False), encoding="utf-8")

    w = appmod.MainWindow()
    ok, err = _wait_model(w, qapp, timeout=90)
    if not ok:
        print(f"[FAIL] 模型没就绪: {err}")
        return 1
    print(f"[ ok ] 模型就绪 ({w.cfg.model})，设备 {w.device_combo.currentText()}")

    seconds = args.speak or 6
    if args.speak:
        print(f"[ .. ] 开始录音 {seconds} 秒，请说话 …")
    w.start_recording()
    time.sleep(0.6)
    if not w.recording:
        print("[FAIL] start_recording 之后仍处于未录音状态")
        return 1
    print("[ ok ] 录音已启动，线程存活")

    t0 = time.time()
    last_rows = 0
    while time.time() - t0 < seconds:
        time.sleep(0.5)
        qapp.processEvents()
        if w.rows and len(w.rows) != last_rows:
            last_rows = len(w.rows)
            print(f"       已入库 {len(w.rows)} 段，最新：{w.rows[-1][2][:34]}")
    w.stop_recording()
    print("[ .. ] 已请求停止，等待收尾 …")
    t1 = time.time()
    while w.recording and time.time() - t1 < 30:
        time.sleep(0.3)
        qapp.processEvents()

    if w.recording:
        print("[FAIL] 停止后仍显示在录音（收尾卡住）")
        return 1
    print(f"[ ok ] 收尾完成，正文 {len(w.rows)} 段")
    print(f"       状态栏：{w.status_label.text()}")
    print(f"       正文长度：{len(w.text_edit.toPlainText())} 字")
    if errors:
        print(f"[FAIL] 运行中报错 {len(errors)} 次：")
        for e in errors[:3]:
            print("   ", e.replace("\n", " ")[:160])
        return 1
    if args.speak and not w.rows:
        print("[warn] 说了话但一��没识别出来，检查麦克风/灵敏度")
        return 1
    print("[PASS] 完整链路正常")
    w.close()
    return 0


def _wait_model(w, qapp, timeout: float):
    """等后台模型加载完成。

    必须转 Qt 事件循环——加载跑在 QThread 上，结果是通过 signal 回到主线程的，
    光 sleep 不处理事件的话信号永远不会被投递。
    """
    t0 = time.time()
    while time.time() - t0 < timeout:
        qapp.processEvents()
        if w.transcriber is not None and getattr(w.transcriber, "_engine", None) is not None:
            return True, ""
        if "失败" in w.status_label.text():
            return False, w.status_label.text()
        time.sleep(0.1)
    return False, f"超时（状态：{w.status_label.text()}）"


if __name__ == "__main__":
    raise SystemExit(main())
