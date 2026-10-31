"""离屏渲染 UI 截图，给界面精修做前后对比。

不加载真模型、不碰麦克风：_start_model_load 打补丁，其余状态直接调用
内部方法摆出来。产物存 tests/tmp/ui/，gitignore 内。

用法:
    python tools/ui_preview.py            # 输出全部基线图
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from PySide6.QtGui import QFontDatabase  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

OUT = ROOT / "tests" / "tmp" / "ui"


def _load_cjk_fonts() -> None:
    """offscreen 平台不带任何字体，中文会渲染成方块——手动挂系统字体。"""
    for f in ("msyh.ttc", "msyhbd.ttc", "simhei.ttf", "simsun.ttc"):
        p = Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts" / f
        if p.is_file():
            QFontDatabase.addApplicationFont(str(p))


def snap(win, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    win.grab().save(str(OUT / f"{name}.png"))
    print(f"  {name}.png")


def main() -> int:
    import app as appmod
    from settings import SettingsDialog

    app = QApplication.instance() or QApplication([])
    _load_cjk_fonts()
    appmod.MainWindow._start_model_load = lambda self: None

    for theme_name in ("dark", "light"):
        w = appmod.MainWindow()
        w.set_theme(theme_name)
        w.resize(1000, 720)
        w.show()
        app.processEvents()

        # 1) 空闲空态
        snap(w, f"{theme_name}-idle")

        # 2) 录音中：塞几行结果、预览、电平、录音 pill
        w._on_final_text("今天下午三点我们开个短会，讨论一下语音转文字的优化方向。", 0, 3200)
        w._on_final_text("第二点，断句要自然，不能一句一断，导出格式要支持字幕。", 3200, 7100)
        w._on_partial_text("第三点，模型的体积还要再小一点，部署成本才")
        w.rec_pill.setVisible(True)
        w._update_rec_pill("01:24")
        w.level_bar.set_level(0.22)
        w._on_vad_state("speaking")
        w._on_status("录音中 · GPU 加速 · CUDA")
        w.start_btn.setText("停止录音")
        w.start_btn.setStyleSheet(
            f"background:{w.theme.danger}; color:#ffffff;"
            f"border:1px solid {w.theme.danger}; border-radius:8px;")
        app.processEvents()
        snap(w, f"{theme_name}-recording")

        # 3) 设置对话框
        dlg = SettingsDialog(w.cfg, w)
        dlg.show()
        app.processEvents()
        snap(dlg, f"{theme_name}-settings")
        dlg.close()
        w.close()
        app.processEvents()

    print(f"完成，输出在 {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
