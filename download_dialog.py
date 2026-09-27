"""首次运行的下载对话框。

打包后的 exe 不带 tools/fetch_assets.py，所以下载要在程序内完成。
用线程跑下载、signal 回主线程更新进度，界面不卡。
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QObject, Qt, Signal, Slot
from PySide6.QtWidgets import (
    QDialog,
    QLabel,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
)

import assetdl


class _Bridge(QObject):
    progress = Signal(object)   # AssetFetcher
    finished = Signal(object)   # AssetFetcher


def human(n: float) -> str:
    if n >= 1e9:
        return f"{n / 1e9:.2f} GB"
    if n >= 1e6:
        return f"{n / 1e6:.0f} MB"
    if n >= 1e3:
        return f"{n / 1e3:.0f} KB"
    return f"{n:.0f} B"


class DownloadDialog(QDialog):
    def __init__(self, parent, model: str = "small", with_cuda: bool = False):
        super().__init__(parent)
        self.setWindowTitle("下载所需文件")
        self.setMinimumWidth(420)
        self.setWindowFlag(Qt.WindowCloseButtonHint, False)

        self.ok = False
        self._model = model
        self._cuda = with_cuda

        self.bridge = _Bridge()
        self.bridge.progress.connect(self._on_progress)
        self.bridge.finished.connect(self._on_finished)

        root = QVBoxLayout(self)
        self.msg = QLabel("首次运行需要下载 whisper.cpp 运行时和模型，约 500MB。\n"
                          "只需一次，之后完全离线运行。")
        self.msg.setWordWrap(True)
        root.addWidget(self.msg)

        self.bar = QProgressBar()
        self.bar.setRange(0, 100)
        root.addWidget(self.bar)

        self.detail = QLabel("准备中…")
        self.detail.setStyleSheet("color:#888;")
        root.addWidget(self.detail)

        self.cancel_btn = QPushButton("取消")
        self.cancel_btn.clicked.connect(self._on_cancel)
        root.addWidget(self.cancel_btn, 0)

    def start_download(self, root: Path) -> None:
        self.show()
        assetdl.fetch_background(
            root,
            self._model,
            self._cuda,
            on_progress=lambda f: self.bridge.progress.emit(f),
            on_done=lambda f: self.bridge.finished.emit(f),
        )

    def _on_cancel(self) -> None:
        self.msg.setText("已取消。重新启动程序可以再次下载。")
        self.reject()

    @Slot(object)
    def _on_progress(self, f) -> None:
        s = f.step
        self.detail.setText(f"{s.desc}  {human(s.done)} / {human(s.total)}"
                            if s.total else s.desc)
        if s.total:
            self.bar.setValue(int(s.fraction * 100))

    @Slot(object)
    def _on_finished(self, f) -> None:
        if f.error:
            self.msg.setText(f"下载失败：{f.error}")
            self.detail.setText("可以检查网络后重试，或手动下载后放进程序目录。")
            self.cancel_btn.setText("关闭")
            return
        self.ok = True
        self.bar.setValue(100)
        self.accept()
