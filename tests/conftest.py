"""pytest 全局配置。

所有 GUI 测试跑在 offscreen 平台上：CI 的 Windows runner 没有交互式桌面，
不设这个会直接起不来窗口。这里顺带把 QMessageBox 的模态弹窗换成自动应答，
否则任何错误分支都会把测试卡死在等待点击上。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture(scope="session")
def qapp():
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture(autouse=True)
def auto_dismiss_dialogs(monkeypatch):
    """所有模态对话框自动返回第一个按钮，避免测试卡死。"""
    from PySide6.QtWidgets import QDialogButtonBox, QMessageBox

    def _dismiss(*_a, **_k):
        return QMessageBox.Ok

    for name in ("information", "warning", "critical", "question"):
        monkeypatch.setattr(QMessageBox, name, staticmethod(_dismiss))
    monkeypatch.setattr(QMessageBox, "Ok", QMessageBox.Ok, raising=False)
    yield
