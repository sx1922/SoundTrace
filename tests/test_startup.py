"""启动冒烟测试。

这一组是为 0.1.0 的事故写的：`_check_assets` 方法头被批量编辑整行覆盖，
程序一启动就 AttributeError，而当时没有任何测试覆盖启动路径，包照样发出去了。

这里验证的是"类能不能构造出来、各个方法是不是都真的存在"，属于最基础的
结构检查。把它放在最前面，因为它一失败就说明整个程序起不来。
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def _modules():
    return sorted(p for p in ROOT.glob("*.py") if not p.name.startswith("_"))


def test_all_modules_parse():
    """每个模块都能被解析——批量编辑最容易在这里留下语法错误。"""
    for f in _modules():
        ast.parse(f.read_text(encoding="utf-8"), filename=str(f))


def test_no_dangling_self_calls():
    """每个 self.xxx() 调用都要有对应的定义。

    这条直接对应 0.1.0 的事故：方法头被删掉，函数体变成悬空代码，
    语法合法（因为它还在某个方法体内），但方法本身不存在了。
    纯语法检查抓不到这种，AST 遍历可以。
    """
    problems = []
    for f in _modules():
        tree = ast.parse(f.read_text(encoding="utf-8"))

        called: set[str] = set()
        for n in ast.walk(tree):
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute):
                if isinstance(n.func.value, ast.Name) and n.func.value.id == "self":
                    called.add(n.func.attr)

        available: set[str] = set()
        for n in ast.walk(tree):
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
                available.add(n.name)
            if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) \
                    and n.value.id == "self":
                available.add(n.attr)
            if isinstance(n, ast.ClassDef):
                for base in n.bases:
                    if isinstance(base, ast.Name):
                        available.add(base.id)   # 继承来的（QWidget 之类）放过

        # 去掉明显来自 Qt/Python 的调用
        missing = sorted(c for c in called - available
                         if not c.startswith(("set", "is", "has", "add", "repaint")))
        if missing:
            problems.append(f"{f.name}: {missing}")

    assert not problems, "以下 self 调用没有对应定义:\n" + "\n".join(problems)


def test_main_window_constructs(qapp, tmp_path, monkeypatch):
    """主窗口能真正构造出来。

    用真实的构造流程，但把模型加载换成假的——这一步只关心"起不来"，
    不关心识别质量。构造失败是 0.1.0 的原始症状。
    """
    from PySide6.QtWidgets import QMessageBox

    import app as appmod

    monkeypatch.setattr(QMessageBox, "warning", staticmethod(lambda *a, **k: QMessageBox.Ok))
    monkeypatch.setattr(appmod.MainWindow, "_start_model_load", lambda self: None)

    w = appmod.MainWindow()
    try:
        assert w.windowTitle().startswith("SoundTrace")
        # 关键控件都在
        for name in ("start_btn", "text_edit", "preview", "device_combo",
                     "model_combo", "level_bar", "settings_btn", "export_btn",
                     "clear_btn", "vad_dot", "status_label"):
            assert hasattr(w, name), f"缺少控件 {name}"
        assert w.sens_slider.value() >= 0
    finally:
        w.close()


def test_essential_methods_exist():
    """关键方法不能被误删。这些是用户会点到的路径。"""
    from app import MainWindow

    for name in (
        "_check_assets", "_offer_download", "_refresh_devices", "_refresh_models",
        "start_recording", "stop_recording", "_on_stream_ended",
        "_on_final_text", "_on_partial_text", "clear_all", "copy_all",
        "export", "_to_srt", "set_theme", "open_settings", "_shutdown",
    ):
        assert hasattr(MainWindow, name), f"MainWindow 缺少 {name}"
        assert callable(getattr(MainWindow, name))


def test_no_methods_unreachable():
    """类体里不能出现游离语句（缩进错位会留下这种）。"""
    tree = ast.parse((ROOT / "app.py").read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        for stmt in node.body:
            if isinstance(stmt, ast.Expr):
                # 只允许 docstring
                if not isinstance(stmt.value, ast.Constant):
                    problems = f"类 {node.name} 里有游离表达式 (line {stmt.lineno})"
                    assert not problems, problems
