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


def _method_names(node: ast.ClassDef) -> set[str]:
    """类体里定义的方法名，包含被 @Slot 等装饰器包的。

    ast.walk 能看到装饰器下的 FunctionDef，但只遍历 cls.body 的顶层节点时
    容易漏掉，这里显式取一遍 body。
    """
    out: set[str] = set()
    for item in node.body:
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
            out.add(item.name)
    return out


def _dataclass_fields(node: ast.ClassDef) -> set[str]:
    """@dataclass 装饰的类，其字段在 __init__ 里赋值，静态扫不到。"""
    for dec in node.decorator_list:
        f = dec.func if isinstance(dec, ast.Call) else dec
        if getattr(f, "id", "") == "dataclass" or getattr(f, "attr", "") == "dataclass":
            return {it.target.id for it in node.body
                    if isinstance(it, ast.AnnAssign) and isinstance(it.target, ast.Name)}
    return set()


# Qt / Python 基类提供的方法，无法从本模块静态枚举。
# 必须显式列出——之前用"有外部基类就全放行"，结果 MainWindow 和
# AudioWorker（都继承 QObject）整个被跳过，注入的 bug 一个都抓不到。
_EXTERNAL_MEMBERS = {
    # QWidget / QObject / QDialog / QThread 通用
    "setObjectName", "objectName", "parent", "setParent", "deleteLater",
    "blockSignals", "signalsBlocked", "installEventFilter", "removeEventFilter",
    "setEnabled", "isEnabled", "setVisible", "isVisible", "isHidden", "hide", "show",
    "raise_", "update", "repaint", "setWindowTitle", "windowTitle",
    "setWindowIcon", "setFont", "font", "setStyleSheet", "styleSheet",
    "setLayout", "layout", "setToolTip", "toolTip", "setMinimumSize",
    "setMaximumSize", "setMinimumWidth", "setMaximumWidth", "setMinimumHeight",
    "setMaximumHeight", "resize", "width", "height", "size", "rect",
    "setCentralWidget", "centralWidget", "setWindowFlags", "windowFlags",
    "setFocus", "setFocusPolicy", "focusWidget", "setCursor", "setEnabled",
    "move", "setGeometry", "geometry", "mapToGlobal", "mapFromGlobal",
    "setAttribute", "testAttribute", "setContentsMargins", "activateWindow",
    "isVisibleTo", "isEnabledTo", "window", "close", "showNormal",
    "setWindowState", "windowState", "restoreGeometry", "saveGeometry",
    # QDialog
    "accept", "reject", "done", "exec", "open", "setModal", "result",
    "setWindowModality", "windowModality",
    # QThread
    "isRunning", "wait", "quit", "terminate", "requestInterruption",
    "isInterruptionRequested", "setStackSize", "stackSize",
    # QSize / QRect 等值类型
    "width", "height", "size", "rect", "left", "top", "right", "bottom",
    # 其他
    "closeEvent", "showEvent", "hideEvent", "resizeEvent", "paintEvent",
    "palette", "parentWidget", "addAction", "removeAction", "actions",
    "findChild", "findChildren", "setAccessibleName", "accessibleName",
    "setWindowState", "windowModality", "setModal", "createWindowContainer",
    "setWindowFlag", "setWindowFlags", "windowFlags", "setAccessibleDescription",
    "setLayout", "viewport", "setViewport", "hasHeightForWidth",
    "windowHandle", "createQPainter", "grab", "isActiveWindow",
    "keyPressEvent", "eventFilter", "installEventFilter",
    "processEvents", "exec_", "aboutToQuit", "quit",
}


def _external_allowed(name: str) -> bool:
    """这个名字是不是可能来自外部基类。

    之前这里还写了 name.startswith(("set","is","has")) 的启发式，结果
    self._never_set_x 这种从未赋值的属性因为含 "set" 被放行——注入测试
    就是这么溜过去的。只认白名单里的确切名字。
    """
    return name in _EXTERNAL_MEMBERS


def _inherited_names(class_node: ast.ClassDef) -> set[str]:
    """基类（含基类的基类）上定义的属性和方法。

    只处理本模块内的基类；外部基类（QWidget 之类）一律放行，
    否则满屏都是 Qt 的方法，测试就没有信号了。
    """
    out: set[str] = set()
    for base in class_node.bases:
        name = None
        if isinstance(base, ast.Name):
            name = base.id
        elif isinstance(base, ast.Attribute):
            name = base.attr
        if not name or name[0].isupper():
            continue                       # 外部类（PySide6.QtWidgets.QWidget）
        for node in _MODULE_CLASSES.get(name, []):
            out |= _method_names(node)
            out |= _attrs_assigned(node)
            out |= _signal_names(node)     # ModelLoader 继承 Signals，signal 在基类上
            out |= _inherited_names(node)
    return out


# 本模块里所有类，按类名索引。_inherited_names 要靠它解析基类。
_MODULE_CLASSES: dict[str, list[ast.ClassDef]] = {}


def _signal_names(node: ast.ClassDef) -> set[str]:
    """类体里用 Signal(...) 声明的属性。

    Signal 是类属性声明，运行时由 Qt 的元对象系统创建，静态赋值扫不到。
    不把它们算进去的话 ModelLoader.loaded/failed 会被误报。
    """
    out: set[str] = set()
    for item in node.body:
        if not isinstance(item, ast.Assign):
            continue
        val = item.value
        if not isinstance(val, ast.Call):
            continue
        # Signal 是 import 进来的名字，节点是 ast.Name（id）而不是 ast.Attribute
        fname = getattr(val.func, "id", "") or getattr(val.func, "attr", "")
        if fname == "Signal":
            for t in item.targets:
                if isinstance(t, ast.Name):
                    out.add(t.id)
    return out


def _attrs_assigned(node: ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef) -> set[str]:
    """类体/函数体里真正被赋值的 self.X。

    关键：只看 Store 上下文。旧版本把 self.X 的"读取"也算成已定义，
    结果 self._loss_warned 这种从没赋值的东西永远自证通过。
    """
    out: set[str] = set()
    for n in ast.walk(node):
        if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) \
                and n.value.id == "self" and isinstance(n.ctx, ast.Store):
            out.add(n.attr)
    return out


def test_no_dangling_self_calls():
    """每个类里的 self.xxx() 调用都必须在本类有定义。

    这条对应两类真实事故：
      1. 方法头被批量编辑误删，函数体悬空（v0.1.0 启动即崩）
      2. 跨对象读别人的私有属性（v0.1.2 录音线程 AttributeError）

    旧版本有两个结构性漏洞，让它对这两类 bug 完全无效却显示"通过"：
      · 把整个模块的所有函数名混在一起，不区分类——AudioWorker 里调
        self._on_loss_warn 会被 MainWindow 的同名定义"满足"
      · 把 self.X 的读取也算作"已定义"
    """
    problems: list[str] = []
    for f in _modules():
        tree = ast.parse(f.read_text(encoding="utf-8"))
        # 先填满本模块的类表，基类解析才能用上（ModelLoader 继承 Signals）
        _MODULE_CLASSES.clear()
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                _MODULE_CLASSES.setdefault(node.name, []).append(node)

        for cls in _MODULE_CLASSES.values():
            for node in cls:
                available = (_method_names(node) | _attrs_assigned(node)
                            | _signal_names(node) | _inherited_names(node))
                called: set[str] = set()
                for n in ast.walk(node):
                    if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute):
                        if isinstance(n.func.value, ast.Name) and n.func.value.id == "self":
                            called.add(n.func.attr)
                missing = sorted(c for c in called - available
                                   if not _external_allowed(c))
                # 外部基类提供的一律放行：跨 Qt 的调用无法静态确认
                if missing:
                    problems.append(
                        f"{f.name}:{node.lineno} {node.name} -> {missing}")
    assert not problems, "以下 self 调用在本类没有定义:\n" + "\n".join(problems)


def test_no_unassigned_attribute_reads():
    """读了但从没赋值的 self.X 属性。

    和上一条互补：那条查方法，这一条查实例属性。
    `AudioWorker.run()` 里读 self._loss_warned 就是栽在这里——
    它只在 MainWindow 上赋值，AudioWorker 自己从来没有。
    """
    problems: list[str] = []
    for f in _modules():
        tree = ast.parse(f.read_text(encoding="utf-8"))
        _MODULE_CLASSES.clear()
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                _MODULE_CLASSES.setdefault(node.name, []).append(node)
        for cls in [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]:
            assigned = (_attrs_assigned(cls) | _signal_names(cls)
                        | _inherited_names(cls) | _dataclass_fields(cls)
                        | _method_names(cls))
            read: dict[str, int] = {}
            for n in ast.walk(cls):
                if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) \
                        and n.value.id == "self" and isinstance(n.ctx, ast.Load):
                    read.setdefault(n.attr, n.lineno)
            for name, line in sorted(read.items()):
                if name in assigned or _external_allowed(name):
                    continue
                if name.startswith("__"):
                    continue
                problems.append(f"{f.name}:{line} {cls.name}.{name}")
    assert not problems, ("以下 self 属性被读取但从未在本类赋值:\n"
                          + "\n".join(problems)
                          + "\n（若确实来自基类，请在本类显式初始化一次）")


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
