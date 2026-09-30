"""裁掉 dist/ 里用不到的 Qt 模块和插件。

本项目只用 QtCore / QtGui / QtWidgets 三个模块，但 PyInstaller 会把整个
PySide6 打包进来：Quick、Qml、Pdf、Svg、Network、VirtualKeyboard，还有
软件渲染的 opengl32sw，实测多出约 45MB。

**关键：Qt6*.dll 和 plugins/ 下的插件必须成对删除。**
Qt 启动时枚举 plugins/ 目录逐个加载，缺依赖的插件会让程序直接崩。
第一次裁剪只删了 DLL 没删插件，打出来的包启动即崩——踩的就是这个坑。
依赖关系用 tools/pe_deps.py 静态解析导入表得出（不加载、不执行任何东西）。
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

from pe_deps import imports  # noqa: E402

# 用不到的 Qt 模块。删它们之前要先删依赖它们的插件。
DROP_QT = {
    "Qt6Network.dll", "Qt6OpenGL.dll", "Qt6Pdf.dll",
    "Qt6Qml.dll", "Qt6QmlMeta.dll", "Qt6QmlModels.dll", "Qt6QmlWorkerScript.dll",
    "Qt6Quick.dll", "Qt6Svg.dll", "Qt6VirtualKeyboard.dll",
    "opengl32sw.dll",   # 软件渲染 OpenGL，界面用 raster 够
}

# 必须保留的 Qt 模块，少一个程序起不来。
KEEP_QT = {"Qt6Core.dll", "Qt6Gui.dll", "Qt6Widgets.dll"}

# 依赖上面那些模块的插件。
# 保留 platforms（qwindows.dll 是 Windows 平台插件，必须在）、styles、
# imageformats 里的常规位图格式、translations 的中文。
DROP_PLUGINS = {
    "generic/qtuiotouchplugin.dll",                  # 触摸转鼠标
    "iconengines/qsvgicon.dll",                      # SVG 图标引擎
    "imageformats/qpdf.dll",                         # PDF 读图
    "imageformats/qsvg.dll",                         # SVG 读图
    "networkinformation/qnetworklistmanager.dll",    # 网络状态查询
    "platforminputcontexts/qtvirtualkeyboardplugin.dll",  # 屏幕虚拟键盘
    "tls/qcertonlybackend.dll", "tls/qopensslbackend.dll",
    "tls/qschannelbackend.dll",                      # 证书后端，不联网用不到
}


def plan(internal: Path) -> list[Path]:
    """算出要删哪些文件，但不动手。"""
    qdir = internal / "PySide6"
    out: list[Path] = []
    for name in DROP_QT:
        f = qdir / name
        if f.is_file():
            out.append(f)
    for rel in DROP_PLUGINS:
        f = qdir / "plugins" / rel
        if f.is_file():
            out.append(f)
    tr = qdir / "translations"
    if tr.is_dir():
        for f in tr.glob("*"):
            if f.is_file() and not f.name.startswith(("zh_CN", "zh_")):
                out.append(f)
    return out


def verify(internal: Path) -> list[str]:
    """裁剪后自检：确认没有文件还依赖被删的东西。"""
    qdir = internal / "PySide6"
    present = {p.name.lower() for p in qdir.glob("*.dll")}
    problems: list[str] = []
    for f in qdir.rglob("*.dll"):
        try:
            deps = imports(f)
        except Exception:
            continue
        for d in deps:
            if d.lower().startswith("qt6") and d.lower() not in present:
                problems.append(f"{f.relative_to(qdir).as_posix()} 仍需要 {d}")
    return problems


def trim(internal: Path | None = None, dry_run: bool = False) -> int:
    internal = internal or (ROOT / "dist" / "SoundTrace" / "_internal")
    if not (internal / "PySide6").is_dir():
        print("  dist/ 还不存在，先跑 pyinstaller")
        return 0

    targets = plan(internal)
    drop_set = {f.resolve() for f in targets}
    saved = sum(f.stat().st_size for f in targets)
    if dry_run:
        print(f"  [试运行] 将删除 {len(targets)} 个文件，省下 {saved / 1e6:.0f} MB")
        return saved

    # 自检：确认没有"不在删除列表里"的文件依赖将被删的 DLL。
    # 列表内的文件本来就和 DLL 一起删，依赖它们是正常的。
    present = {p.name.lower() for p in (internal / "PySide6").glob("*.dll")}
    will_drop = {n.lower() for n in DROP_QT}
    orphans = []
    for f in (internal / "PySide6").rglob("*.dll"):
        if f.resolve() in drop_set:
            continue
        try:
            deps = imports(f)
        except Exception:
            continue
        bad = sorted({d for d in deps if d.lower() in will_drop})
        if bad:
            orphans.append((f, bad))
    if orphans:
        print("  自检失败：有文件依赖将被删除的 DLL，但它们不在删除列表里：")
        for f, bad in orphans:
            print(f"    {f.relative_to(internal / 'PySide6').as_posix()} <- {', '.join(bad)}")
        return -1

    for f in targets:
        f.unlink(missing_ok=True)

    problems = verify(internal)
    if problems:
        print("  裁剪后自检发现问题：")
        for p in problems:
            print("   ", p)
        return -1

    print(f"  裁掉 {len(targets)} 个文件，省下 {saved / 1e6:.0f} MB")
    return saved


if __name__ == "__main__":
    trim(dry_run="--dry-run" in sys.argv)
