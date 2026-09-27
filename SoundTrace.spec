# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller 打包配置。

只打包 Python 代码。whisper.cpp 运行时和 GGUF 模型不进 exe：
前者约 40MB 且要按平台区分，后者 466MB 起，而且都是首次运行由
tools/fetch_assets.py 下载的。打包进去既臃肿又难更新。

用法:
    python -m PyInstaller SoundTrace.spec
产物:
    dist/SoundTrace/SoundTrace.exe
"""

import sys
from pathlib import Path

# PyInstaller 6.x 用 collect/subclass 钩子
block_cipher = None

ROOT = Path(SPECPATH).resolve()

a = Analysis(
    [str(ROOT / "app.py")],
    pathex=[str(ROOT)],
    binaries=[],
    datas=[],
    hiddenimports=[
        "opencc",
        "webrtcvad",
        "sounddevice",
        "platforms",
        "branding",
        "theme",
        "settings",
        "widgets",
        "assetdl",
        "download_dialog",
        "platforms",
    ],
    hookspath=[str(ROOT / "hooks")],
    hooksconfig={},
    runtime_hooks=[],
    # Qt 的部分插件是按需加载的，静态分析扫不出来
    excludes=[
        "tkinter",
        "matplotlib",
        "pandas",
        "scipy",
        "notebook",
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="SoundTrace",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,   # UPX 容易被杀软误报
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=str(ROOT / "assets" / "SoundTrace.ico") if (ROOT / "assets" / "SoundTrace.ico").is_file() else None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="SoundTrace",
)
