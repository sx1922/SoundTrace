"""覆盖 _pyinstaller_hooks_contrib 里那个坏掉的 webrtcvad 钩子。

那个钩子只做 copy_metadata('webrtcvad')，但这个包的分发名是
webrtcvad-wheels，dist-info 叫 webrtcvad_wheels-*.dist-info，
按 import 名找不到元数据，PyInstaller 直接抛 ImportError。
元数据对运行没用，这里换成实际需要的：把 C 扩展和 .py 都带上。
"""
import os
from PyInstaller.utils.hooks import collect_dynamic_libs, get_module_file_attribute

hiddenimports = ["webrtcvad"]
datas = []
try:
    src = get_module_file_attribute("webrtcvad")
    root = os.path.dirname(src)
    if os.path.isfile(src):
        datas.append((src, "."))
    binlibs = collect_dynamic_libs("webrtcvad")
    datas += binlibs
except Exception:
    pass
