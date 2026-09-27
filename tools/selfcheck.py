"""环境自检：报告这台机器能不能跑、缺什么、怎么补。

给跨平台用的第一步。在新系统上先跑这个，比直接启动看到一堆报错强。

用法: python tools/selfcheck.py
"""

from __future__ import annotations

import sys

# 控制台编码：GitHub Actions 的 Windows runner 是 cp1252，编不了中文，
# 脚本里的中文提示会直接抛 UnicodeEncodeError。本地中文系统是 GBK 不会
# 暴露这个问题，所以必须在这里兜住。
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import platforms  # noqa: E402

OK, BAD, WARN = "[ ok ]", "[FAIL]", "[warn]"


def main() -> int:
    plat = platforms.current()
    print(f"== SoundTrace 环境自检 ==")
    print(f"平台        {plat.key}  ({sys.platform})")
    print(f"支持开箱即用 {platforms.is_supported()}")
    if not platforms.is_supported():
        print(f"  {WARN} 该平台没有官方预编译包：{plat.build_hint}")

    print(f"\n-- Python --")
    if sys.version_info >= (3, 10):
        print(f"{OK} {sys.version.split()[0]}")
    else:
        print(f"{BAD} {sys.version.split()[0]}  需要 3.10+")

    print(f"\n-- 依赖 --")
    missing = []
    for mod, name in (("numpy", "numpy"), ("sounddevice", "sounddevice"),
                      ("webrtcvad", "webrtcvad"), ("PySide6", "PySide6")):
        try:
            __import__(mod)
            print(f"{OK} {name}")
        except ImportError:
            missing.append(name)
            print(f"{BAD} {name}  缺失")
    try:
        import opencc  # noqa: F401
        print(f"{OK} opencc（繁简统一）")
    except ImportError:
        print(f"{WARN} opencc 缺失，中文输出不会统一转简体")

    print(f"\n-- 麦克风 --")
    try:
        import sounddevice as sd
        devs = [d for d in sd.query_devices() if d.get("max_input_channels", 0) > 0]
        if devs:
            print(f"{OK} 找到 {len(devs)} 个输入设备")
            for d in devs[:4]:
                print(f"       [{d['name']}]")
        else:
            print(f"{BAD} 没有可用的输入设备")
    except Exception as e:
        print(f"{WARN} 枚举失败: {e}")

    print(f"\n-- whisper.cpp 运行时 --")
    main_lib = plat.whisper_path_name
    found = False
    for sub in ("whisper", "whisper_cuda"):
        d = ROOT / "vendor" / sub / "Release"
        if (d / main_lib).is_file():
            print(f"{OK} vendor/{sub}/Release/{main_lib}")
            found = True
    if not found:
        print(f"{BAD} 找不到 {main_lib}")
        print(f"       修复: python tools/fetch_assets.py")

    print(f"\n-- 模型 --")
    models = sorted((ROOT / "models").glob("ggml-*.bin"))
    if models:
        for m in models:
            print(f"{OK} {m.name}  {m.stat().st_size / 1e6:.0f}MB")
    else:
        print(f"{BAD} models/ 下没有模型")
        print(f"       修复: python tools/fetch_assets.py")

    print(f"\n-- 推理设备 --")
    try:
        import device
        info = device.detect("auto")
        print(f"{OK if info.use_gpu else WARN} {info.label} — {info.detail}")
    except Exception as e:
        print(f"{WARN} 探测失败: {e}")

    print(f"\n-- 端到端 --")
    try:
        from pathlib import Path as P
        from whisper_bridge import WhisperEngine
        eng = WhisperEngine(sorted((ROOT / "models").glob("ggml-*.bin"))[0],
                            use_gpu=False, n_threads=4)
        eng.load()
        eng.silence_logs()
        import numpy as np
        r = eng.transcribe(np.zeros(16000, dtype=np.int16), language="zh")
        print(f"{OK} 引擎可用，whisper.cpp {eng.version}")
        eng.free()
    except Exception as e:
        print(f"{BAD} 引擎无法加载: {e}")
        return 1

    if missing:
        print(f"\n缺少依赖: pip install -r requirements.txt")
    print("\n自检完成。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
